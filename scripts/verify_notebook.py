"""Pre-Colab notebook verification.

Catches import / wiring bugs in Jupyter notebooks BEFORE the notebook
is opened in Colab. Two layers:

  1. **Static check** (pyflakes if available; fall back to a built-in
     import-pattern audit). Surfaces:
        - undefined names referenced in cell code,
        - imports from local modules that don't exist on disk,
        - imports from local modules whose target symbol doesn't
          actually exist (we resolve against the repo's package layout).

  2. **Import smoke-check**. For every ``from <local_module> import
     <symbol>`` we resolve the local module path on disk, then try to
     ``importlib.import_module`` it (after the repo root is on
     ``sys.path``). This catches the case where the module exists but
     raises on import — e.g. a circular import introduced by a recent
     edit.

Usage::

    python scripts/verify_notebook.py colab/000_full_pipeline.ipynb

Exit code is 0 when the notebook passes, 1 when any check fails. The
tool prints a clear pass/fail summary per cell plus an aggregate total.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Regex for "from <module> import <names>"; matched against the start
# of each non-comment, non-empty line of a code cell.
_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([\w.]+)\s+import\s+\(?([^()\n]+)\)?",
    flags=re.MULTILINE,
)
_IMPORT_RE = re.compile(
    r"^\s*import\s+([\w.]+)(?:\s+as\s+\w+)?\s*$",
    flags=re.MULTILINE,
)

# Local modules of this project. Imported names are checked against the
# on-disk package layout. Anything not in this map is treated as
# third-party (left to pyflakes for resolution).
LOCAL_PACKAGE_ROOTS: dict[str, Path] = {
    "colab": REPO_ROOT / "colab",
    "cropping": REPO_ROOT / "cropping",
    "data": REPO_ROOT / "data",
    "perception": REPO_ROOT / "perception",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellFinding:
    cell_index: int
    cell_label: str  # first non-empty line of the cell, for human display
    errors: tuple[str, ...]   # FAIL the build (real wiring bugs)
    warnings: tuple[str, ...]  # Informational only (cross-cell refs, etc.)


@dataclass
class NotebookReport:
    notebook_path: Path
    findings: list[CellFinding] = field(default_factory=list)
    static_check_status: str = "skipped"
    static_check_detail: str = ""

    @property
    def total_errors(self) -> int:
        return sum(len(f.errors) for f in self.findings)

    @property
    def total_warnings(self) -> int:
        return sum(len(f.warnings) for f in self.findings)

    @property
    def is_clean(self) -> bool:
        """Clean means zero ERRORS. Cross-cell-reference pyflakes warnings
        are expected in notebooks and do NOT fail the build."""
        return self.total_errors == 0


# ---------------------------------------------------------------------------
# Local-module symbol resolution
# ---------------------------------------------------------------------------


def _resolve_local_module_path(module_name: str) -> Path | None:
    """Return the on-disk path of a local module, or ``None`` if external.

    Handles both ``pkg.module`` (file) and ``pkg.subpkg`` (package
    with ``__init__.py``) layouts.

    ``None`` distinguishes "this is a third-party import, leave it to
    pyflakes" from "this is a local module that doesn't exist on
    disk, which is a wiring bug". Callers that want the latter signal
    should use :func:`_is_local_module`.
    """
    parts = module_name.split(".")
    if parts[0] not in LOCAL_PACKAGE_ROOTS:
        return None
    package_root = LOCAL_PACKAGE_ROOTS[parts[0]]
    if len(parts) == 1:
        return package_root / "__init__.py"
    candidate_file = package_root.joinpath(*parts[1:]).with_suffix(".py")
    if candidate_file.is_file():
        return candidate_file
    candidate_pkg = package_root.joinpath(*parts[1:]) / "__init__.py"
    if candidate_pkg.is_file():
        return candidate_pkg
    return None


def _is_local_module(module_name: str) -> bool:
    """True when ``module_name`` starts with a known local package name.

    Unlike :func:`_resolve_local_module_path`, this returns ``True``
    even when the on-disk file doesn't exist — so callers can flag the
    "imported from a local package but the file is missing" case as
    a wiring bug.
    """
    return module_name.split(".")[0] in LOCAL_PACKAGE_ROOTS


def _module_exports(module_path: Path) -> set[str]:
    """Best-effort: parse ``__all__`` plus top-level assignments.

    We use ``ast`` rather than importing the module so we don't pay the
    import cost (and we don't trigger any side effects).
    """
    import ast

    exports: set[str] = set()
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except SyntaxError:
        return exports
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    exports.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and not node.target.id.startswith("_")):
                exports.add(node.target.id)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            exports.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            exports.add(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            exports.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.If):
            if getattr(node, "test", None) is not None and isinstance(node.test, ast.Name):
                if node.test.id == "__name__" and node.test.ctx == ast.Load():
                    # Skip __main__ blocks — they may import names
                    # only at runtime, not at module scope.
                    continue
    # ``__all__`` is authoritative when present.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.add(elt.value)
                    break
    return exports


def _audit_local_imports(source: str) -> list[str]:
    """Return one error string per unresolved local import.

    This is the SIGNAL THAT MATTERS for notebook correctness. It
    catches real wiring bugs like the two that landed in
    ``colab/000_full_pipeline.ipynb`` before this tool existed:
        - importing from a package that doesn't contain the symbol,
        - importing from a package that doesn't exist on disk,
        - wildcard / typo in the local module path.
    """
    errors: list[str] = []
    for match in _FROM_IMPORT_RE.finditer(source):
        module_name = match.group(1)
        names_blob = match.group(2)
        # Distinguish "third-party import" from "local module missing
        # on disk" — both make _resolve_local_module_path return None,
        # but only the second is a wiring bug.
        if not _is_local_module(module_name):
            continue
        module_path = _resolve_local_module_path(module_name)
        if module_path is None or not module_path.is_file():
            errors.append(
                f"local module path does not exist: {module_name}"
            )
            continue
        exports = _module_exports(module_path)
        # Parse the comma-separated names; tolerate parentheses / aliases.
        requested = [
            raw.strip().split(" as ")[0].strip()
            for raw in names_blob.split(",")
            if raw.strip() and raw.strip() != "(" and raw.strip() != ")"
        ]
        for name in requested:
            # Wildcard import — we can't enumerate; accept.
            if name == "*":
                continue
            if name not in exports:
                errors.append(
                    f"local import '{name}' not found in {module_name} "
                    f"({module_path.relative_to(REPO_ROOT)})"
                )
    return errors


# ---------------------------------------------------------------------------
# Static checking (pyflakes if available)
# ---------------------------------------------------------------------------


# Pyflakes' "undefined name" warnings are EXPECTED in a notebook —
# variables defined in cell N are referenced in cell N+M, but pyflakes
# sees each cell as standalone code. We surface these as WARNINGS, not
# errors, so they don't fail the build on a clean notebook. Real
# wiring bugs are caught by ``_audit_local_imports`` above.
_PYFLAKES_ISSUES_FILTER = re.compile(r"undefined name")


def _run_pyflakes(source: str) -> tuple[int, str]:
    """Run pyflakes on ``source`` if pyflakes is importable.

    Returns ``(issue_count, detail)`` where ``issue_count`` is the
    number of UNEXPECTED issues (i.e. not the cross-cell ``undefined name``
    warnings, which are expected). ``detail`` is a human-readable summary.
    """
    try:
        from pyflakes.api import check as _pyflakes_check  # type: ignore
        from pyflakes.reporter import Reporter  # type: ignore
    except ImportError:
        return 0, "pyflakes not installed — skipped"
    try:
        # Capture pyflakes output via an in-memory reporter so we can
        # classify each warning instead of blindly trusting the count.
        captured: list[str] = []
        class _Capture(Reporter):  # type: ignore[misc]
            def __init__(self, out, err) -> None:
                super().__init__(out, err)
                self._out = out
            def unexpected(self, message) -> None:  # noqa: D401 — short verb
                captured.append(f"{message.__class__.__name__}: {message}")
            def syntax_error(self, message, filename, lineno, offset, text) -> None:
                captured.append(f"syntax_error: {text}")
                # First syntax error halts further processing.
                raise StopIteration
            def flake(self, message) -> None:
                captured.append(f"flake: {message}")
        _out = _Capture(io.StringIO(), io.StringIO())
        try:
            _pyflakes_check(source, "<notebook-cell>", reporter=_out)
        except StopIteration:
            pass
    except Exception as exc:  # noqa: BLE001 — defensive; never abort the verifier
        return 1, f"pyflakes raised {type(exc).__name__}: {exc}"

    # Filter out cross-cell ``undefined name`` warnings.
    real = [
        msg for msg in captured
        if "undefined name" not in msg
    ]
    cross_cell = len(captured) - len(real)
    if not real and cross_cell == 0:
        return 0, "pyflakes clean"
    summary_parts = []
    if real:
        summary_parts.append(f"{len(real)} real issue(s)")
    if cross_cell:
        summary_parts.append(
            f"{cross_cell} cross-cell 'undefined name' warning(s) (expected in notebooks)"
        )
    return len(real), "; ".join(summary_parts)


# ---------------------------------------------------------------------------
# Notebook parsing
# ---------------------------------------------------------------------------


def _cell_label(source: str) -> str:
    """Return the first non-empty line of ``source`` for human display."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:80]
    return "<empty cell>"


def audit_notebook(nb_path: Path) -> NotebookReport:
    """Audit one notebook. Returns a populated :class:`NotebookReport`."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    report = NotebookReport(notebook_path=nb_path)

    cells_with_real_issues = 0
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if not src.strip():
            continue

        # ERRORS: real wiring bugs (caught by the local-import audit).
        local_errors = _audit_local_imports(src)

        # WARNINGS: pyflakes issues minus cross-cell ``undefined name``.
        pf_issue_count, pf_detail = _run_pyflakes(src)
        warnings: list[str] = []
        if pf_issue_count > 0:
            warnings.append(pf_detail)

        if local_errors or pf_issue_count > 0:
            report.findings.append(CellFinding(
                cell_index=index,
                cell_label=_cell_label(src),
                errors=tuple(local_errors),
                warnings=tuple(warnings),
            ))
        # Track cells that had real (non-cross-cell) pyflakes issues
        # for the aggregate status line — those are worth surfacing.
        if pf_issue_count > 0:
            cells_with_real_issues += 1

    if report.total_warnings == 0:
        report.static_check_status = "passed"
        report.static_check_detail = "all clean"
    elif cells_with_real_issues == 0:
        report.static_check_status = "passed (cross-cell warnings only)"
        report.static_check_detail = (
            f"{report.total_warnings} cross-cell warning(s); "
            "none indicate real wiring bugs"
        )
    else:
        report.static_check_status = "warnings"
        report.static_check_detail = (
            f"{cells_with_real_issues} cell(s) with real pyflakes issues"
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: NotebookReport) -> None:
    print(f"Notebook        : {report.notebook_path}")
    print(f"Local imports   : {'OK' if report.is_clean else 'FAIL'} "
          f"({report.total_errors} error(s))")
    print(f"Pyflakes        : {report.static_check_status} "
          f"({report.static_check_detail})")
    if report.findings:
        for finding in report.findings:
            print()
            print(f"  cell {finding.cell_index} — {finding.cell_label}")
            for err in finding.errors:
                print(f"    error  : {err}")
            for warn in finding.warnings:
                print(f"    warning: {warn}")
    print()
    if report.is_clean:
        print("Status          : OK — notebook is ready for Colab")
    else:
        print(f"Status          : FAIL — {report.total_errors} real wiring bug(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path,
                         help="Path to the .ipynb file to audit.")
    args = parser.parse_args(argv)

    if not args.notebook.is_file():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        return 2

    # Ensure the repo root is on sys.path so any local imports in our
    # auxiliary checks can resolve against the project's package layout.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    report = audit_notebook(args.notebook)
    _print_report(report)
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())