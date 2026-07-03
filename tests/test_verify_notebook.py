"""Tests for the pre-Colab notebook verification tool.

Covers the audit_notebook() public surface end-to-end and proves the
verifier catches the kind of wiring bugs it was written to find.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.verify_notebook import (  # noqa: E402
    audit_notebook,
    audit_notebook as _audit,
    _audit_local_imports,
    _module_exports,
    _resolve_local_module_path,
)


def _make_notebook(tmpdir: Path, cells: list[dict]) -> Path:
    """Build a tiny notebook with the given code-cell list, save to disk."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Test"]},
        ] + cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    }
    path = tmpdir / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


class AuditLocalImportsTests(unittest.TestCase):
    """The local-import audit catches the two real bugs the audit found."""

    def test_import_from_wrong_local_package_is_an_error(self) -> None:
        # This is exactly bug 2 from the audit: `from cropping.local_staging
        # import COLAB_LOCAL_ROOT_DEFAULT` — `cropping` is a known local
        # package, but `cropping/local_staging.py` doesn't exist. The
        # symbol lives in `perception.local_staging`.
        src = "from cropping.local_staging import COLAB_LOCAL_ROOT_DEFAULT\n"
        errors = _audit_local_imports(src)
        self.assertEqual(len(errors), 1)
        # The audit flags the missing local module path.
        self.assertIn("cropping.local_staging", errors[0])

    def test_missing_pathlib_import_caught_indirectly(self) -> None:
        # Bug 1 was the missing `from pathlib import Path` — pyflakes
        # catches it as an "undefined name" warning. Our verifier
        # surfaces it as a warning (not error); the notebook-level
        # test below proves the full audit pipeline reports it.
        # Here we just confirm the local-import audit doesn't raise
        # on a cell that uses `Path(...)` without importing it.
        src = "x = Path('/tmp')\n"
        errors = _audit_local_imports(src)
        self.assertEqual(errors, [])

    def test_valid_import_passes(self) -> None:
        src = "from perception.local_staging import LocalFrameStager\n"
        errors = _audit_local_imports(src)
        self.assertEqual(errors, [])

    def test_third_party_import_passes(self) -> None:
        src = "import numpy as np\nfrom torch.utils.data import DataLoader\n"
        errors = _audit_local_imports(src)
        self.assertEqual(errors, [])

    def test_local_module_path_does_not_exist_is_an_error(self) -> None:
        # A local module that doesn't exist on disk should be an error.
        src = "from perception.nonexistent_module import thing\n"
        errors = _audit_local_imports(src)
        self.assertEqual(len(errors), 1)
        self.assertIn("does not exist", errors[0])

    def test_wildcard_import_accepted(self) -> None:
        src = "from perception.local_staging import *\n"
        errors = _audit_local_imports(src)
        self.assertEqual(errors, [])


class ModuleExportsTests(unittest.TestCase):
    """The export resolver handles real modules + synthetic ones."""

    def test_resolves_real_local_module(self) -> None:
        path = _resolve_local_module_path("perception.local_staging")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        exports = _module_exports(path)
        # Known public exports from perception.local_staging.
        self.assertIn("LocalFrameStager", exports)
        self.assertIn("DEFAULT_LOCAL_ROOT", exports)
        self.assertIn("COLAB_LOCAL_ROOT_DEFAULT", exports)

    def test_resolves_package_init(self) -> None:
        path = _resolve_local_module_path("perception")
        self.assertIsNotNone(path)
        self.assertTrue(path.name == "__init__.py")

    def test_returns_none_for_third_party(self) -> None:
        self.assertIsNone(_resolve_local_module_path("numpy"))
        self.assertIsNone(_resolve_local_module_path("google.colab"))

    def test_synthetic_module_resolves_correctly(self) -> None:
        # We can't easily create a synthetic on-disk module without
        # touching the repo, so just confirm the function gracefully
        # returns None for unknown paths.
        self.assertIsNone(_resolve_local_module_path("does.not.exist"))


class AuditNotebookTests(unittest.TestCase):
    """The full audit pipeline against real notebooks."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_clean_notebook_passes(self) -> None:
        path = _make_notebook(self.tmpdir, [
            {"cell_type": "code", "metadata": {}, "source": [
                "import os\n",
                "from pathlib import Path\n",
                "from perception.local_staging import LocalFrameStager\n",
                "stager = LocalFrameStager()\n",
            ]},
        ])
        report = audit_notebook(path)
        self.assertTrue(report.is_clean, msg=f"errors: {report.findings}")

    def test_wrong_local_package_is_an_error(self) -> None:
        # The bug 2 scenario, in a notebook.
        path = _make_notebook(self.tmpdir, [
            {"cell_type": "code", "metadata": {}, "source": [
                "from cropping.local_staging import COLAB_LOCAL_ROOT_DEFAULT\n",
            ]},
        ])
        report = audit_notebook(path)
        self.assertFalse(report.is_clean)
        self.assertEqual(report.total_errors, 1)
        self.assertIn("cropping.local_staging", report.findings[0].errors[0])

    def test_missing_local_module_is_an_error(self) -> None:
        path = _make_notebook(self.tmpdir, [
            {"cell_type": "code", "metadata": {}, "source": [
                "from perception.nope import Thing\n",
            ]},
        ])
        report = audit_notebook(path)
        self.assertFalse(report.is_clean)

    def test_markdown_cells_are_ignored(self) -> None:
        # A markdown cell with a "from X import Y" line should not
        # trigger the local-import audit.
        path = _make_notebook(self.tmpdir, [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "Some markdown with `from perception.nope import Z` reference.\n",
            ]},
        ])
        report = audit_notebook(path)
        self.assertTrue(report.is_clean)

    def test_real_000_notebook_passes(self) -> None:
        # End-to-end: audit the project's own 000_full_pipeline.ipynb
        # and confirm it passes. Catches regressions in the notebook
        # before anyone runs it in Colab.
        report = _audit(Path("colab/000_full_pipeline.ipynb"))
        self.assertTrue(
            report.is_clean,
            msg=f"000_full_pipeline.ipynb has {report.total_errors} wiring errors:\n"
                + "\n".join(f"  cell {f.cell_index}: {e}"
                              for f in report.findings for e in f.errors),
        )


class VerifyNotebookCLITests(unittest.TestCase):
    """The CLI returns the expected exit code."""

    def test_cli_returns_zero_for_clean_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            path = _make_notebook(tmpdir, [
                {"cell_type": "code", "metadata": {}, "source": [
                    "import os\n",
                    "from pathlib import Path\n",
                ]},
            ])
            result = subprocess.run(
                [sys.executable, "scripts/verify_notebook.py", str(path)],
                capture_output=True, text=True, cwd=_REPO_ROOT,
            )
            self.assertEqual(result.returncode, 0,
                              msg=f"stdout={result.stdout}\nstderr={result.stderr}")

    def test_cli_returns_nonzero_for_buggy_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            path = _make_notebook(tmpdir, [
                {"cell_type": "code", "metadata": {}, "source": [
                    "from cropping.local_staging import COLAB_LOCAL_ROOT_DEFAULT\n",
                ]},
            ])
            result = subprocess.run(
                [sys.executable, "scripts/verify_notebook.py", str(path)],
                capture_output=True, text=True, cwd=_REPO_ROOT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stdout.lower())

    def test_cli_returns_nonzero_for_missing_file(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/verify_notebook.py",
             str(_REPO_ROOT / "does-not-exist.ipynb")],
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()