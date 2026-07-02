"""Dataset manifest schema, loader, and validator for the fall-detection project.

Public surface:
    - :class:`ClipRole`, :class:`FallLabel`, :class:`SliceTag` — typed enums.
    - :class:`ClipRecord` — one manifest row.
    - :class:`Manifest` — the full collection of clips + version metadata.
    - :func:`load_manifest` — parse YAML or JSON from disk.
    - :func:`validate_manifest` — run the full validation suite, return a structured report.
    - :data:`FROZEN_VAULT_DATASETS` — the unseen-test wall.

Schema version: ``1.1``.

The schema enforces **clip-level role locking**, not dataset-level roles:
a single dataset (e.g. ``le2i``) may appear under multiple roles, but each
individual clip is assigned exactly one role at the row level. This is the
shape required by the PRD's *role-locked dataset splits* decision.

Leakage enforcement (Issue 001 review):
    - ``clip_id`` is unique across the whole manifest.
    - Train and validate never share the same ``source_path`` (raw video file).
    - Train and validate never share the same ``subject_id`` when one is set.
    - Frozen-vault datasets only appear in ``frozen_unseen_test``.

The validator is intentionally split into named check functions so each rule
fails or passes independently and can be exercised by the test suite in
isolation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ClipRole(str, Enum):
    """The four roles a clip can be assigned to.

    Strings (not ints) so the manifest is human-readable on disk and grep-friendly.
    """

    DEBUG = "debug"
    TRAIN = "train"
    VALIDATE = "validate"
    FROZEN_UNSEEN_TEST = "frozen_unseen_test"


# Schema version this loader knows how to interpret. Bump the loader first
# when introducing a new version; bump the manifest second; the schema-
# version check then forces the two to stay in sync.
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0", "1.1"})
CURRENT_SCHEMA_VERSION: str = "1.1"


# Datasets that may legitimately appear in debug AND train/validate.
# Documented explicitly so the "leakage" alarm doesn't fire on the
# cross-listed Le2i / GMDCSA-24 clips.
CROSS_LISTED_DATASETS: frozenset[str] = frozenset({"le2i", "gmdcsa24"})

# The hard vault — these datasets must NEVER appear outside ``frozen_unseen_test``.
# Any clip from one of these datasets showing up in debug/train/validate is a
# test-integrity failure, not just a leakage warning.
FROZEN_VAULT_DATASETS: frozenset[str] = frozenset({
    "omnifall",
    "caucafall",
    "mcfd",
    "fallvision",
})

# Datasets that are usable in debug / train / validate roles.
IN_SCOPE_DATASETS: frozenset[str] = frozenset({
    "urfd",
    "up_fall",
    "le2i",
    "gmdcsa24",
})

# All known datasets, for fast membership / error-message reporting.
ALL_KNOWN_DATASETS: frozenset[str] = (
    IN_SCOPE_DATASETS | FROZEN_VAULT_DATASETS
)


class FallLabel(str, Enum):
    """Clip-level supervision label."""

    FALL = "fall"
    NO_FALL = "no_fall"


# ---------------------------------------------------------------------------
# Slice tags (Issue 004 — slice-based evaluation)
# ---------------------------------------------------------------------------


class LightingCondition(str, Enum):
    """Coarse lighting buckets; expand deliberately, don't auto-accept new values."""

    DAYLIGHT = "daylight"
    DIM = "dim"
    LOW_LIGHT = "low_light"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class OcclusionLevel(str, Enum):
    """How much of the subject is occluded during the clip."""

    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class ActionConfuser(str, Enum):
    """Confuser actions that look fall-like but aren't.

    Sourced from PRD user stories 18 / 29 ("sitting / sleeping / resting /
    exercising / crawling"). ``NONE`` means "not a confuser scene" — i.e.
    the clip is a clean fall or a clean non-fall.
    """

    NONE = "none"
    SITTING = "sitting"
    SLEEPING = "sleeping"
    RESTING = "resting"
    EXERCISING = "exercising"
    CRAWLING = "crawling"


# Validators look values up in these sets; anything outside is a warning,
# not a hard error — researchers may legitimately need a new bucket, and
# adding one should be a one-line change here rather than a schema bump.
LIGHTING_VALUES: frozenset[str] = frozenset(v.value for v in LightingCondition)
OCCLUSION_VALUES: frozenset[str] = frozenset(v.value for v in OcclusionLevel)
ACTION_CONFUSER_VALUES: frozenset[str] = frozenset(v.value for v in ActionConfuser)


# ---------------------------------------------------------------------------
# Clip record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipRecord:
    """One row of the manifest.

    Required fields: ``clip_id``, ``dataset``, ``role``, ``label``,
    ``source_path``. Everything else is optional and may be ``None`` for
    placeholder rows — slice tags only become useful once Issue 004
    (eval harness) starts aggregating by them.

    All fields are keyword-only so callers can't accidentally swap e.g.
    ``label`` and ``role`` positionally.
    """

    clip_id: str
    dataset: str
    role: ClipRole
    label: FallLabel
    source_path: str

    # Optional provenance / metadata. ``subject_id`` is required for the
    # subject-level leakage check (train vs validate disjointness) to fire;
    # if it's None for both halves of a comparison, that comparison just
    # skips — the rule never blocks a manifest that simply lacks subject IDs.
    subject_id: str | None = None
    duration_sec: float | None = None
    frame_count: int | None = None
    notes: str | None = None

    # Slice tags (Issue 004 — slice-based evaluation).
    # All nullable so existing rows from schema 1.0 keep validating.
    lighting: str | None = None
    occlusion: str | None = None
    multi_person: bool | None = None
    action_confuser: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON/YAML-friendly dict, normalising enums to strings."""
        raw = asdict(self)
        raw["role"] = self.role.value
        raw["label"] = self.label.value
        return raw


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """A versioned collection of :class:`ClipRecord` rows."""

    schema_version: str
    clips: list[ClipRecord] = field(default_factory=list)

    def by_role(self, role: ClipRole) -> list[ClipRecord]:
        """Return clips assigned to ``role``. Order is preserved."""
        return [c for c in self.clips if c.role is role]

    def to_serialisable(self) -> dict[str, object]:
        """Serialise the full manifest for YAML/JSON dump."""
        return {
            "schema_version": self.schema_version,
            "clips": [c.to_dict() for c in self.clips],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str) -> Manifest:
    """Load a manifest from a YAML or JSON file.

    The format is auto-detected from the file extension. Both formats
    round-trip the same dict shape. ``pyyaml`` is in the approved stack
    (see ``colab/setup.py``); if PyYAML is missing we still try JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML manifests and is missing. "
                "Install it (it's in the approved stack) or convert the file to JSON."
            ) from exc
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported manifest extension {path.suffix!r}; expected .yaml, .yml, or .json."
        )

    if not isinstance(data, Mapping):
        raise ValueError("Manifest root must be a mapping with 'schema_version' and 'clips'.")

    schema_version = str(data.get("schema_version", ""))
    raw_clips = data.get("clips", [])
    if not isinstance(raw_clips, list):
        raise ValueError("Manifest 'clips' must be a list.")

    clips: list[ClipRecord] = []
    for index, raw in enumerate(raw_clips):
        clips.append(_parse_clip(raw, index))

    return Manifest(schema_version=schema_version, clips=clips)


def _parse_clip(raw: object, index: int) -> ClipRecord:
    """Build a :class:`ClipRecord` from one raw dict, with clear error context."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"clips[{index}] must be a mapping, got {type(raw).__name__}.")

    def _required(key: str) -> object:
        if key not in raw or raw[key] in (None, ""):
            raise ValueError(f"clips[{index}] missing required field {key!r}.")
        return raw[key]

    try:
        return ClipRecord(
            clip_id=str(_required("clip_id")),
            dataset=str(_required("dataset")).strip().lower(),
            role=ClipRole(str(_required("role")).strip().lower()),
            label=FallLabel(str(_required("label")).strip().lower()),
            source_path=str(_required("source_path")),
            subject_id=_optional_str(raw.get("subject_id")),
            duration_sec=_optional_float(raw.get("duration_sec")),
            frame_count=_optional_int(raw.get("frame_count")),
            notes=_optional_str(raw.get("notes")),
            lighting=_optional_str_normalised(raw.get("lighting")),
            occlusion=_optional_str_normalised(raw.get("occlusion")),
            multi_person=_optional_bool(raw.get("multi_person")),
            action_confuser=_optional_str_normalised(raw.get("action_confuser")),
        )
    except ValueError as exc:
        raise ValueError(f"clips[{index}]: {exc}") from exc


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str_normalised(value: object) -> str | None:
    """Same as :func:`_optional_str` but lower-cases / strips so YAML and JSON
    both round-trip the same enum-style value regardless of case."""
    if value is None:
        return None
    return str(value).strip().lower()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    """Parse a YAML/JSON bool, accepting ``True``/``False`` and ``true``/``false``.

    Returns ``None`` for ``None`` or empty input so a missing field stays
    missing rather than defaulting to ``False`` (which would falsely imply
    "single-person clip confirmed").
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of :func:`validate_manifest`.

    ``errors`` are hard failures (manifest is unsafe to use); ``warnings`` are
    things the caller should know about but that do not invalidate the file.
    Both are lists of human-readable strings so the report can be printed or
    asserted on directly.
    """

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_manifest(manifest: Manifest) -> ValidationReport:
    """Run the full validation suite against ``manifest``.

    Each rule is its own function so failures isolate to the responsible
    check. Order is intentional: shape / required-field checks first (so the
    later checks can assume well-formed records), then leakage rules.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest.schema_version:
        errors.append("Manifest has no 'schema_version'; refusing to validate.")

    _check_schema_version_supported(manifest.schema_version, errors)

    _check_required_fields(manifest.clips, errors)
    _check_every_clip_has_role(manifest.clips, errors)
    _check_every_clip_has_label(manifest.clips, errors)
    _check_every_clip_has_dataset(manifest.clips, errors)
    _check_known_datasets(manifest.clips, warnings)
    _check_no_duplicate_clip_ids(manifest.clips, errors)
    _check_train_validate_disjoint(manifest.clips, errors)
    _check_source_path_disjoint(manifest.clips, errors)
    _check_subject_id_disjoint(manifest.clips, errors)
    _check_frozen_vault_isolation(manifest.clips, errors)
    _check_slice_tags(manifest.clips, warnings)
    _check_cross_listed_datasets_recorded_clearly(manifest, warnings)

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_required_fields(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Every clip has the four required fields populated.

    The dataclass constructor enforces this for ``clip_id``/``dataset``/
    ``role``/``label``/``source_path``; this check is a defence-in-depth
    guard against future schema additions where a field becomes required.
    """
    for clip in clips:
        missing = [
            field_name for field_name in ("clip_id", "dataset", "role", "label", "source_path")
            if not getattr(clip, field_name)
        ]
        if missing:
            errors.append(
                f"clip {clip.clip_id or '<no id>'!r}: missing required field(s): {missing}"
            )


def _check_every_clip_has_role(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Every clip's ``role`` is one of the four allowed values."""
    allowed = {r.value for r in ClipRole}
    for clip in clips:
        if clip.role.value not in allowed:
            errors.append(
                f"clip {clip.clip_id!r}: role {clip.role.value!r} is not one of {sorted(allowed)}."
            )


def _check_every_clip_has_label(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Every clip's ``label`` is ``fall`` or ``no_fall``."""
    allowed = {l.value for l in FallLabel}
    for clip in clips:
        if clip.label.value not in allowed:
            errors.append(
                f"clip {clip.clip_id!r}: label {clip.label.value!r} is not one of {sorted(allowed)}."
            )


def _check_every_clip_has_dataset(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Every clip records a non-empty dataset (provenance)."""
    for clip in clips:
        if not clip.dataset:
            errors.append(f"clip {clip.clip_id!r}: dataset provenance is empty.")


def _check_known_datasets(clips: Iterable[ClipRecord], warnings: list[str]) -> None:
    """Flag datasets that are not in :data:`ALL_KNOWN_DATASETS` so typos surface."""
    seen_unknown: set[str] = set()
    for clip in clips:
        if clip.dataset not in ALL_KNOWN_DATASETS and clip.dataset not in seen_unknown:
            seen_unknown.add(clip.dataset)
            warnings.append(
                f"clip {clip.clip_id!r}: dataset {clip.dataset!r} is not in ALL_KNOWN_DATASETS "
                f"({sorted(ALL_KNOWN_DATASETS)}). Add it intentionally if this is a new source."
            )


def _check_no_duplicate_clip_ids(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Every ``clip_id`` is unique across the manifest."""
    counts = Counter(c.clip_id for c in clips)
    duplicates = sorted(cid for cid, n in counts.items() if n > 1)
    for dup in duplicates:
        errors.append(
            f"duplicate clip_id {dup!r} appears {counts[dup]} times — clip IDs must be unique."
        )


def _check_train_validate_disjoint(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """A clip cannot be in both ``train`` and ``validate`` simultaneously."""
    # A clip can appear in many roles for cross-listed datasets (debug + train
    # is legal); the hard wall is ONLY between train and validate.
    role_by_id: dict[str, set[str]] = {}
    for clip in clips:
        role_by_id.setdefault(clip.clip_id, set()).add(clip.role.value)

    conflicts = sorted(
        cid for cid, roles in role_by_id.items()
        if "train" in roles and "validate" in roles
    )
    for cid in conflicts:
        errors.append(
            f"clip_id {cid!r} appears in both 'train' and 'validate' — these roles must be disjoint."
        )


def _check_source_path_disjoint(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Train and validate must not share the same ``source_path``.

    Two clips with different ``clip_id``s can still leak the same raw video
    file into both train and validate — for example, two annotators writing
    different ``clip_id``s for overlapping windows of the same source video.
    The duplicate-``clip_id`` check does not catch this; this check does.
    """
    train_paths = {c.source_path for c in clips if c.role is ClipRole.TRAIN}
    validate_paths = {c.source_path for c in clips if c.role is ClipRole.VALIDATE}

    conflicts = sorted(train_paths & validate_paths)
    for path in conflicts:
        errors.append(
            f"source_path {path!r} appears in both 'train' and 'validate' — "
            f"the same raw video file must not span train and validate."
        )


def _check_subject_id_disjoint(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """Train and validate must not share the same ``subject_id`` when present.

    A subject (person) appearing in both train and validate lets the model
    memorise subject-specific cues rather than learning the fall pattern —
    even when the raw videos are different. When ``subject_id`` is ``None``
    on both sides of a comparison there is nothing to check, so a manifest
    with sparse subject IDs will not be blocked — but the rule fires
    immediately the moment a conflict exists.
    """
    train_subjects = {
        c.subject_id for c in clips
        if c.role is ClipRole.TRAIN and c.subject_id is not None
    }
    validate_subjects = {
        c.subject_id for c in clips
        if c.role is ClipRole.VALIDATE and c.subject_id is not None
    }

    conflicts = sorted(train_subjects & validate_subjects)
    for subject in conflicts:
        errors.append(
            f"subject_id {subject!r} appears in both 'train' and 'validate' — "
            f"the same person must not span train and validate."
        )


def _check_slice_tags(clips: Iterable[ClipRecord], warnings: list[str]) -> None:
    """Slice tags are free-form on the wire but typed in the schema.

    When a clip carries a slice tag, the value should be in the known enum
    set. Unknown values are *warnings*, not errors — adding a new bucket
    should be a one-line change here rather than a schema bump. Missing
    tags are not warned about (placeholder rows are legal until Issue 004
    starts aggregating by slice).
    """
    for clip in clips:
        if clip.lighting is not None and clip.lighting not in LIGHTING_VALUES:
            warnings.append(
                f"clip {clip.clip_id!r}: lighting {clip.lighting!r} is not one of "
                f"{sorted(LIGHTING_VALUES)} — add it to LightingCondition if this is intentional."
            )
        if clip.occlusion is not None and clip.occlusion not in OCCLUSION_VALUES:
            warnings.append(
                f"clip {clip.clip_id!r}: occlusion {clip.occlusion!r} is not one of "
                f"{sorted(OCCLUSION_VALUES)} — add it to OcclusionLevel if this is intentional."
            )
        if clip.action_confuser is not None and clip.action_confuser not in ACTION_CONFUSER_VALUES:
            warnings.append(
                f"clip {clip.clip_id!r}: action_confuser {clip.action_confuser!r} is not one of "
                f"{sorted(ACTION_CONFUSER_VALUES)} — add it to ActionConfuser if this is intentional."
            )


def _check_schema_version_supported(version: str, errors: list[str]) -> None:
    """Reject schema versions this loader doesn't know how to interpret.

    Adding a new schema version means writing a migration / loader path
    here; bumping the loader without bumping the manifest should be a
    deliberate choice, not a silent fall-through.
    """
    supported = SUPPORTED_SCHEMA_VERSIONS
    if version not in supported:
        errors.append(
            f"Manifest schema_version {version!r} is not supported by this loader. "
            f"Supported: {sorted(supported)}."
        )


def _check_frozen_vault_isolation(clips: Iterable[ClipRecord], errors: list[str]) -> None:
    """No clip from a vault dataset may appear outside ``frozen_unseen_test``.

    Vault datasets (OmniFall, CAUCAFall, MCFD, FallVision) are the unseen-test
    wall. The validator catches:
        - any clip from a vault dataset assigned a non-vault role,
        - any clip assigned ``frozen_unseen_test`` from a non-vault dataset
          (which would also indicate the manifest is mis-tagged).
    """
    for clip in clips:
        is_vault_dataset = clip.dataset in FROZEN_VAULT_DATASETS
        is_vault_role = clip.role is ClipRole.FROZEN_UNSEEN_TEST

        if is_vault_dataset and not is_vault_role:
            errors.append(
                f"clip {clip.clip_id!r}: dataset {clip.dataset!r} is a frozen-vault dataset "
                f"and must only appear in role 'frozen_unseen_test', not {clip.role.value!r}."
            )
        if is_vault_role and not is_vault_dataset:
            errors.append(
                f"clip {clip.clip_id!r}: role 'frozen_unseen_test' is reserved for vault "
                f"datasets only; got non-vault dataset {clip.dataset!r}."
            )


def _check_cross_listed_datasets_recorded_clearly(
    manifest: Manifest, warnings: list[str]
) -> None:
    """Warn when a cross-listed dataset (Le2i / GMDCSA-24) has no debug clips.

    These datasets are *expected* to appear in both debug and train/validate
    per the PRD. The check exists to remind the author to populate debug
    examples rather than silently skipping them.
    """
    debug_datasets = {c.dataset for c in manifest.by_role(ClipRole.DEBUG)}
    for ds in CROSS_LISTED_DATASETS:
        if ds not in debug_datasets:
            warnings.append(
                f"cross-listed dataset {ds!r} has no debug clips — add at least one so the "
                f"debug tier exercises both shapes (debug and train/validate) for this dataset."
            )


__all__: tuple[str, ...] = (
    "ClipRole",
    "FallLabel",
    "LightingCondition",
    "OcclusionLevel",
    "ActionConfuser",
    "CROSS_LISTED_DATASETS",
    "FROZEN_VAULT_DATASETS",
    "IN_SCOPE_DATASETS",
    "ALL_KNOWN_DATASETS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "CURRENT_SCHEMA_VERSION",
    "LIGHTING_VALUES",
    "OCCLUSION_VALUES",
    "ACTION_CONFUSER_VALUES",
    "ClipRecord",
    "Manifest",
    "load_manifest",
    "ValidationReport",
    "validate_manifest",
)