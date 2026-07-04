"""Toy fixtures for evaluation-harness tests.

In-memory builders rather than YAML files — tests should not depend
on disk layout for their inputs, and Step 1 does not need to
exercise the YAML loader (that already lives in
``data.manifests``).

The fixtures cover the roles, datasets, and slice-tag combinations
the eval harness must handle:

- debug, train, validate, frozen_unseen_test
- in-scope datasets (urfd, up_fall, le2i, gmdcsa24) and vault
  datasets (omnifall, caucafall, mcfd, fallvision)
- slice tags (lighting, occlusion, multi_person, action_confuser)
  in their known + missing forms
- a tiny frozen-tier manifest that's well-formed under the existing
  ``validate_manifest`` rules
"""

from __future__ import annotations

from data.manifests import ClipRecord, ClipRole, FallLabel, Manifest


# ---------------------------------------------------------------------------
# Clip builders
# ---------------------------------------------------------------------------


def make_clip(
    clip_id: str,
    *,
    dataset: str = "urfd",
    role: ClipRole = ClipRole.TRAIN,
    label: FallLabel = FallLabel.FALL,
    source_path: str | None = None,
    subject_id: str | None = None,
    lighting: str | None = None,
    occlusion: str | None = None,
    multi_person: bool | None = None,
    action_confuser: str | None = None,
) -> ClipRecord:
    """Build a :class:`ClipRecord` with sensible defaults.

    The defaults favour ``urfd / train / fall`` — change the kwargs
    per-test, don't change the defaults unless you intend to update
    every fixture call.
    """
    return ClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        role=role,
        label=label,
        source_path=source_path or f"datasets/{dataset}/{clip_id}.mp4",
        subject_id=subject_id,
        lighting=lighting,
        occlusion=occlusion,
        multi_person=multi_person,
        action_confuser=action_confuser,
    )


def make_manifest(*clips: ClipRecord, schema_version: str = "1.1") -> Manifest:
    """Wrap a sequence of clips in a :class:`Manifest`."""
    return Manifest(schema_version=schema_version, clips=list(clips))


# ---------------------------------------------------------------------------
# Mixed-role manifest used across multiple test files
# ---------------------------------------------------------------------------


def make_mixed_manifest() -> Manifest:
    """Build a manifest with at least one clip per role + slice-tag variety.

    Returns a manifest that exercises every role the guard cares about
    (debug, train, validate, frozen_unseen_test) and every slice tag
    the eval harness wants to aggregate over (lighting, occlusion,
    multi_person, action_confuser).

    Disjointness rules from the existing validator are satisfied:
    - ``train`` and ``validate`` carry different subjects.
    - Vault datasets only appear in ``frozen_unseen_test``.
    """
    return make_manifest(
        # Debug — fast smoke test clips.
        make_clip(
            "urfd-debug-01",
            role=ClipRole.DEBUG,
            label=FallLabel.FALL,
            lighting="daylight",
            occlusion="none",
            multi_person=False,
            action_confuser="none",
        ),
        make_clip(
            "le2i-debug-01",
            dataset="le2i",
            role=ClipRole.DEBUG,
            label=FallLabel.NO_FALL,
            lighting="dim",
            occlusion="partial",
            multi_person=False,
            action_confuser="sitting",
        ),
        # Train — well-formed subject ids, no leakage with validate.
        make_clip(
            "up_fall-train-01",
            dataset="up_fall",
            role=ClipRole.TRAIN,
            subject_id="subject-01",
            lighting="daylight",
            occlusion="none",
        ),
        make_clip(
            "up_fall-train-02",
            dataset="up_fall",
            role=ClipRole.TRAIN,
            label=FallLabel.NO_FALL,
            subject_id="subject-01",
            lighting="daylight",
            action_confuser="sleeping",
        ),
        # Validate — disjoint subject from train.
        make_clip(
            "up_fall-val-01",
            dataset="up_fall",
            role=ClipRole.VALIDATE,
            subject_id="subject-02",
            lighting="low_light",
            occlusion="heavy",
            multi_person=True,
        ),
        # Frozen vault — every kind of vault dataset so the guard
        # can prove it filters them out for training/tuning.
        make_clip(
            "omnifall-vault-01",
            dataset="omnifall",
            role=ClipRole.FROZEN_UNSEEN_TEST,
            label=FallLabel.FALL,
            lighting="low_light",
            occlusion="heavy",
            multi_person=True,
        ),
        make_clip(
            "caucafall-vault-01",
            dataset="caucafall",
            role=ClipRole.FROZEN_UNSEEN_TEST,
            label=FallLabel.NO_FALL,
            lighting="mixed",
            action_confuser="exercising",
        ),
        make_clip(
            "mcfd-vault-01",
            dataset="mcfd",
            role=ClipRole.FROZEN_UNSEEN_TEST,
            lighting="daylight",
            occlusion="partial",
        ),
        make_clip(
            "fallvision-vault-01",
            dataset="fallvision",
            role=ClipRole.FROZEN_UNSEEN_TEST,
            lighting="dim",
            occlusion="heavy",
            multi_person=True,
            action_confuser="crawling",
        ),
    )


__all__: tuple[str, ...] = (
    "make_clip",
    "make_manifest",
    "make_mixed_manifest",
)
