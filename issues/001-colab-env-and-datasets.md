## Parent PRD

`issues/prd.md`

## What to build

Stand up the shared substrate for every pipeline: a reproducible **Google Colab** environment and a
**leakage-safe, role-locked dataset manifest**. The environment mounts Google Drive for persistence
(datasets, cached artifacts, checkpoints, metrics), pins dependencies for reproducibility, and creates
the persistent directory layout everything else reads/writes. The manifest assigns every dataset to
exactly one role — debug, train/validate, or frozen unseen-test — and guarantees the unseen tier is
disjoint from all training/tuning data.

See PRD → *Solution* (role-locked splits), *Implementation Decisions → Platform constraints (Google
Colab)* and *Phase I decisions*, *Testing Decisions* (frozen golden set), *Further Notes* (leakage
caution). This is a foundation issue shared by all three pipelines.

## Acceptance criteria

- [ ] Colab notebook mounts Google Drive and creates persistent dirs for datasets, cached artifacts, checkpoints, and metrics.
- [ ] Dependencies are pinned (setup cell / requirements file) so the environment reproduces across sessions.
- [ ] A version-controlled split manifest assigns each dataset to exactly one role: debug (URFD, GMDCSA-24, Le2i slice), train/validate (UP-Fall, Le2i, GMDCSA-24), frozen unseen-test (OmniFall, CAUCAFall, MCFD, FallVision).
- [ ] The frozen unseen-test tier is provably disjoint from all debug/train/validate data; an overlap check reports any shared clips.
- [ ] Every clip carries a fall/no-fall label and dataset provenance in the manifest.
- [ ] Datasets are staged on Drive (or via a documented per-session fetch) within quota; large datasets are stored once, not re-downloaded every session.

## Blocked by

None - can start immediately.

## User stories addressed

- User story 1
- User story 2
- User story 3
- User story 4
- User story 13
- User story 26
