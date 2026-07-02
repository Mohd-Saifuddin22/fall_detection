## Parent PRD

`issues/prd.md`

## What to build

**Pipeline A — data collection & preprocessing.** The A-specific step that turns cached crop clips
(issue 003) into **VideoMAE-ready input tensors** on the debug tier, verifying the input contract
before any training. Reads from the Drive cache only — no perception models (YOLO/ViTPose) in this
path — per the Colab decouple-and-cache constraint.

See PRD → *Solution* (Pipeline A), *Phase II decisions* (build order), *Platform constraints* (read
from cache), *Modules (2/4)*. First tracer-bullet slice of Pipeline A.

## Acceptance criteria

- [ ] Reads cached crop-clip shards (issue 003) and produces VideoMAE-ready tensors matching the model's expected shape and normalization.
- [ ] Verified end-to-end on a small debug-tier batch before scaling.
- [ ] Labels and provenance are preserved through to the model input.
- [ ] Reads from the Drive cache without recomputing perception.

## Blocked by

- Blocked by `issues/003-crop-clip-generator.md`

## User stories addressed

- User story 8
- User story 13
