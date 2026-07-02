## Parent PRD

`issues/prd.md`

## What to build

A deterministic module that converts per-person tracks (issue 002) into fixed-length, model-ready
**crop clips**, cached to Drive as **compact sharded archives** (not thousands of loose per-frame
files — a hard Colab I/O constraint). Each clip is 16 or 32 frames at 224×224 with a configurable
20–40% margin around the person so the model keeps body extremities and floor context.

See PRD → *Modules (2: clip builder)*, *Phase I decisions* (fixed clip contract), *Platform
constraints* (sharded caching), *Testing Decisions* (deterministic → unit-tested). This cache is
consumed by both Pipeline A (issue 005) and Pipeline B (issue 008).

## Acceptance criteria

- [ ] Given tracks from issue 002, emits per-person clips of exactly 16 or 32 frames at 224×224 (configurable).
- [ ] Applies a configurable margin (20–40%) around the person before crop/resize.
- [ ] Writes clips to Drive as compact sharded archives; a debug-tier run does not produce loose per-frame files.
- [ ] Deterministic — same input + config yields identical output; unit tests cover frame count, size, margin, resize, and short-track edge cases.
- [ ] Each cached clip retains track ID, source video, label, and dataset provenance.

## Blocked by

- Blocked by `issues/002-perception-frontend-yolo-bytetrack.md`

## User stories addressed

- User story 8
- User story 9
