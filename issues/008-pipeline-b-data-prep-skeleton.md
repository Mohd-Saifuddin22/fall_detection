## Parent PRD

`issues/prd.md`

## What to build

**Pipeline B — data collection & preprocessing.** Run **ViTPose** on cached crop clips (issue 003) to
extract 17 keypoints/frame, assemble **skeleton feature sequences** in the fixed contract T×17×3
(x, y, confidence), and cache them to Drive as **sharded archives**. Owns the missing/low-confidence
keypoint policy so sequences degrade gracefully under occlusion. Reporting ViTPose quality here also
retires the "pose quality on low-res CCTV" risk early.

See PRD → *Solution* (Pipeline B), *Modules (3)*, *Phase I decisions* (skeleton contract), *Platform
constraints* (sharded cache), *Testing Decisions*, *Further Notes* (pose-on-CCTV risk).

## Acceptance criteria

- [ ] Runs ViTPose over cached crop clips (issue 003) and outputs skeleton sequences in the fixed contract T×17×3 (x, y, confidence).
- [ ] Missing/low-confidence keypoints handled by an explicit, documented policy (no silent garbage); sequences degrade gracefully under occlusion.
- [ ] Sequences cached to Drive as compact sharded archives with labels/provenance preserved.
- [ ] ViTPose component quality reported on the debug tier (keypoint confidence, missing-keypoint rate; PCK where ground-truth exists).
- [ ] Sequence assembly (shape, missing-keypoint handling) covered by unit tests.

## Blocked by

- Blocked by `issues/003-crop-clip-generator.md`

## User stories addressed

- User story 10
- User story 11
- User story 12
