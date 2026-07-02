## Parent PRD

`issues/prd.md`

## What to build

**Pipeline C — data collection & preprocessing.** For each track, assemble the aligned **pair**
(VideoMAE-ready clip + skeleton sequence, keyed by the same track ID and time window) plus the
**geometry feature series** the post-verification rule engine needs. Cache to Drive. This is the
preprocessing that the fusion + rule engine (issue 012) consumes.

See PRD → *Solution* (Pipeline C), *Modules (6/7)*, *Phase II decisions* (post-verification),
*Platform constraints*.

## Acceptance criteria

- [ ] For each track, produces the aligned (clip, skeleton-sequence) pair keyed by the same track ID and time window.
- [ ] Computes and caches the geometry feature series used by post-verification: body-height change, bounding-box height/width ratio, head/hip downward motion, body angle (horizontality), post-fall inactivity.
- [ ] Handles tracks where one modality is degraded/missing without dropping the other (graceful degradation).
- [ ] Outputs cached to Drive in sharded format with labels/provenance.

## Blocked by

- Blocked by `issues/005-pipeline-a-data-prep.md`
- Blocked by `issues/008-pipeline-b-data-prep-skeleton.md`

## User stories addressed

- User story 18
- User story 20
- User story 37
