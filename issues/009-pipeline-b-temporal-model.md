## Parent PRD

`issues/prd.md`

## What to build

**Pipeline B — implementation & testing.** Classify fall / no-fall from cached skeleton sequences
(issue 008). Start simple — **GRU or TCN** — for a lightweight, explainable baseline. Climb to
ST-GCN / PoseC3D / Transformer **only if a specific GRU/TCN eval failure justifies it** (the
climb-the-ladder gate; metered Colab compute makes speculative climbing costly).

See PRD → *Solution* (Pipeline B, capability ladder), *Phase II decisions*, *Modules (5)*, *Platform
constraints* (checkpoint/resume, don't climb speculatively), *Testing Decisions* (overfit a tiny
slice first).

## Acceptance criteria

- [ ] GRU/TCN baseline trains on cached skeleton shards (issue 008) and classifies fall/no-fall end-to-end.
- [ ] Overfits a tiny batch first to confirm wiring.
- [ ] Clip-level metrics reported via the issue-004 harness, plus FPS/latency and model size.
- [ ] Escalation to ST-GCN/PoseC3D/Transformer happens only if a documented GRU/TCN eval failure justifies it; the decision and its evidence are recorded.
- [ ] Training checkpoints to Drive and resumes after a session reset; configs/checkpoints versioned.

## Blocked by

- Blocked by `issues/008-pipeline-b-data-prep-skeleton.md`
- Blocked by `issues/004-eval-harness-and-golden-set.md`

## User stories addressed

- User story 16
- User story 17
- User story 22
- User story 23
- User story 24
- User story 25
