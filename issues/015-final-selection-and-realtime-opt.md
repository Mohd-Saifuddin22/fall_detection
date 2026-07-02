## Parent PRD

`issues/prd.md`

## What to build

**Capstone — final selection & real-time optimization.** Select the final architecture using the
**recall-first priority order**, then optimize the selected model for real-time inference. This is
**HITL**: the selection is a load-bearing, hard-to-reverse decision that everything downstream gets
optimized around, so it requires human sign-off.

See PRD → *Phase III decisions* (recall-first selection), *Final Judgement Criteria*, *Platform
constraints* (Colab FPS is relative, not a deployment guarantee), *Trade-off Analysis Protocol*.

## Acceptance criteria

- [ ] Final architecture selected by the documented priority: recall → low false alarms/hour → F1 → detection delay → cross-dataset performance → FPS/latency → model size.
- [ ] Selection is signed off by a human (HITL) with the trade-offs stated explicitly.
- [ ] The selected model is optimized for real-time inference; before/after FPS/latency reported.
- [ ] FPS/latency explicitly labeled as Colab-relative, not a target-hardware guarantee.
- [ ] The recommendation names what it trades away.

## Blocked by

- Blocked by `issues/014-final-comparison-and-ablation.md`

## User stories addressed

- User story 29
- User story 30
- User story 31
- User story 34
- User story 35
- User story 38
