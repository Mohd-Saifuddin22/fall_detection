## Parent PRD

`issues/prd.md`

## What to build

**Pipeline B — results judgement.** Judge each temporal-model variant built (GRU/TCN and any escalated
model) through the issue-004 harness on validation + frozen unseen tier. Record comparison-table rows
and tie failures to pose quality (occlusion, low resolution, missing keypoints) — the known weakness of
the skeleton pipeline.

See PRD → *Phase III decisions*, *Solution* (results narrative), *Testing Decisions*.

## Acceptance criteria

- [ ] Each Pipeline B variant evaluated on validation + frozen unseen tier, metrics by slice.
- [ ] Cross-dataset performance drop quantified.
- [ ] Failure analysis ties errors to pose quality (occlusion, low-resolution, missing keypoints).
- [ ] Comparison-table rows recorded for each temporal-model variant.
- [ ] Robustness under occlusion / missing keypoints reported.

## Blocked by

- Blocked by `issues/009-pipeline-b-temporal-model.md`
- Blocked by `issues/004-eval-harness-and-golden-set.md`

## User stories addressed

- User story 27
- User story 29
- User story 32
- User story 37
