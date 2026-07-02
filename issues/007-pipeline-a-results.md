## Parent PRD

`issues/prd.md`

## What to build

**Pipeline A — results judgement.** Judge the fine-tuned VideoMAE through the issue-004 harness on the
validation set, then take the first **cross-dataset read** on the frozen unseen tier. Record Pipeline
A's comparison-table row and analyze its worst failure slices.

See PRD → *Phase III decisions*, *Solution* (results narrative), *Testing Decisions* (slice-based,
generalization, false-negative priority).

## Acceptance criteria

- [ ] Pipeline A evaluated on validation + frozen unseen tier via the issue-004 harness, metrics reported by slice.
- [ ] Cross-dataset performance drop (validation → unseen) is quantified and recorded.
- [ ] Failure analysis identifies the worst slices (action-confusers, occlusion, low light, multi-person).
- [ ] A comparison-table row for Pipeline A is recorded (accuracy, precision, recall, F1, FPS, delay, false alarms/hour).
- [ ] False negatives are surfaced explicitly as the priority error.

## Blocked by

- Blocked by `issues/006-pipeline-a-videomae-classifier.md`
- Blocked by `issues/004-eval-harness-and-golden-set.md`

## User stories addressed

- User story 27
- User story 29
- User story 32
- User story 38
