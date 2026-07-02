## Parent PRD

`issues/prd.md`

## What to build

**Pipeline C — results judgement.** Judge the hybrid variants (fusion; and fusion + verification)
through the issue-004 harness on validation + frozen unseen tier. Report the false-alarm reduction that
post-verification buys and the detection-delay it costs, and record the comparison-table rows.

See PRD → *Phase III decisions*, *Solution* (results narrative), *Further Notes* (persistence-N is the
recall-vs-delay knob).

## Acceptance criteria

- [ ] Pipeline C variants (fusion; fusion + verification) evaluated on validation + frozen unseen tier via the harness, metrics by slice.
- [ ] False-alarms/hour and detection-delay reported, showing the effect of post-verification (fewer false alarms vs. added delay).
- [ ] Cross-dataset performance drop quantified.
- [ ] Comparison-table rows recorded for Hybrid Fusion and Hybrid Fusion + Verification.
- [ ] Missed-fall (false-negative) rate reported as the priority metric.

## Blocked by

- Blocked by `issues/012-pipeline-c-fusion-postverification.md`
- Blocked by `issues/004-eval-harness-and-golden-set.md`

## User stories addressed

- User story 27
- User story 29
- User story 32
- User story 33
- User story 35
