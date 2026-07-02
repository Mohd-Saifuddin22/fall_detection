## Parent PRD

`issues/prd.md`

## What to build

**Pipeline C — implementation & testing.** Assemble the full decision chain: configurable **late
fusion** of the VideoMAE and skeleton scores → deterministic **post-verification rule engine** →
**alarm decision** (threshold + persistence). Fusion runs on **precomputed per-branch scores** (Colab
VRAM constraint — no live models resident). Tune the fusion weights and the persistence threshold N on
validation.

See PRD → *Solution* (Pipeline C), *Modules (6: fusion, 7: post-verification/alert)*, *Phase II
decisions* (late fusion first, deterministic gate, trigger rule), *Platform constraints* (precomputed
scores), *Testing Decisions* (unit-test fusion + rule engine).

## Acceptance criteria

- [ ] Late-fusion module combines per-branch scores with configurable weights (e.g., 0.5/0.5 and 0.6 skeleton / 0.4 video), operating on precomputed scores, not live models.
- [ ] Post-verification rule engine fires only when geometric conditions + persistence are jointly met (e.g., fall prob > threshold for N consecutive frames); a single high-score spike does not trigger.
- [ ] Fusion arithmetic and the rule engine are covered by unit tests (including boundary weights and "no-fire on single spike").
- [ ] Fusion weights and persistence N are tuned on validation (not left at defaults); the sweep is recorded.
- [ ] End-to-end: a known fall clip yields an alert; a known no-fall confuser (sitting/sleeping/exercising) stays silent.

## Blocked by

- Blocked by `issues/006-pipeline-a-videomae-classifier.md`
- Blocked by `issues/009-pipeline-b-temporal-model.md`
- Blocked by `issues/011-pipeline-c-data-prep.md`

## User stories addressed

- User story 18
- User story 19
- User story 20
- User story 33
- User story 34
