## Parent PRD

`issues/prd.md`

## What to build

The shared, deterministic **evaluation harness** every pipeline reports through, plus enforcement of
the **frozen golden set**. It computes component- and system-level metrics **by slice**, persists
versioned results to Drive, and enforces the wall that no training/tuning code path can read from the
frozen unseen tier. Includes a "detector-of-the-detector" self-test (inject a known-bad change; confirm
the harness catches it).

See PRD → *Modules (8)*, *Testing Decisions* (frozen golden set, slice-based reporting, evaluate
perception separately from classification, test the detector), *Platform constraints* (metrics on
Drive). Foundation issue used by every results/judgement issue (007, 010, 013, 014, 015).

## Acceptance criteria

- [ ] Computes classification metrics (accuracy, precision, recall, specificity, F1, AUC-ROC, AUPRC, confusion matrix) reported by slice (dataset, lighting, occlusion, multi-person, action-confusers: sitting/sleeping/exercising).
- [ ] Provides system/event-level metric scaffolding: event-level recall, false alarms/hour, detection delay, cross-dataset F1.
- [ ] Reads final-judgement data only from the frozen unseen tier; a guard prevents any training/tuning path from reading the frozen tier.
- [ ] Metric computations verified against hand-computed values on toy inputs (unit tests).
- [ ] Detector self-test: an injected known-bad change (e.g., shuffled labels) is caught by the harness.
- [ ] Results persisted to Drive and versioned so runs stay comparable across sessions.

## Blocked by

- Blocked by `issues/001-colab-env-and-datasets.md`

## User stories addressed

- User story 22
- User story 26
- User story 36
- User story 38
