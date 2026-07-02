## Parent PRD

`issues/prd.md`

## What to build

**Capstone — cross-pipeline results.** Assemble the single **comparison table** across all pipelines
and variants (from issues 007, 010, 013) and run the **ablation study** that proves which components
actually help — rather than assuming they do. This is inherently cross-pipeline: it needs A, B, and C
finished.

See PRD → *Phase III decisions* (comparison + ablation mandatory), *Solution*, *Testing Decisions*
(evaluate under matched conditions, versioned runs).

## Acceptance criteria

- [ ] A single comparison table aggregates all pipelines/variants (accuracy, precision, recall, F1, FPS, delay, false alarms/hour) evaluated under matched conditions.
- [ ] Ablation study run and reported: full-frame VideoMAE vs YOLO-crop VideoMAE; GRU vs ST-GCN; single-branch vs fusion; fusion with vs without verification.
- [ ] Each ablation states whether the component delivered a real improvement (evidence, not assumption).
- [ ] All numbers trace to versioned runs via the issue-004 harness.

## Blocked by

- Blocked by `issues/007-pipeline-a-results.md`
- Blocked by `issues/010-pipeline-b-results.md`
- Blocked by `issues/013-pipeline-c-results.md`

## User stories addressed

- User story 24
- User story 27
- User story 28
