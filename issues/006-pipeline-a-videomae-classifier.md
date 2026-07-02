## Parent PRD

`issues/prd.md`

## What to build

**Pipeline A — implementation & testing.** End-to-end raw-video classifier: feed VideoMAE-ready clips
(issue 005) to **VideoMAE** and classify fall / no-fall. Start with pretrained VideoMAE for a fast
capability read, then fine-tune on the train tier. The **cheapest killer risk for A must be retired
first**: that VideoMAE fine-tune fits within a ~16 GB Colab T4 at a usable batch size (mixed precision,
gradient accumulation, smaller variant / frozen backbone as needed).

See PRD → *Solution* (Pipeline A), *Phase II decisions*, *Platform constraints* (16 GB budget,
checkpoint/resume), *Modules (4)*, *Further Notes* (VideoMAE-in-16GB is the cheapest killer risk),
*Testing Decisions* (overfit a tiny slice first).

## Acceptance criteria

- [ ] **Record actual GPU tier first.** Colab Pro tiers (T4 / L4 / A100 / etc.) are now available; record the actual GPU name, VRAM, and CUDA version observed at session start. The T4/deployment-fit concern is moved to Issue 015 — this issue is no longer gated on "must fit in T4".
- [ ] **Feasibility gate:** VideoMAE fine-tune runs within the recorded GPU tier at a usable batch size via mixed precision + gradient accumulation (smaller variant / frozen backbone if required); the working config is recorded against the actual VRAM, not assumed ~16 GB.
- [ ] Overfits a tiny batch first to confirm wiring before scaling compute.
- [ ] Pretrained VideoMAE gives a baseline read on debug clips; then fine-tuned on the train tier.
- [ ] Training checkpoints to Drive frequently and resumes from checkpoint after a session reset.
- [ ] Clip-level metrics reported via the issue-004 harness (accuracy, precision, recall, specificity, F1, AUC-ROC, AUPRC, confusion matrix) plus FPS/latency and model size.
- [ ] Model config, params, and checkpoints are versioned and tied to the result that justified them.

## Blocked by

- Blocked by `issues/005-pipeline-a-data-prep.md`  (data-prep contract + loader; CLOSED)
- Blocked by `issues/004-eval-harness-and-golden-set.md` (metrics infrastructure — STILL REQUIRED before formal results)

## User stories addressed

- User story 14
- User story 15
- User story 22
- User story 23
- User story 24
- User story 25

## Blocked by

- Blocked by `issues/005-pipeline-a-data-prep.md`
- Blocked by `issues/004-eval-harness-and-golden-set.md`

## User stories addressed

- User story 14
- User story 15
- User story 22
- User story 23
- User story 24
- User story 25
