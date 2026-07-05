## Parent PRD

`issues/prd.md`

## What to build

**Pipeline A — implementation & testing.** End-to-end raw-video classifier: feed VideoMAE-ready clips
(issue 005) to **VideoMAE** and classify fall / no-fall. Start with pretrained VideoMAE for a fast
capability read, then fine-tune on the train tier. The **cheapest killer risk for A must be retired
first**: whether VideoMAE can actually separate fall from no-fall on these crops at all — retired
cheaply via a pretrained baseline read plus a tiny-slice overfit *before* any expensive fine-tune run.
GPU fit is now a **recorded feasibility check** against the actual Colab Pro tier (L4 default for the
fine-tune; A100 only on eval-proven need), no longer a T4 gate — deployment / real-time fit is Issue
015's concern. The fine-tune stays inside the recorded VRAM via mixed precision (fp16/bf16, capability-
detected), gradient accumulation, and a smaller variant / frozen-backbone fine-tune as needed.

See PRD → *Solution* (Pipeline A), *Phase II decisions*, *Platform constraints* (checkpoint/resume; VRAM
budget recorded against the actual tier, not assumed 16 GB), *Modules (4)*, *Testing Decisions* (overfit
a tiny slice first).

> Note: PRD *Further Notes* still names "VideoMAE-fits-in-16 GB-T4" as the cheapest killer risk — that
> framing predates Colab Pro tier availability and is superseded here (capability first; hardware-fit
> recorded, not gated). The PRD text is left unedited as a load-bearing document; this issue is authoritative for 006.

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
