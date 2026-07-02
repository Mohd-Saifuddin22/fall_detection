# PRD: CCTV Fall / Faint Detection System (Computer Vision)

> Source: `fall_detection_rough_plan.md`. This PRD reorganizes that plan into a build-ready
> specification structured chronologically around three macro-phases:
> **Phase I — Data Collecting & Processing**, **Phase II — Implementing & Testing**,
> **Phase III — Results**. Every phase in the original plan maps into one of these three.

---

## Problem Statement

We need to detect when a person **falls, faints, or collapses** in CCTV/surveillance video, reliably
enough to be useful in the real world — not just on a single benchmark.

A fall is not a single-image problem. A person lying on the ground in one frame may simply be
sleeping, sitting on the floor, resting, exercising, or crawling. Judging "fall vs. not-fall" from
one frame produces both **missed falls** and **false alarms**. The signal that actually distinguishes
a fall is *motion over time*: standing → sudden downward movement → lying down → stillness.

Real deployments make this harder still: awkward CCTV angles, low video quality, multiple people,
occlusion, and varied lighting. The hardest requirement is **generalization** — a model that scores
well on one dataset routinely collapses on unseen footage from a different environment.

In this domain the costs are asymmetric. A **false negative (a real fall the system misses)** is the
most dangerous error, because someone who needs help doesn't get it. A system that is merely accurate
on average — but misses falls, or cries wolf so often that operators stop trusting it — has failed.

## Solution

Build a temporal fall-detection system as a **shared perception front-end feeding interchangeable
classifier heads**, then build, compare, and combine three pipelines rather than betting on one:

- **Shared front-end:** `CCTV video → YOLO person detection → ByteTrack tracking → per-person crop clips`.
  This isolates each person as a continuous clip so classifiers focus on the person, not the background.
- **Pipeline A (raw-video):** crop clip → **VideoMAE** → fall / no-fall.
- **Pipeline B (skeleton):** crop clip → **ViTPose** keypoints → skeleton feature sequence → **temporal model**
  (start GRU/TCN; escalate to ST-GCN / PoseC3D / Transformer only if evals justify it) → fall / no-fall.
- **Pipeline C (hybrid fusion):** run Branch A and Branch B, combine via configurable **late fusion**,
  then apply a **deterministic rule-based post-verification** layer before raising an alert.

Evaluate each *component* in isolation and each *full pipeline* across **role-locked dataset splits**
(debug / train-validate / frozen-unseen-test), using **recall-first, false-alarm-aware, cross-dataset**
metrics. Run an ablation to prove each component earns its place, select the final architecture on a
safety-first priority order, then optimize it for real-time inference.

The work proceeds in three chronological phases:

1. **Phase I — Data Collecting & Processing:** acquire tiered datasets; stand up the shared front-end
   (detection → tracking → crop clips → skeleton sequences); produce clean, labeled, leakage-safe
   model-ready data.
2. **Phase II — Implementing & Testing:** build Pipelines A, B, and C plus post-verification; evaluate
   each component and pipeline under fair conditions.
3. **Phase III — Results:** cross-dataset generalization testing, comparison table, ablation study,
   final model selection, and real-time optimization.

---

## User Stories

### Phase I — Data Collecting & Processing

1. As a data engineer, I want to acquire small debug datasets (URFD, GMDCSA-24, a slice of Le2i), so that I can validate the pipeline end-to-end before spending effort on large-scale training.
2. As a data engineer, I want to acquire larger training/validation datasets (UP-Fall, Le2i, GMDCSA-24), so that models train on diverse fall examples rather than one environment.
3. As a data engineer, I want to reserve unseen datasets (OmniFall, CAUCAFall, MCFD, FallVision, and any unseen CCTV-style footage) strictly for final testing, so that I can measure real-world generalization without leakage.
4. As a data engineer, I want to avoid relying on a single dataset or environment, so that the model does not overfit to one scene, camera, or lighting condition.
5. As an ML engineer, I want YOLO to place a bounding box around every person in each frame, so that downstream stages operate on person regions instead of the full CCTV scene.
6. As an ML engineer, I want ByteTrack to assign a stable ID to the same person across frames, so that I can assemble one continuous clip per individual.
7. As an ML engineer, I want tracking to survive the fall motion itself (not only normal walking), so that the most important moment isn't lost to an ID switch or fragmented track.
8. As a data engineer, I want to crop each tracked person and assemble fixed-length clips (16 or 32 frames at 224×224), so that clips are uniform and model-ready.
9. As a data engineer, I want a configurable crop margin (20%–40%) rather than a tight crop, so that the clip retains body extremities and floor context needed to judge a fall.
10. As an ML engineer, I want ViTPose to extract 17 body keypoints per frame, so that I can build skeleton sequences for the skeleton-based pipeline.
11. As a data engineer, I want skeleton sequences stored in a fixed tensor shape (frames × 17 × 3, e.g. 30×17×3 carrying x, y, confidence), so that temporal models receive a stable input contract.
12. As a data engineer, I want missing or low-confidence keypoints handled explicitly, so that skeleton sequences degrade gracefully under occlusion instead of producing garbage.
13. As a data engineer, I want each clip labeled fall / no-fall with its dataset provenance recorded, so that I can run slice-based evaluation and enforce train/test separation.

### Phase II — Implementing & Testing

14. As an ML engineer, I want to first run a pretrained VideoMAE on crop clips, so that I get a fast capability read before committing to fine-tuning.
15. As an ML engineer, I want to fine-tune VideoMAE on fall datasets, so that Pipeline A becomes a strong raw-video baseline.
16. As an ML engineer, I want a skeleton temporal model that starts simple (GRU or TCN), so that I get a lightweight, explainable Pipeline B baseline quickly.
17. As an ML engineer, I want the option to upgrade the skeleton model to ST-GCN, PoseC3D, or a Transformer, so that I can test whether added complexity actually improves detection.
18. As an ML engineer, I want a deterministic rule-based post-verification layer (sudden body-height drop, bounding-box aspect-ratio change, head/hip moving downward, body angle becoming horizontal, low movement after the fall, sustained high score), so that a single spurious prediction cannot trigger an alarm.
19. As an ML engineer, I want a configurable trigger rule (e.g., fall probability > 0.80 for 10 consecutive frames), so that I can tune the sensitivity vs. false-alarm trade-off.
20. As an ML engineer, I want a hybrid fusion pipeline that combines VideoMAE and the skeleton model via configurable late fusion (e.g., 0.5/0.5 or 0.6 skeleton + 0.4 video), so that I can exploit both raw-visual and body-motion cues.
21. As an ML engineer, I want each component evaluated in isolation — YOLO (mAP@0.5, mAP@0.5:0.95, person precision/recall); ByteTrack (ID switches, fragmentation, IDF1/MOTA/HOTA where possible); ViTPose (keypoint confidence, missing-keypoint rate, PCK where ground truth exists) — so that I can localize failures to the responsible stage.
22. As an ML engineer, I want clip-level classification metrics for every model (accuracy, precision, recall, specificity, F1, AUC-ROC, AUPRC, confusion matrix), so that I can compare pipelines fairly under controlled training conditions.
23. As an ML engineer, I want speed and footprint metrics (FPS, latency, model size) captured alongside accuracy from the start, so that real-time viability is never a phase-2 surprise.
24. As an ML engineer, I want to compare Pipeline A and Pipeline B under fair, matched training conditions, so that the comparison reflects the models and not the setup.
25. As an ML engineer, I want to first overfit a tiny slice to confirm the wiring, so that I can distinguish a plumbing bug from a genuine capacity problem before scaling compute.
26. As an ML engineer, I want experiment configs (model choice, params, fusion weights, thresholds) versioned and tied to their results, so that every number is reproducible.

### Phase III — Results

27. As a researcher, I want a final comparison table across all pipelines (YOLO+VideoMAE, ViTPose+GRU, ViTPose+TCN, ViTPose+ST-GCN, Hybrid Fusion, Hybrid+Verification) reporting accuracy, precision, recall, F1, FPS, delay, and false alarms/hour, so that I can select the best architecture on the axes that matter.
28. As a researcher, I want an ablation study (full-frame VideoMAE vs. YOLO-crop VideoMAE; GRU vs. ST-GCN; single-branch vs. fusion; with vs. without inactivity verification), so that I know which component delivers real improvement rather than assuming it does.
29. As a researcher, I want real-world final-testing metrics on unseen data (event-level recall, false alarms/hour, detection delay, cross-dataset F1, performance drop, and robustness to occlusion, low light, and multiple people), so that I can credibly claim generalization.
30. As a researcher, I want the final model chosen by a recall-first priority (high recall → low false alarms/hour → high F1 → low detection delay → good cross-dataset performance → good FPS/latency → acceptable model size), so that selection reflects safety, not average-case accuracy.
31. As an ML engineer, I want the selected architecture optimized for real-time inference, so that it runs fast enough for practical CCTV use.
32. As a researcher, I want the results narrative documented as "separate video-based and skeleton-based pipelines → compared across datasets → combined into a hybrid fusion model," so that the project is coherent, reproducible, and publishable.

### End-User & Cross-Cutting

33. As a monitoring operator, I want a fall alert only when the model is confident and the motion pattern is consistent over several frames, so that I'm not desensitized by constant false alarms.
34. As a monitoring operator, I want low detection delay, so that help can be dispatched quickly after a fall.
35. As a care-facility manager, I want a low missed-fall rate on my own camera setup, so that residents reliably get timely help.
36. As a person being monitored, I want the system to distinguish my sitting, sleeping, resting, and exercising from an actual fall, so that I'm neither falsely alarmed on nor missed when I really fall.
37. As an ML engineer, I want the system to keep functioning when one person is occluded or their keypoints are missing, so that one degraded input doesn't break detection for everyone in frame.
38. As a researcher, I want false negatives treated as the most costly error throughout evaluation and selection, so that the entire system is oriented around not missing real falls.

---

## Implementation Decisions

### Shared architecture

- **Two-stage design: one shared perception front-end feeding interchangeable classifier heads.**
  The front-end (`video → {track_id → ordered person frames}`) is built once and reused by Pipelines
  A, B, and C. This is what makes a *fair* comparison possible and keeps data preparation DRY.
- **Temporal-first framing.** The unit of classification is a *clip / sequence*, never a single frame,
  because the discriminating signal is motion over time.
- **Uniform classifier interface.** Every classifier head — VideoMAE, skeleton temporal model, and the
  fused model — emits a single scalar **fall score in [0, 1]**. Fusion and post-verification are written
  against this scalar, so they are head-agnostic and any head can be swapped without touching them.
- **Deterministic where possible, model judgment only where necessary.** Cropping, clip assembly,
  fusion arithmetic, geometric feature computation, the alarm decision rule, and all metrics live in
  plain, testable code. The models are reserved for the genuine perceptual/temporal judgment.
- **Climb the capability ladder; stop at the lowest rung that clears the bar.** Pretrained VideoMAE
  before fine-tuning; GRU/TCN before ST-GCN/PoseC3D/Transformer. Escalate only when a *specific eval
  failure* proves the current rung insufficient — not preemptively.

### Modules (deep modules with simple, stable interfaces)

1. **Perception front-end** — `video → {track_id → ordered frames}`. Wraps YOLO + ByteTrack; internal
   weights/tracker params are swappable behind a stable interface.
2. **Clip builder** — `(tracked frames, config{clip_len ∈ {16,32}, size=224, margin ∈ [0.2,0.4]}) → clip tensor`.
   Pure, deterministic preprocessing.
3. **Skeleton extractor** — `clip → skeleton sequence (T×17×3: x, y, confidence)`. Wraps ViTPose;
   owns the missing-/low-confidence-keypoint policy.
4. **Raw-video classifier (Pipeline A head)** — `clip → fall_score`. Wraps VideoMAE.
5. **Skeleton temporal classifier (Pipeline B head)** — `skeleton sequence → fall_score`. Swappable
   backend (GRU / TCN / ST-GCN / PoseC3D / Transformer) behind one interface.
6. **Fusion module (Pipeline C)** — `(video_score, skeleton_score, weights) → fused_score`. Pure
   function; late fusion; weights are configuration, not code.
7. **Post-verification / alert-decision engine** — `(score sequence, geometry features, rules) → alert`.
   Deterministic rule engine (persistence + geometric conditions).
8. **Evaluation harness** — `(predictions, labels, split, slice) → metrics`. Deterministic; supports
   both component-level and system-level metrics, reported by slice.

### Platform constraints (Google Colab)

All development, training, and evaluation run on **Google Colab**. This is a hard constraint that
shapes the data pipeline, the training approach, and how performance numbers are read — not an
environment footnote.

- **Sessions are ephemeral — assume disconnects and runtime resets.** Timeouts/resets wipe the local
  `/content` disk. Every training run must **checkpoint frequently and resume from checkpoint**
  (VideoMAE fine-tune, temporal models). No long run may depend on one uninterrupted session.
- **Google Drive is the persistence layer, and small-file I/O from Drive is a bottleneck.** Datasets,
  cached crop clips, skeleton sequences, checkpoints, and metrics live on Drive. Reading thousands of
  tiny per-frame/per-clip files from Drive is prohibitively slow — cached artifacts must be written in
  **compact sharded formats** (batched tensor archives / tar shards), not loose small files.
- **Decouple perception from classifier training** to fit VRAM and avoid recompute. The baseline GPU
  is a ~16 GB **T4 and is not guaranteed** (type varies per session); loading YOLO + ByteTrack +
  ViTPose + VideoMAE at once can exceed VRAM. So **precompute and cache** crop clips (issue 003) and
  skeleton sequences (issue 008) to Drive once, then train each head by reading the cache — nothing
  needs all models resident simultaneously. Pipeline C fusion then runs on **precomputed per-branch
  scores**, reinforcing the late-fusion choice.
- **Train within ~16 GB:** mixed precision (fp16/bf16), gradient accumulation for effective batch
  size, small physical batches, and a smaller VideoMAE variant / frozen-backbone fine-tune where a
  full fine-tune won't fit.
- **Compute is metered and throttleable,** making speculative training costly — reinforces climbing the
  model ladder (GRU/TCN → ST-GCN/PoseC3D) only on eval-proven need, and running the most informative
  ablations rather than the full grid.
- **Pin dependencies.** Colab's base image drifts (CUDA/torch/package versions); a pinned setup cell /
  requirements file is required for reproducibility.
- **FPS/latency on Colab is indicative, not a deployment guarantee.** Colab GPUs are shared and are not
  the target hardware — treat timing as **relative comparison between pipelines**, not an absolute
  real-time claim (target-hardware benchmarking is out of scope).

### Phase I decisions (Data Collecting & Processing)

- **Role-locked dataset splits, enforced from day one.** Debug/start = URFD, GMDCSA-24, Le2i (slice);
  train/validate = UP-Fall, Le2i, GMDCSA-24; frozen unseen test = OmniFall, CAUCAFall, MCFD, FallVision
  (+ unseen CCTV if available). The unseen-test set is walled off from all training and tuning.
- **Fixed clip contract:** 16 or 32 frames × 224×224, with a 20–40% margin around the person.
- **Fixed skeleton contract:** T×17×3 (x, y, confidence), e.g. 30×17×3.
- **Pretrained perception, not trained from scratch.** YOLO, ByteTrack, and ViTPose are used as
  off-the-shelf components; project training effort goes into the classifier heads.
- **Environment:** Google Colab — a *hard constraint*; see **Platform constraints (Google Colab)**
  above (ephemeral sessions → checkpoint/resume; Drive caching in sharded formats; ~16 GB VRAM budget;
  metered compute).

### Phase II decisions (Implementing & Testing)

- **Build order:** Front-end → clip generation → Pipeline A (pretrained VideoMAE → fine-tuned) →
  Pipeline B (GRU/TCN → stronger backends) → post-verification → Pipeline C (fusion).
- **Late fusion first.** Start with simple weighted late fusion; candidate weightings 0.5/0.5 and
  0.6 skeleton / 0.4 video are *starting hyperparameters to tune on validation*, not fixed truths.
  Escalate to intermediate/learned fusion only if late fusion underperforms in evals.
- **Post-verification is a deterministic gate, not a model.** Rules operate on temporal persistence
  and geometry (sudden height drop, box aspect-ratio change, head/hip downward motion, horizontal body
  angle, post-fall inactivity, sustained high score). Default trigger: `fall_prob > 0.80 for N=10
  consecutive frames`; threshold and N are configurable and are the primary sensitivity knobs.
- **Observability before traffic.** Per-stage outputs, fall scores, latency, and cost are logged for
  every run so failures can be traced to a specific stage.

### Phase III decisions (Results)

- **Component + system evaluation, both required.** Components: YOLO (mAP), ByteTrack (ID switches /
  IDF1 / MOTA / HOTA), ViTPose (confidence / missing-rate / PCK). System: event-level recall, false
  alarms/hour, detection delay, cross-dataset F1, FPS, latency.
- **Ablation is mandatory before claiming a component helps.**
- **Recall-first selection priority:** recall → low false alarms/hour → F1 → detection delay →
  cross-dataset performance → FPS/latency → model size.
- **Real-time optimization is the final step,** performed on the *selected* architecture only. Note
  FPS/latency measured on Colab is indicative-only (see *Platform constraints*); absolute real-time
  validation needs the target deployment hardware (out of scope).

---

## Testing Decisions

### What makes a good test here

- **Test external behavior and contracts, not internals.** Assert on a module's input→output contract
  (shapes, ranges, decisions), not on how it computes them, so tests survive refactors.
- **Split by determinism.** Deterministic modules get **unit tests** with fixed inputs and exact
  expected outputs. Probabilistic modules (the classifiers) are *not* unit-tested for exact values;
  they are measured against a **frozen golden set** with slice-based metrics.

### Modules recommended for unit tests (deterministic)

- **Clip builder** — output frame count (16/32), spatial size (224×224), and margin applied correctly;
  behavior on short/edge tracks.
- **Skeleton extractor assembly** — output shape T×17×3; documented handling of missing/low-confidence
  keypoints.
- **Fusion module** — weighted combination is correct, including boundary weights (0/1) and that
  weights sum as intended.
- **Post-verification engine** — fires *only* when all geometric conditions plus the persistence
  threshold are met; does not fire on a single high-score spike.
- **Evaluation harness** — metric computations (precision, recall, F1, specificity, confusion matrix)
  match hand-computed values on toy inputs.

### Behavioral / integration tests

- Front-end produces exactly one continuous clip per tracked ID on a short fixture video, and keeps the
  same ID across a simulated fall.
- End-to-end: a known fall clip yields an alert; a known no-fall clip (sitting/sleeping/exercising)
  stays silent.

### Evaluation for probabilistic modules

- **Frozen golden regression set**, walled off from all training/fine-tuning data, to measure
  generalization rather than memorization.
- **Evaluate perception separately from classification** (the direct analog of evaluating retrieval
  separately from generation in RAG): measure detection/tracking/pose quality *before* attributing
  errors to the classifier, so you always know which half is broken.
- **Slice-based reporting**, never aggregate-only: by dataset, lighting, occlusion, multi-person, and
  action-confusers (sitting / sleeping / resting / exercising / crawling).
- **Test the detector.** Periodically inject a known-bad change (shuffled labels, degraded tracking)
  into a shadow path and confirm the eval catches it — an eval nobody validates is decorative.

### Prior art to reuse rather than hand-roll

- Object-detection metrics (mAP@0.5, mAP@0.5:0.95), MOT metrics (IDF1, MOTA, HOTA), pose PCK, and
  classification curves (ROC, PR/AUPRC) all have established reference implementations; use them.

> Note: the skill normally asks the developer which modules to test. Since questions were skipped, the
> above is the recommended default (unit-test the five deterministic modules; golden-set-eval the
> classifiers); adjust if priorities differ.

---

## Out of Scope

- **Production deployment and streaming infrastructure.** Real-time *optimization* of the chosen model
  is in scope; live camera ingestion, edge/hardware deployment, alert routing/notification delivery,
  and operator dashboards/UI are not.
- **Absolute real-time performance guarantees on target hardware.** All timing is measured on Colab as
  a *relative* comparison between pipelines; benchmarking on the actual deployment device is out of
  scope for this PRD.
- **Multi-camera fusion and cross-camera person re-identification.**
- **Training YOLO / ByteTrack / ViTPose from scratch.** These are used pretrained; only the classifier
  heads are trained/fine-tuned.
- **Detection of events other than fall/faint/collapse** (e.g., violence, intrusion).
- **Fall severity or injury classification**, and general activity recognition beyond fall / no-fall.
- **Collecting new proprietary CCTV footage / data annotation programs.** The project uses public
  datasets plus unseen CCTV "if available." Pose ground-truth is assumed only "where available" (PCK).
- **Privacy, consent, and legal governance framework.** Load-bearing for any real deployment, but not
  part of this modeling PRD (see Further Notes).

---

## Further Notes

*The following are engineering flags and assumptions surfaced from analyzing the plan; they are not
new scope, but they should steer sequencing and evaluation.*

- **Cheapest killer risk to retire first: the shared front-end on real CCTV.** If YOLO+ByteTrack can't
  hold a stable track *through the fall*, or ViTPose can't produce usable keypoints on low-resolution
  footage, Pipeline B is undermined before it starts. Validate detection → tracking → pose quality on
  the debug datasets **before** investing in A/B/C. Natural fallback if pose proves unreliable on CCTV:
  Pipeline A (crop or full-frame VideoMAE), which does not depend on pose.
- **Leakage caution.** GMDCSA-24 and Le2i appear in *both* the debug and the train/validate lists.
  That is acceptable for those two roles, but the unseen-test set (OmniFall, CAUCAFall, MCFD,
  FallVision) must stay fully disjoint, or the cross-dataset generalization numbers will be optimistic
  and misleading.
- **Class imbalance is intrinsic.** Falls are rare events, so accuracy will look deceptively high.
  AUPRC and event-level recall are the honest metrics; expect to weight the loss or resample, and
  report the confusion matrix, not just headline accuracy.
- **Cross-dataset drift is the real target.** Within-dataset validation scores will overstate quality;
  the number that matters for the project's stated goal ("works on unseen videos") is cross-dataset F1
  and the performance drop from validation to unseen test.
- **Fusion weights are asserted, not learned.** The 0.6/0.4 (and 0.5/0.5) splits are starting points;
  tune them on validation and report the sweep rather than treating a single split as ground truth.
- **The main safety knob is the persistence threshold N.** Raising the "high score for N consecutive
  frames" requirement lowers false alarms but *increases detection delay* — this is the central
  recall-vs-nuisance-vs-latency trade-off and should be tuned explicitly, not left at a default.
- **Governance is a real one-way door for deployment.** CCTV of identifiable people carries privacy and
  consent obligations. Out of scope for the modeling work here, but must be gated before any production
  use.
- **Colab is a live viability risk, not just an environment note** (see *Implementation Decisions →
  Platform constraints (Google Colab)*). The sharpest, cheapest early test: can VideoMAE fine-tune
  within a ~16 GB T4 at a usable batch size? If not, that bounds Pipeline A before anything else —
  retire that risk early on the debug tier rather than discovering it after building the data pipeline.
- **Resist premature ladder-climbing.** The plan lists many temporal models (GRU, TCN, ST-GCN, PoseC3D,
  Transformer) and a heavy hybrid. Build GRU/TCN first; only climb to heavier models when a specific
  eval failure — not intuition — justifies the added cost, latency, and debugging burden.
