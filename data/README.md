# Dataset Staging Guide

Where raw fall-detection datasets live, how they get there, and the rules
they must obey. Read this before adding any dataset to the project.

This file covers PRD acceptance criterion:
> *"Datasets are staged on Drive (or via a documented per-session fetch)
> within quota; large datasets are stored once, not re-downloaded every
> session."*

---

## TL;DR

| Where on Drive | What lives there | Who writes | Who reads |
|---|---|---|---|
| `MyDrive/fall_detection/datasets/<dataset_slug>/...` | Raw + staged video files | Data acquisition (manual / per-session fetch) | Issue 002 front-end, Issue 003 clip generator, Issue 008 skeleton extractor |
| `MyDrive/fall_detection/artifacts/...` | Precomputed crops, skeletons, env lock | Issues 003 / 008 / notebooks | Issues 006 / 009 / 011 training runs |
| `MyDrive/fall_detection/checkpoints/...` | Training checkpoints | Issues 006 / 009 / 011 | Resumed runs, Issue 015 realtime opt |
| `MyDrive/fall_detection/metrics/...` | Eval results, golden-set scores | Issue 004 / downstream | Researchers, final report |
| `MyDrive/fall_detection/logs/...` | Per-run logs, run metadata | Every Colab run | Humans debugging |

The persistent layout is created by `colab/setup.py:DriveLayout.ensure()`.
Do not create these directories by hand — re-running the setup notebook
is idempotent.

---

## Where each dataset is staged

```
MyDrive/fall_detection/datasets/
├── urfd/                 # debug tier
├── gmdcsa24/             # debug + train/validate (cross-listed)
├── le2i/                 # debug + train/validate (cross-listed)
├── up_fall/              # train/validate
├── omnifall/             # FROZEN VAULT — frozen_unseen_test ONLY
├── caucafall/            # FROZEN VAULT — frozen_unseen_test ONLY
├── mcfd/                 # FROZEN VAULT — frozen_unseen_test ONLY
└── fallvision/           # FROZEN VAULT — frozen_unseen_test ONLY
```

The slug naming matches `data/manifests/__init__.py:ALL_KNOWN_DATASETS`
(lowercase, underscores for spaces). Adding a new dataset means adding
the slug to that file and the directory on Drive.

---

## The hard rules

### 1. Store once, reuse everywhere

**Rule.** A dataset that has been staged on Drive must never be
re-downloaded in a Colab session. Issue 003 / 008 / 002 etc. must
read from Drive, not from a fresh `wget` / `gdown` / Hugging Face hub
fetch.

**Why.** PRD Platform Constraints → *"Sessions are ephemeral — assume
disconnects and runtime resets. … Drive is the persistence layer … large
datasets are stored once, not re-downloaded every session."*  Re-downloading
burns Drive egress, network quota, and runtime wall-clock, and risks
version drift between sessions.

**How to enforce.**
- Acquire datasets once with a documented script (issue 002 / 003 will
  land these) and write to `datasets/<slug>/` on Drive.
- Issue 003's clip generator reads from `datasets/<slug>/`, never from
  a remote URL.
- If a per-session fetch is genuinely necessary (e.g. dataset gated by
  auth, quota cap forces chunked download), it must be documented in
  this file with the exact command.

### 2. Frozen unseen-test vault — hard wall

**Rule.** Clips from `omnifall`, `caucafall`, `mcfd`, `fallvision` may
**only** appear in `datasets/<vault_slug>/` on Drive, and they must
**only** be referenced from `frozen_unseen_test` rows in the manifest.
They must never be copied into `datasets/<debug_or_train_slug>/` paths.

**Why.** These datasets are the unseen-test wall for cross-dataset
generalisation numbers. If they leak into training or validation —
even accidentally, via a hardlink or a copy — every reported
cross-dataset number becomes optimistic and the result is unsalvageable.

**How to enforce.**
- The validator (`data/manifests/__init__.py:_check_frozen_vault_isolation`)
  rejects any vault-dataset clip assigned a non-`frozen_unseen_test` role.
- The validator rejects any `frozen_unseen_test` row from a non-vault
  dataset (the role is reserved for the vault).
- This file documents the path discipline: vault datasets live under
  their own slugs; no script moves them under another slug.

### 3. Licensed / gated datasets — manual staging only

**Rule.** Datasets that require accepting a license, filling a form,
or signing a click-through EULA must **not** be fetched from a public
script. They are staged manually by the project owner into
`datasets/<slug>/` on Drive, and the manifest references the resulting
local paths.

**Why.** Auto-downloading licensed material that the user has not
agreed to is a license violation. Click-through EULAs are user-bound,
not machine-bound, and the user must do them.

**How to enforce.**
- Each licensed dataset gets an entry in the table below with the
  acquisition URL and the license summary.
- The manifest's `notes` field on each clip can cite the license
  and acquisition date so audits are possible.

Current licensed datasets:

| Slug | Source URL | License | Manual step required |
|---|---|---|---|
| urfd | `tanmaydacha/urfd-dataset` on Kaggle | (research-use) | Set `KAGGLE_USERNAME` + `KAGGLE_KEY` in Colab Secrets; staging is automated via `data/stage_urfd.py` |
| up_fall | (publisher site) | (research-use) | Accept terms on publisher site |
| omnifall | (publisher site) | (research-use) | Accept terms on publisher site |
| caucafall | (publisher site) | (research-use) | Accept terms on publisher site |
| mcfd | (publisher site) | (research-use) | Accept terms on publisher site |
| fallvision | (publisher site) | (research-use) | Accept terms on publisher site |

*(Populated when each dataset is first staged. Empty rows above are a
flag for the next person doing data acquisition.)*

### 4. No raw dataset material in version control

**Rule.** Raw videos, intermediate frame dumps, and tar shards of
per-clip data must **not** be committed to git. The `.gitignore` at
the repo root excludes them implicitly via the `data/` and `artifacts/`
conventions; this README makes the convention explicit.

**Why.** A single multi-GB video file in git history bloats every
clone forever. Drive is the storage layer for big data; git is the
storage layer for source, manifests, and small metadata.

**How to enforce.**
- Git tracks only: the manifest files (`data/manifests/*.yaml`),
  the validator code, the schema, and this README.
- Heavy artefacts (clips, skeletons, checkpoints, env lock) live on
  Drive. The Colab setup writes them there; this repo points at them
  by path in the manifest.

### 5. Manifest paths must be Drive-relative

**Rule.** Every `source_path` in the manifest is **relative to
`MyDrive/fall_detection/`**, not absolute. e.g. `datasets/urfd/clips/fall-01.mp4`,
not `/content/drive/MyDrive/fall_detection/datasets/...`.

**Why.** Absolute paths break the moment Drive is mounted at a different
prefix (e.g. on a different machine, or after Drive layout changes).
Drive-relative paths let any session resolve them via the
`FALL_DETECTION_DRIVE_ROOT` env var that `colab/setup.py` honours.

**How to enforce.**
- `data/manifests/sample_manifest.yaml` demonstrates the convention.
- Issue 003's clip generator resolves manifest paths against the
  Drive layout root, so absolute paths would be rejected at runtime.

---

## How to add a new dataset

1. Stage it once on Drive under `MyDrive/fall_detection/datasets/<slug>/`.
2. Add the slug to `data/manifests/__init__.py` (either
   `IN_SCOPE_DATASETS` or `FROZEN_VAULT_DATASETS`, never both).
3. Add manifest rows referencing the dataset, with `source_path` set
   relative to `MyDrive/fall_detection/`.
4. Run `python -m unittest tests.test_manifest_validator` and confirm
   the new rows pass.
5. If the dataset is licensed or gated, add a row to the
   *Licensed datasets* table above.
6. Update `context.txt` (decisions / open issues / assumptions).

---

## URFD — Issue 002 staging rules

URFD is the first real dataset the project touches (Issue 002). It has
its own staging rules on top of the generic rules above because it
arrives as **ordered PNG frame folders**, not videos.

### Where URFD lives on Drive

```
MyDrive/fall_detection/datasets/urfd/
├── fall-01-cam0/                 # one folder per (sequence, camera)
├── fall-01-cam1/
├── adl-01-cam0/                  # ADL = activities of daily living (no fall)
├── adl-01-cam1/
├── ...
├── .staged_from_kaggle.txt       # provenance marker
└── manifest.yaml                 # auto-generated debug manifest (schema 1.1)
```

### Folder-name convention

| Prefix | Role | Label |
|---|---|---|
| `fall-NN-camM` | debug | `fall` |
| `adl-NN-camM`  | debug | `no_fall` |

The trailing `-camM` is the camera index (URFD has two cameras per
scene). The parser is case-insensitive and lives in
`data/stage_urfd.py:parse_urfd_folder_name`.

### Frame ordering — CRITICAL

URFD PNG frames are named like `frame_0001.png`, `frame_0042.png`, etc.
The tracker contract requires them in temporal order. Out-of-order
frames produce silently invalid tracks (the bug is invisible until you
visualise the annotated output).

**Always sort numerically before tracking.** The loader does this for
you — see `perception/frames.py:FrameFolderReader`. Do not bypass it.

### How URFD gets there

1. Add two secrets in Colab Secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY`.
   **Never** hardcode them, write them to disk, or commit them.
2. Run `colab/002_perception_urfd.ipynb` step 4 — it calls
   `data.stage_urfd.stage_urfd_from_kaggle(drive_root)`.
3. The script downloads **only** from the whitelisted slug
   `tanmaydacha/urfd-dataset`. Any other slug raises before any network call.
4. On success, a `.staged_from_kaggle.txt` marker file is written so
   re-runs short-circuit and never re-download.

Future sessions must reuse the staged Drive copy — no re-downloads.

### URFD credential rule

- Credentials live in **Colab Secrets** (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
- The staging script reads them, sets two environment variables, and
  returns. It never `print()`s the values, never writes them to a file,
  never returns them from any function, never logs them.
- The provenance marker file records only the **slug** and the source
  ("colab_secrets"), not the credential values.
- If you run outside Colab, the staging script looks for
  `~/.kaggle/kaggle.json` on disk instead. We do NOT auto-create it.

### Honest-metric note (URFD has no ground truth)

URFD ships with raw frames only — no detection or tracking labels. The
perception front-end therefore can NOT report mAP, IDF1, MOTA, or HOTA
as numbers. The report records them as:

```json
{
  "map_50":     "n/a (no detection ground truth)",
  "idf1":       "n/a (no tracking ground truth)",
  "mota":       "n/a (no tracking ground truth)",
  "hota":       "n/a (no tracking ground truth)"
}
```

Debug-tier validation is done by **annotated visual review** plus the
track-continuity / fragmentation report. Per-clip annotated frames live
at `artifacts/perception/<clip_id>/annotated/` on Drive.

---

## What does NOT live in Drive

- Source code (this repo).
- The Colab env lock and run log *do* live on Drive (under
  `artifacts/` and `logs/`); they are produced by the setup notebook
  in a real Colab session and are NOT fabricated locally. If you run
  tests outside Colab you will not have these files — that's expected.
- This README, the manifest schema, and the validator live in the repo.