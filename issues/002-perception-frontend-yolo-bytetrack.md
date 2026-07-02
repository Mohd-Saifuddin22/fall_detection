## Parent PRD

`issues/prd.md`

## What to build

The shared perception front-end that turns a raw video into per-person tracks: **YOLO** detects every
person per frame, **ByteTrack** assigns a stable ID to the same person across frames. Output is
`{track_id → ordered frames/boxes}` plus an annotated video (boxes + track IDs) for visual review.
Validate on the debug tier — critically, that a track **survives the fall motion itself**, since a
broken track destroys every downstream clip. This is the cheapest killer risk to retire before any
pipeline is built.

See PRD → *Solution* (shared front-end), *Modules (1)*, *Phase I decisions*, *Component evaluation*,
*Further Notes* (tracking-through-the-fall risk). Uses pretrained models — not trained from scratch.

## Acceptance criteria

- [ ] Given a debug-tier video, produces per-person tracks with consistent IDs and an annotated output video for visual review.
- [ ] Tracking holds a stable ID through a fall event on debug clips (no ID switch at the moment of collapse) in the majority of cases; failure cases are logged.
- [ ] Component metrics reported: YOLO (mAP@0.5, mAP@0.5:0.95, person precision/recall) and ByteTrack (ID switches, fragmentation; IDF1/MOTA/HOTA where computable).
- [ ] Uses pretrained YOLO + ByteTrack (no from-scratch training).
- [ ] Runs within the Colab GPU; the observed GPU type/VRAM is recorded.

## Blocked by

- Blocked by `issues/001-colab-env-and-datasets.md`

## User stories addressed

- User story 5
- User story 6
- User story 7
- User story 21
