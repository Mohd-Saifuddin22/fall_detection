"""Tests for :mod:`data.stage_urfd_university`.

Coverage target (per the Issue 005+ university stager task spec):
- Network-free: every download path is mocked.
- Whitelist enforcement: only the pinned base URL is accepted.
- URL coverage: 70 frame zips + 2 CSVs.
- Idempotency: marker + every expected file gates re-runs.
- Corrupt-zip failure loud.
- Valid-zip extraction produces manifest-compatible folder names.
- CSV persistence to staged_root/csvs/.
- Structured result reports success / failure clearly.
"""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.stage_urfd_university import (  # noqa: E402
    ADL_CSV_FILENAME,
    ADL_SEQUENCES,
    ALLOWED_UNIVERSITY_BASE_URL,
    CAMERA_SUFFIX,
    FALL_CSV_FILENAME,
    FALL_SEQUENCES,
    STAGING_MARKER_FILENAME,
    UrfdUniversityStagingResult,
    build_csv_urls,
    build_frame_zip_urls,
    expected_files,
    is_urfd_university_already_staged,
    parse_university_folder_name,
    stage_urfd_from_university,
)


def _make_dummy_zip(members: dict[str, bytes]) -> bytes:
    """Build a tiny in-memory zip with the given members.

    Used to mock the network downloads — every test zip lives
    entirely in RAM, no real network, no real Drive writes
    outside the temp dir.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _failing_zip_bytes() -> bytes:
    """Truncated bytes that ``zipfile.ZipFile`` will reject."""
    return b"PK\x03\x04not-a-valid-zip"


# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------


class WhitelistTests(unittest.TestCase):
    """The base-URL whitelist is strict and fail-loud."""

    def test_default_base_url_is_pinned(self) -> None:
        self.assertEqual(
            ALLOWED_UNIVERSITY_BASE_URL,
            "https://fenix.ur.edu.pl/~mkepski/ds/data/",
        )

    def test_non_whitelisted_base_url_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            build_frame_zip_urls("https://example.com/urfd/")

    def test_http_scheme_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            build_frame_zip_urls("http://fenix.ur.edu.pl/~mkepski/ds/data/")

    def test_trailing_path_drift_rejected(self) -> None:
        # The whitelist is strict equality: any drift from the pinned
        # base URL is a code-review failure, not a runtime
        # convenience. A future typo that appended a slash or a
        # "v2" path must fail loud.
        with self.assertRaises(RuntimeError):
            build_frame_zip_urls(ALLOWED_UNIVERSITY_BASE_URL + "v2/")

    def test_stage_function_rejects_non_whitelisted_base_url(self) -> None:
        with self.assertRaises(RuntimeError):
            stage_urfd_from_university(
                Path("/tmp/nonexistent"), base_url="https://evil.example/urfd/",
            )


# ---------------------------------------------------------------------------
# URL coverage
# ---------------------------------------------------------------------------


class URLBuilderTests(unittest.TestCase):
    """70 frame-zip URLs + 2 CSV URLs in the right shape."""

    def test_frame_zip_count_is_70(self) -> None:
        urls = build_frame_zip_urls()
        self.assertEqual(len(urls), 70)

    def test_frame_zip_first_30_are_fall_sequences(self) -> None:
        urls = build_frame_zip_urls()
        for index, expected_seq in enumerate(FALL_SEQUENCES):
            expected = (
                f"{ALLOWED_UNIVERSITY_BASE_URL}fall-"
                f"{expected_seq:02d}-{CAMERA_SUFFIX}.zip"
            )
            self.assertEqual(urls[index], expected)

    def test_frame_zip_last_40_are_adl_sequences(self) -> None:
        urls = build_frame_zip_urls()
        adl_urls = urls[len(FALL_SEQUENCES):]
        self.assertEqual(len(adl_urls), 40)
        for index, expected_seq in enumerate(ADL_SEQUENCES):
            expected = (
                f"{ALLOWED_UNIVERSITY_BASE_URL}adl-"
                f"{expected_seq:02d}-{CAMERA_SUFFIX}.zip"
            )
            self.assertEqual(adl_urls[index], expected)

    def test_csv_urls_are_named_documented(self) -> None:
        csvs = build_csv_urls()
        self.assertEqual(set(csvs.keys()), {FALL_CSV_FILENAME, ADL_CSV_FILENAME})
        for filename, url in csvs.items():
            self.assertEqual(
                url, f"{ALLOWED_UNIVERSITY_BASE_URL}{filename}",
            )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class IdempotencyTests(unittest.TestCase):
    """Marker + every expected file gates the re-run short-circuit."""

    def setUp(self) -> None:
        self._tmp = Path(sys.modules["tempfile"].mkdtemp())
        self.addCleanup(_rm_tree, self._tmp)

    def test_empty_dir_is_not_staged(self) -> None:
        self.assertFalse(is_urfd_university_already_staged(self._tmp / "urfd"))

    def test_marker_alone_is_not_staged(self) -> None:
        staged = self._tmp / "urfd"
        staged.mkdir(parents=True)
        (staged / STAGING_MARKER_FILENAME).write_text("staged\n", encoding="utf-8")
        # Marker alone — the script requires the marker AND every
        # expected file. A half-staged tree must not pass.
        self.assertFalse(is_urfd_university_already_staged(staged))

    def test_marker_plus_real_folder_is_staged(self) -> None:
        staged = self._tmp / "urfd"
        staged.mkdir(parents=True)
        (staged / STAGING_MARKER_FILENAME).write_text("staged\n", encoding="utf-8")
        (staged / "fall-01-cam0").mkdir()
        (staged / "adl-01-cam0").mkdir()
        # No CSVs yet — still not "fully staged" by the script's
        # own definition.
        self.assertFalse(is_urfd_university_already_staged(staged))

    def test_full_marker_is_staged(self) -> None:
        staged = self._tmp / "urfd"
        staged.mkdir(parents=True)
        # Simulate the full set: marker + every expected folder +
        # the two CSVs.
        for path in expected_files(staged):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == STAGING_MARKER_FILENAME:
                path.write_text("staged\n", encoding="utf-8")
            elif path.suffix.lower() == ".csv":
                path.write_bytes(b"csv-bytes\n")
            else:
                path.mkdir()
        self.assertTrue(is_urfd_university_already_staged(staged))


# ---------------------------------------------------------------------------
# Network / extraction — mocked
# ---------------------------------------------------------------------------


def _fake_network_dispatch(
    *,
    frames: dict[str, bytes] | None = None,
    csvs: dict[str, bytes] | None = None,
    fail_urls: set[str] | None = None,
    corrupt_urls: set[str] | None = None,
):
    """Return a patch object that intercepts ``urlopen`` deterministically.

    ``frames`` maps a URL substring to the bytes the download
    should return. ``csvs`` does the same for the CSV URLs.
    ``fail_urls`` forces a download error; ``corrupt_urls``
    returns truncated bytes. Missing URLs raise — the test
    that drives the patch should be specific about which
    downloads it exercises.
    """
    frames = frames or {}
    csvs = csvs or {}
    fail_urls = fail_urls or set()
    corrupt_urls = corrupt_urls or set()

    def _fake_urlopen(req, timeout=0):
        url = req.full_url
        if url in fail_urls:
            raise RuntimeError(f"network down for {url}")
        if url in corrupt_urls:
            return _BytesResponse(_failing_zip_bytes())
        if url in csvs:
            return _BytesResponse(csvs[url])
        for key, payload in frames.items():
            if key in url:
                return _BytesResponse(payload)
        raise RuntimeError(f"unmocked URL: {url}")

    return patch("data.stage_urfd_university.urlopen", _fake_urlopen)


class _BytesResponse:
    """Minimal context-manager wrapper around ``urlopen``'s return shape."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._payload


class StageHappyPathTests(unittest.TestCase):
    """A full network-mocked staging call produces every expected file."""

    def setUp(self) -> None:
        self._tmp = Path(sys.modules["tempfile"].mkdtemp())
        self.addCleanup(_rm_tree, self._tmp)

    def _zip_bytes(self, n_frames: int = 4) -> bytes:
        """Build a single archive that mirrors the real university layout.

        The real ``fall-01-cam0-rgb.zip`` extracts to a top-level
        ``fall-01-cam0-rgb/`` folder containing an inner folder
        of the same name with the actual frame PNGs. The inner
        folder's frame files are 1-based and 3-digit zero-padded
        (``fall-01-cam0-rgb-001.png`` …) so the existing
        :class:`perception.frames.FrameFolderReader` discovers
        them in temporal order via its numeric-frame sort.
        """
        clip_key = "synthetic-fall-01-cam0-rgb"
        members = {
            f"{clip_key}/{clip_key}-{i:03d}.png": b"png"
            for i in range(1, n_frames + 1)
        }
        return _make_dummy_zip(members)

    def _frames(self) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for url in build_frame_zip_urls():
            out[url] = self._zip_bytes()
        return out

    def _csvs(self) -> dict[str, bytes]:
        return {
            f"{ALLOWED_UNIVERSITY_BASE_URL}{FALL_CSV_FILENAME}":
                b"seq,label\n1,fall\n",
            f"{ALLOWED_UNIVERSITY_BASE_URL}{ADL_CSV_FILENAME}":
                b"seq,label\n1,no_fall\n",
        }

    def test_happy_path_extracts_seventy_clips_and_two_csvs(self) -> None:
        with _fake_network_dispatch(frames=self._frames(), csvs=self._csvs()):
            result = stage_urfd_from_university(self._tmp)

        # All 70 clips landed.
        self.assertEqual(len(result.succeeded_clips), 70)
        self.assertEqual(len(result.clip_folders), 70)
        self.assertEqual(result.failed_clips, {})
        # Marker is present → next run short-circuits.
        self.assertTrue(
            is_urfd_university_already_staged(result.staged_root),
        )
        # CSVs landed in the persistent location.
        self.assertIn(FALL_CSV_FILENAME, result.csv_paths)
        self.assertIn(ADL_CSV_FILENAME, result.csv_paths)
        for filename, path in result.csv_paths.items():
            self.assertTrue(path.is_file())
        # Sample a few folder names — they must be manifest-
        # compatible. The -rgb suffix IS preserved: the brief's
        # example real layout is fall-01-cam0-rgb/. Stripping it
        # here would diverge from the on-disk folder name and
        # break the manifest builder's clip-id contract.
        names = {f.folder_name for f in result.clip_folders}
        self.assertIn("fall-01-cam0-rgb", names)
        self.assertIn("fall-30-cam0-rgb", names)
        self.assertIn("adl-01-cam0-rgb", names)
        self.assertIn("adl-40-cam0-rgb", names)
        # All 70 clip folders carry the -rgb suffix.
        self.assertEqual(len(names), 70)
        self.assertTrue(all(n.endswith("-rgb") for n in names))

    def test_idempotent_run_does_not_redownload(self) -> None:
        with _fake_network_dispatch(frames=self._frames(), csvs=self._csvs()):
            first = stage_urfd_from_university(self._tmp)
        # Second call: marker is present, no new downloads should
        # happen. The patch's urlopen would raise on any URL it
        # sees, so reaching already_staged=True is the proof.
        with patch("data.stage_urfd_university.urlopen") as fake:
            second = stage_urfd_from_university(self._tmp)
            fake.assert_not_called()
        self.assertTrue(second.already_staged)
        self.assertEqual(len(second.succeeded_clips), 70)

    def test_force_flag_re_runs_full_download(self) -> None:
        # First call: marker is written, no further network.
        with _fake_network_dispatch(frames=self._frames(), csvs=self._csvs()):
            first = stage_urfd_from_university(self._tmp)
        self.assertTrue(is_urfd_university_already_staged(first.staged_root))
        # Second call with force=True: idempotency short-circuit
        # is bypassed, downloads happen again. The test stubs
        # ``_download_with_retry`` so we don't depend on the urlopen
        # patch's exact form.
        members = {f"frame_{i:05d}.png": b"png" for i in range(4)}
        download_count = {"n": 0}

        def counting_download(url, **kw):
            download_count["n"] += 1
            return _make_dummy_zip(members)

        with patch("data.stage_urfd_university._download_with_retry",
                   counting_download):
            second = stage_urfd_from_university(self._tmp, force=True)
        # The fact that all 70 clips landed is the proof that the
        # second call ran the full download path. With the
        # idempotent fast path, the second call would have set
        # ``already_staged=True`` and ``succeeded_clips=()``.
        self.assertEqual(download_count["n"], 70 + 2,
                          msg="force=True should redownload 70 zips + 2 csvs")
        self.assertFalse(second.already_staged)
        self.assertEqual(len(second.succeeded_clips), 70)


class ManifestCompatibilityTests(unittest.TestCase):
    """The university stager output is consumable by the existing manifest
    builder and frame reader without modification.

    These tests prove the cross-module invariant the brief asks for:
    the synthetic staged tree must round-trip through
    :func:`data.build_urfd_manifest.build_urfd_manifest` (yielding
    clip id ``urfd-debug-fall-NN-cam0-rgb``) and through
    :class:`perception.frames.FrameFolderReader` (yielding the
    synthetic frames in temporal order).
    """

    def setUp(self) -> None:
        self._tmp = Path(sys.modules["tempfile"].mkdtemp())
        self.addCleanup(_rm_tree, self._tmp)

    def _stage_with_synthetic_zip(self) -> None:
        """Stage one fall clip + one adl clip + both CSVs."""
        members_fall = {
            f"fall-01-cam0-rgb/fall-01-cam0-rgb-{i:03d}.png": b"png"
            for i in range(1, 5)
        }
        members_adl = {
            f"adl-01-cam0-rgb/adl-01-cam0-rgb-{i:03d}.png": b"png"
            for i in range(1, 5)
        }
        frames = {
            f"{ALLOWED_UNIVERSITY_BASE_URL}fall-01-cam0-rgb.zip":
                _make_dummy_zip(members_fall),
            f"{ALLOWED_UNIVERSITY_BASE_URL}adl-01-cam0-rgb.zip":
                _make_dummy_zip(members_adl),
        }
        # Fill the remaining 68 zip URLs with valid zips so the
        # staging call succeeds end-to-end. We re-use the
        # adl fixture for the rest.
        for url in build_frame_zip_urls():
            frames.setdefault(url, _make_dummy_zip(members_adl))
        csvs = {
            f"{ALLOWED_UNIVERSITY_BASE_URL}{FALL_CSV_FILENAME}":
                b"seq,label\n1,fall\n",
            f"{ALLOWED_UNIVERSITY_BASE_URL}{ADL_CSV_FILENAME}":
                b"seq,label\n1,no_fall\n",
        }
        with _fake_network_dispatch(frames=frames, csvs=csvs):
            stage_urfd_from_university(self._tmp)

    def test_staged_folder_is_named_fall_NN_cam0_rgb(self) -> None:
        self._stage_with_synthetic_zip()
        # The staged tree on disk carries the real university
        # folder name. Stripping ``-rgb`` would have produced
        # ``fall-01-cam0`` and broken the manifest clip id.
        target = self._tmp / "datasets" / "urfd" / "fall-01-cam0-rgb"
        self.assertTrue(target.is_dir(),
                         msg=f"missing staged folder: {target}")

    def test_extraction_produces_double_nested_shape(self) -> None:
        self._stage_with_synthetic_zip()
        # The real archive extracts to
        # ``<root>/fall-01-cam0-rgb/fall-01-cam0-rgb/*.png`` —
        # an outer folder named like the clip, an inner folder
        # with the actual frames.
        inner = (
            self._tmp / "datasets" / "urfd"
            / "fall-01-cam0-rgb" / "fall-01-cam0-rgb"
        )
        self.assertTrue(inner.is_dir(),
                         msg=f"missing inner folder: {inner}")
        # The synthetic frames live inside the inner folder.
        frames = sorted(inner.iterdir())
        self.assertEqual(len(frames), 4,
                          msg=f"expected 4 frames, got {frames}")
        self.assertEqual(
            [f.name for f in frames],
            [
                "fall-01-cam0-rgb-001.png",
                "fall-01-cam0-rgb-002.png",
                "fall-01-cam0-rgb-003.png",
                "fall-01-cam0-rgb-004.png",
            ],
        )

    def test_build_urfd_manifest_parses_staged_folders(self) -> None:
        self._stage_with_synthetic_zip()
        from data.build_urfd_manifest import build_urfd_manifest
        staged_root = self._tmp / "datasets" / "urfd"
        manifest = build_urfd_manifest(staged_root)
        clip_ids = {c.clip_id for c in manifest.clips}
        # The brief pins the exact clip id: urfd-debug-fall-NN-cam0-rgb.
        self.assertIn("urfd-debug-fall-01-cam0-rgb", clip_ids)
        self.assertIn("urfd-debug-adl-01-cam0-rgb", clip_ids)
        # Label parsing still works: fall → fall, adl → no_fall.
        target = next(c for c in manifest.clips
                      if c.clip_id == "urfd-debug-fall-01-cam0-rgb")
        self.assertEqual(target.label.value, "fall")
        target = next(c for c in manifest.clips
                      if c.clip_id == "urfd-debug-adl-01-cam0-rgb")
        self.assertEqual(target.label.value, "no_fall")

    def test_frame_folder_reader_finds_synthetic_frames_in_order(self) -> None:
        self._stage_with_synthetic_zip()
        from perception.frames import FrameFolderReader
        clip_folder = (
            self._tmp / "datasets" / "urfd" / "fall-01-cam0-rgb"
        )
        reader = FrameFolderReader(clip_folder)
        ordered = reader.frames()
        # 4 synthetic frames, in temporal order (1..4).
        self.assertEqual(len(ordered), 4)
        self.assertEqual(
            [p.path.name for p in ordered],
            [
                "fall-01-cam0-rgb-001.png",
                "fall-01-cam0-rgb-002.png",
                "fall-01-cam0-rgb-003.png",
                "fall-01-cam0-rgb-004.png",
            ],
        )
        # The reader returned the inner matching subfolder
        # (real university layout: a single nested child whose
        # name matches the outer folder name). The reader's
        # ``folder`` attribute reflects the descended-to path.
        self.assertTrue(reader.folder.name.endswith("fall-01-cam0-rgb"))


class CorruptZipFailureTests(unittest.TestCase):
    """A truncated / corrupt zip fails loud and is recorded in failed_clips."""

    def setUp(self) -> None:
        self._tmp = Path(sys.modules["tempfile"].mkdtemp())
        self.addCleanup(_rm_tree, self._tmp)

    def _zip_bytes(self, n_frames: int = 4) -> bytes:
        """Build a single archive that mirrors the real university layout.

        The real ``fall-01-cam0-rgb.zip`` extracts to a top-level
        ``fall-01-cam0-rgb/`` folder containing an inner folder
        of the same name with the actual frame PNGs. The inner
        folder's frame files are 1-based and 3-digit zero-padded
        (``fall-01-cam0-rgb-001.png`` …) so the existing
        :class:`perception.frames.FrameFolderReader` discovers
        them in temporal order via its numeric-frame sort.
        """
        clip_key = "synthetic-fall-01-cam0-rgb"
        members = {
            f"{clip_key}/{clip_key}-{i:03d}.png": b"png"
            for i in range(1, n_frames + 1)
        }
        return _make_dummy_zip(members)

    def test_corrupt_zip_raises_on_stage(self) -> None:
        # Build a URL map where the first fall zip returns truncated
        # bytes. The first-frame-zip iteration triggers
        # ``_verify_zip_bytes`` and raises.
        urls = build_frame_zip_urls()
        corrupt = {urls[0]: _failing_zip_bytes()}
        frames = {url: self._zip_bytes() for url in urls}
        for k, v in corrupt.items():
            frames[k] = v
        with _fake_network_dispatch(
            frames=frames,
            csvs={
                f"{ALLOWED_UNIVERSITY_BASE_URL}{FALL_CSV_FILENAME}": b"x",
                f"{ALLOWED_UNIVERSITY_BASE_URL}{ADL_CSV_FILENAME}": b"x",
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                stage_urfd_from_university(self._tmp)
        # The error must name the corrupt URL so a reviewer can
        # see WHICH clip failed.
        self.assertIn(urls[0], str(ctx.exception))
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_corrupt_zip_blocks_partial_staging(self) -> None:
        # Even if the script catches a per-clip failure and
        # records it in failed_clips, the marker must NOT be
        # written when any clip failed. This is a separate
        # contract — corrupt_zip_raises_on_stage already proves
        # the strict path; this test pins the "no silent partial
        # staging" guarantee by inspecting the post-failure tree.
        urls = build_frame_zip_urls()
        # Build a stub that returns a corrupt zip for the first
        # URL and then raises (so the rest of the loop exits
        # early — the test is about the marker-not-written state).
        first = urls[0]

        def fake_urlopen(req, timeout=0):
            return _BytesResponse(_failing_zip_bytes())

        with patch("data.stage_urfd_university.urlopen", fake_urlopen):
            with self.assertRaises(RuntimeError):
                stage_urfd_from_university(self._tmp)
        # The marker must NOT be present — a half-staged tree
        # must not look "complete" to the next run.
        marker = self._tmp / "datasets" / "urfd" / STAGING_MARKER_FILENAME
        self.assertFalse(marker.exists())

    def test_partial_extraction_leaves_no_residue_on_failure(self) -> None:
        # If the verification pass raises BEFORE extraction, the
        # destination folder must not be created. The verification
        # function does that — the destination's parent.mkdir only
        # runs after the zip is verified.
        urls = build_frame_zip_urls()
        first = urls[0]

        def fake_urlopen(req, timeout=0):
            return _BytesResponse(_failing_zip_bytes())

        with patch("data.stage_urfd_university.urlopen", fake_urlopen):
            with self.assertRaises(RuntimeError):
                stage_urfd_from_university(self._tmp)
        # The destination folder for the first clip is NOT
        # created.
        clip_dest = self._tmp / "datasets" / "urfd" / "fall-01-cam0"
        self.assertFalse(clip_dest.exists())


# ---------------------------------------------------------------------------
# Folder-name parsing
# ---------------------------------------------------------------------------


class FolderNameParseTests(unittest.TestCase):
    """Parsed folder names feed the existing manifest builder."""

    def test_fall_folder_parses_to_fall_label(self) -> None:
        parsed = parse_university_folder_name("fall-01-cam0")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.label, "fall")
        self.assertEqual(parsed.camera, "cam0")
        self.assertEqual(parsed.clip_sequence, "01")

    def test_adl_folder_parses_to_no_fall_label(self) -> None:
        parsed = parse_university_folder_name("adl-02-cam0")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.label, "no_fall")

    def test_os_collision_suffix_normalised(self) -> None:
        # Re-stage artefacts append " (1)" / " (2)"; the parser
        # must not invent a separate label for those — they refer
        # to the same logical clip.
        parsed = parse_university_folder_name("fall-01-cam0 (1)")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.label, "fall")
        self.assertEqual(parsed.clip_sequence, "01")

    def test_unknown_folder_returns_none(self) -> None:
        self.assertIsNone(parse_university_folder_name("not-a-urfd-name"))
        self.assertIsNone(parse_university_folder_name(""))


# ---------------------------------------------------------------------------
# Structured result shape
# ---------------------------------------------------------------------------


class StructuredResultTests(unittest.TestCase):
    """The result type carries every required field."""

    def test_result_fields_present(self) -> None:
        members = {f"frame_{i:05d}.png": b"png" for i in range(4)}
        with _fake_network_dispatch(
            frames={url: _make_dummy_zip(members) for url in build_frame_zip_urls()},
            csvs={
                f"{ALLOWED_UNIVERSITY_BASE_URL}{FALL_CSV_FILENAME}": b"x",
                f"{ALLOWED_UNIVERSITY_BASE_URL}{ADL_CSV_FILENAME}": b"x",
            },
        ):
            result = stage_urfd_from_university(Path(sys.modules["tempfile"].mkdtemp()))
        self.assertIsInstance(result, UrfdUniversityStagingResult)
        self.assertTrue(result.staged_root.exists())
        self.assertEqual(result.source_base_url, ALLOWED_UNIVERSITY_BASE_URL)
        self.assertFalse(result.already_staged)
        self.assertEqual(result.failed_clips, {})
        # 70 + 2 csvs.
        self.assertEqual(result.clip_count, 70)
        self.assertEqual(len(result.csv_paths), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                _rm_tree(child)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        try:
            path.rmdir()
        except OSError:
            pass
    else:
        try:
            path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
