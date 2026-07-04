"""Tests for :mod:`evaluation.not_available`.

Covers:

- The NotAvailable marker is distinct from numeric 0.0 (no equality).
- It carries a reason string and serialises clearly.
- ``bool(not_available) is False`` (always falsy so guard clauses work).
- Round-trip through the JSON marker shape via ``encode_value`` and ``from_dict``.
- Encoding helpers reject bogus shapes (None, bool-as-number).
"""

from __future__ import annotations

import json
import unittest

from evaluation.not_available import (
    NOT_AVAILABLE_JSON_KEY,
    NotAvailable,
    from_dict as na_from_dict,
    is_not_available_marker,
)


class NotAvailableShapeTests(unittest.TestCase):
    """NotAvailable is hashable, comparable, falsy, and structurally tight."""

    def test_carries_reason_string(self) -> None:
        na = NotAvailable(reason="no detection ground truth")
        self.assertEqual(na.reason, "no detection ground truth")

    def test_optional_metric_name_round_trips(self) -> None:
        na = NotAvailable(reason="missing temporal metadata", metric_name="false_alarms_per_hour")
        self.assertEqual(na.metric_name, "false_alarms_per_hour")

    def test_metric_name_none_is_valid(self) -> None:
        # No metric_name supplied — fine, marker is still usable.
        na = NotAvailable(reason="x")
        self.assertIsNone(na.metric_name)

    def test_not_equal_to_zero_point_zero(self) -> None:
        # This is the central guarantee. A missed metric is not the
        # same signal as ``the metric read 0.0``.
        na = NotAvailable(reason="no detection ground truth")
        self.assertFalse(na == 0.0)
        self.assertFalse(na == 0)

    def test_not_equal_to_none(self) -> None:
        na = NotAvailable(reason="x")
        self.assertFalse(na is None)
        self.assertFalse(na == None)  # noqa: E711

    def test_always_falsy(self) -> None:
        # bool(NotAvailable(...)) is False so ``if result: ...`` skips
        # unavailable values like a missing entry.
        self.assertFalse(bool(NotAvailable(reason="x")))
        self.assertFalse(bool(NotAvailable(reason="x", metric_name="m")))

    def test_equality_is_value_based(self) -> None:
        # Two markers with the same fields are interchangeable.
        a = NotAvailable(reason="x", metric_name="m")
        b = NotAvailable(reason="x", metric_name="m")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_inequality_when_reason_differs(self) -> None:
        a = NotAvailable(reason="x")
        b = NotAvailable(reason="y")
        self.assertNotEqual(a, b)

    def test_repr_mentions_reason(self) -> None:
        na = NotAvailable(reason="nope")
        self.assertIn("nope", repr(na))

    def test_repr_mentions_metric_name_when_set(self) -> None:
        na = NotAvailable(reason="nope", metric_name="map_50")
        self.assertIn("map_50", repr(na))

    def test_str_is_human_friendly(self) -> None:
        # Tests downstream rendering — a CSV / log line that includes
        # the value should clearly say it's not 0.0.
        rendered = str(NotAvailable(reason="no detection GT"))
        self.assertIn("n/a", rendered)
        self.assertIn("no detection GT", rendered)

    def test_empty_reason_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NotAvailable(reason="")

    def test_non_string_reason_is_rejected(self) -> None:
        # 123 would round-trip as the string "123" through YAML but we
        # hard-reject it at construction so the type is not silently
        # shaped by accident.
        with self.assertRaises(ValueError):
            NotAvailable(reason=123)  # type: ignore[arg-type]


class NotAvailableSerialisationTests(unittest.TestCase):
    """to_dict + from_dict round-trip cleanly, and JSON marker is identifiable."""

    def test_to_dict_uses_sentinel_key(self) -> None:
        na = NotAvailable(reason="no tracking GT")
        out = na.to_dict()
        self.assertIs(out[NOT_AVAILABLE_JSON_KEY], True)
        self.assertEqual(out["reason"], "no tracking GT")

    def test_to_dict_omits_metric_name_when_none(self) -> None:
        na = NotAvailable(reason="x")
        out = na.to_dict()
        self.assertNotIn("metric_name", out)

    def test_to_dict_includes_metric_name_when_set(self) -> None:
        na = NotAvailable(reason="x", metric_name="hota")
        out = na.to_dict()
        self.assertEqual(out["metric_name"], "hota")

    def test_is_not_available_marker_accepts_a_real_marker(self) -> None:
        na = NotAvailable(reason="x")
        self.assertTrue(is_not_available_marker(na.to_dict()))

    def test_is_not_available_marker_rejects_arbitrary_dicts(self) -> None:
        self.assertFalse(is_not_available_marker({"reason": "x"}))
        self.assertFalse(is_not_available_marker({NOT_AVAILABLE_JSON_KEY: True}))  # no reason
        self.assertFalse(is_not_available_marker({NOT_AVAILABLE_JSON_KEY: "yes"}))  # not bool
        self.assertFalse(is_not_available_marker("not-a-dict"))
        self.assertFalse(is_not_available_marker(None))
        self.assertFalse(is_not_available_marker([NOT_AVAILABLE_JSON_KEY, True]))

    def test_from_dict_round_trips_marker(self) -> None:
        original = NotAvailable(reason="missing temporal metadata", metric_name="delay")
        decoded = na_from_dict(original.to_dict())
        self.assertEqual(decoded, original)
        self.assertEqual(decoded.metric_name, "delay")
        self.assertEqual(decoded.reason, "missing temporal metadata")

    def test_from_dict_rejects_non_marker(self) -> None:
        with self.assertRaises(ValueError):
            na_from_dict({"reason": "x"})
        with self.assertRaises(ValueError):
            na_from_dict({"__not_available__": True, "reason": ""})
        with self.assertRaises(ValueError):
            na_from_dict({"__not_available__": True})  # no reason

    def test_json_dumps_keeps_marker_distinct_from_0(self) -> None:
        # End-to-end through stdlib json: a marker must not be
        # confused with ``0.0`` on reload. We rebuild the value from
        # the JSON form, do a sanity check that JSON preserved it as
        # an object (not a number) and that is_not_available_marker
        # recognises it.
        na = NotAvailable(reason="nope")
        encoded = json.dumps(na.to_dict())
        self.assertNotIn("0", encoded)
        self.assertNotIn("0.0", encoded)
        # Confirm encoded is a JSON object, not a JSON number.
        self.assertTrue(encoded.startswith("{"))
        decoded_payload = json.loads(encoded)
        self.assertTrue(is_not_available_marker(decoded_payload))

    def test_two_reasons_produce_distinct_encodings(self) -> None:
        # Two different reasons must NOT serialise to the same shape
        # so a downstream JSON diff can tell them apart. The reason
        # field is the source of the difference.
        a_json = json.dumps(NotAvailable(reason="a").to_dict(), sort_keys=True)
        b_json = json.dumps(NotAvailable(reason="b").to_dict(), sort_keys=True)
        self.assertNotEqual(a_json, b_json)


if __name__ == "__main__":
    unittest.main()
