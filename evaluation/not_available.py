"""``NotAvailable`` machine-readable metric value type.

Used when a metric cannot be computed — e.g. mAP, IDF1, MOTA, HOTA,
false-alarms-per-hour, detection-delay — because the required ground
truth or temporal metadata is missing. The shape and behaviour:

- Distinct from numeric 0.0: a false negative is not the same signal
  as ``"could not compute"``. Conflating them silently buries data
  scarcity inside the headline numbers.
- Carries a reason string so a reader of the persisted payload knows
  WHY the metric is unavailable ("no detection ground truth",
  "no tracking ground truth", "missing temporal metadata", ...).
- Always falsy (``bool(na) is False``) so a quick guard like
  ``if result: ...`` correctly skips unavailable metrics.
- Hashable + dataclass-equal so it can live in sets and dict keys
  alongside numeric metrics without bespoke typing.
- Round-trips through JSON via :class:`NotAvailableJSONMarker` — a
  sentinel dict ``{"__not_available__": true, "reason": "..."}`` that
  the eval JSON decoder recognises on reload.

Why a dedicated type
--------------------
A bare ``None`` is too generic — every Python consumer treats ``None``
as an error to handle in its own way. A magic number (``float("nan")``)
is hard to grep for and easy to silently mis-aggregate. A dedicated
type makes the semantics explicit at the value layer, so metric
aggregation code, persistence, and reporting all share one shape.
"""

from __future__ import annotations

from dataclasses import dataclass


# The sentinel key the JSON form carries. Picked deliberately so it
# cannot collide with a plausible legitimate metric shape: it is
# spelled out, not numeric, and the value is the boolean ``True`` so
# no legitimate metric payload looks like this.
NOT_AVAILABLE_JSON_KEY: str = "__not_available__"


@dataclass(frozen=True)
class NotAvailable:
    """Marker that a metric could not be computed.

    Args:
        reason: Human-readable explanation of why the metric is
            unavailable. Examples: ``"no detection ground truth"``,
            ``"no tracking ground truth"``, ``"missing temporal
            metadata for event-level metric"``.
        metric_name: Optional. When set, identifies which metric this
            marker is for. Useful when the marker is aggregated into a
            shared list so a reader can pair marker → metric name
            even after a shuffle.

    Two :class:`NotAvailable` instances with the same ``reason`` and
    ``metric_name`` compare equal and hash equal — they are
    interchangeable.
    """

    reason: str
    metric_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                f"NotAvailable.reason must be a non-empty string, got {self.reason!r}."
            )
        if self.metric_name is not None and not isinstance(self.metric_name, str):
            raise ValueError(
                f"NotAvailable.metric_name must be a string or None, "
                f"got {type(self.metric_name).__name__}."
            )

    # ------------------------------------------------------------------
    # Falsy behaviour
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        # ``NotAvailable`` is always falsy, so:

        # - ``bool(na)`` is False.
        # - ``if not_available: ...`` is False (skip the "metric
        #   is real" branch).
        # - ``if not not_available: ...`` is True (enter the "metric
        #   is missing / skip this branch" path).
        # - ``if not_available.value: ...`` is False when ``value``
        #   is the marker, so a guard like
        #   ``if not result.value: continue`` correctly skips
        #   unavailable metrics.

        # This is the standard Python behaviour for a "null marker"
        # sentinel — same idea as ``None``, but typed and carrying a
        # reason so a downstream consumer can tell *why* the value
        # is absent.
        return False

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.metric_name:
            return f"NotAvailable({self.metric_name!r}, reason={self.reason!r})"
        return f"NotAvailable(reason={self.reason!r})"

    def __str__(self) -> str:
        return f"n/a ({self.reason})"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Render as a JSON-friendly marker dict."""
        payload: dict[str, object] = {
            NOT_AVAILABLE_JSON_KEY: True,
            "reason": self.reason,
        }
        if self.metric_name is not None:
            payload["metric_name"] = self.metric_name
        return payload


def is_not_available_marker(payload: object) -> bool:
    """``True`` iff ``payload`` looks like a :class:`NotAvailable` JSON marker.

    The check is intentionally narrow: it requires both
    ``__not_available__ == True`` and a non-empty ``reason``. Extra
    keys are allowed so future marker fields don't break old decoders.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get(NOT_AVAILABLE_JSON_KEY) is not True:
        return False
    reason = payload.get("reason")
    return isinstance(reason, str) and bool(reason)


def from_dict(payload: object) -> NotAvailable:
    """Inverse of :meth:`NotAvailable.to_dict`.

    Raises:
        ValueError: if ``payload`` does not look like a
            :class:`NotAvailable` marker.
    """
    if not is_not_available_marker(payload):
        raise ValueError(
            f"Cannot decode NotAvailable from payload: {payload!r}. "
            f"Expected a dict with {NOT_AVAILABLE_JSON_KEY!r}=True and a 'reason' string."
        )
    reason = str(payload["reason"])
    metric_name = payload.get("metric_name")
    if metric_name is not None:
        metric_name = str(metric_name)
    return NotAvailable(reason=reason, metric_name=metric_name)


__all__: tuple[str, ...] = (
    "NOT_AVAILABLE_JSON_KEY",
    "NotAvailable",
    "from_dict",
    "is_not_available_marker",
)
