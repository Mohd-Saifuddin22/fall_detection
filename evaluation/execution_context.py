"""Frozen-tier read guard.

The ``frozen_unseen_test`` role is the project's hard wall — it holds
vault datasets (OmniFall, CAUCAFall, MCFD, FallVision) that may ONLY
be read by code whose ``ExecutionContext`` permits it. Any other
context must fail closed: training/tuning/evaluation runs must never
see a frozen clip, and unknown / missing contexts must also be denied
frozen access (so a typo or default ``None`` argument cannot
accidentally leak frozen data into the training loop).

Governance rule (applied after Step 1's initial draft)
-------------------------------------------------------

Only ``FINAL_JUDGEMENT`` may read ``frozen_unseen_test`` records.
``EVALUATION`` — which previously had read access for ad-hoc
operator spot-checks — is now denied. ``EVALUATION`` operates on
``debug`` / ``train`` / ``validate`` tiers only; the frozen vault
opens only when the project is delivering the cross-dataset
generalisation numbers (i.e. the final judgement).

This is a strict widening of the deny-side set, not a relaxation:
any code path that previously read frozen under ``EVALUATION`` now
``FrozenAccessError``s. The fix is to either (a) change the call
site's ``ExecutionContext`` to ``FINAL_JUDGEMENT``, or (b) confirm
that the data the call site actually needs is on the
debug / train / validate tiers.

Reuses (does not duplicate) manifest primitives:

- :class:`data.manifests.ClipRole`
- :class:`data.manifests.ClipRole.FROZEN_UNSEEN_TEST`
- :class:`data.manifests.Manifest.by_role`
- The existing :func:`data.manifests.validate_manifest` behaviour
  (vault isolation is enforced upstream; the guard assumes the
  manifest is already well-formed and adds a second layer against
  accidental context misuse).

Public surface
--------------

- :class:`ExecutionContext` — enum: ``TRAINING`` / ``TUNING`` /
  ``EVALUATION`` / ``FINAL_JUDGEMENT`` / ``UNKNOWN``.
- :data:`FROZEN_ALLOWED_CONTEXTS` — only ``FINAL_JUDGEMENT`` may
  read frozen records. Single-entry allow-list by design.
- :class:`FrozenAccessError` — raised when access is denied.
- :func:`select_clips_for_context` — return clips appropriate to
  ``context`` (always filtering frozen out for non-allowed contexts).
- :func:`get_frozen_clips` — explicit frozen-tier accessor; raises
  for non-allowed contexts.
- :func:`coerce_execution_context` — tolerate ``None`` / strings /
  enum values without silently defaulting to a privileged context.

Fails closed
------------

- ``UNKNOWN`` (an explicit "I don't know what I'm doing" sentinel)
  is treated as a non-allowed context.
- ``None`` and any value not in the enum are coerced to ``UNKNOWN``
  via :func:`coerce_execution_context`, which has the same effect.
- There is no constructor / convenience that bypasses the guard;
  every read goes through one of the two entry points.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from data.manifests import ClipRecord, ClipRole, Manifest


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


class ExecutionContext(str, Enum):
    """Where in the pipeline a piece of code is running.

    The four real contexts answer: "is this code allowed to see
    ``frozen_unseen_test`` records?"

    - ``TRAINING``      — model fitting. MUST NOT see frozen.
    - ``TUNING``        — hyperparameter sweeps on ``validate``.
                          MUST NOT see frozen.
    - ``EVALUATION``    — diagnostic / validation-set metrics.
                          Operates only on debug / train / validate
                          tiers. MUST NOT see frozen (governance
                          rule: the vault opens only on final
                          judgement, not on every eval call).
    - ``FINAL_JUDGEMENT`` — the project-stated cross-dataset
                          generalisation numbers. The ONLY context
                          that may read frozen.
    - ``UNKNOWN``       — explicit "I don't know". Treated as
                          non-allowed so a typo / default ``None``
                          can never accidentally be promoted to
                          a privileged context.
    """

    TRAINING = "training"
    TUNING = "tuning"
    EVALUATION = "evaluation"
    FINAL_JUDGEMENT = "final_judgement"
    UNKNOWN = "unknown"


# Hard allow-list for frozen-tier access. Deliberately a
# SINGLE-entry set: the vault opens only for ``FINAL_JUDGEMENT``.
# Widening this list is a governance change that needs an explicit
# review — fail closed until then.
FROZEN_ALLOWED_CONTEXTS: frozenset[ExecutionContext] = frozenset({
    ExecutionContext.FINAL_JUDGEMENT,
})


class FrozenAccessError(RuntimeError):
    """Raised when a context tries to read ``frozen_unseen_test`` records."""


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def coerce_execution_context(value: object) -> ExecutionContext:
    """Coerce ``value`` to an :class:`ExecutionContext`.

    Accepts:

    - An :class:`ExecutionContext` instance — returned unchanged.
    - A string equal to one of the enum values (case-insensitive).
    - ``None`` or any other value — coerced to ``UNKNOWN`` so the
      caller can never accidentally inherit a privileged context
      via a default ``None`` or a typo.

    Coercing non-enum values to ``UNKNOWN`` (rather than raising) is
    intentional: in a notebook / pipeline that uses string flags
    liberally, raising on a typo would be silent; returning
    ``UNKNOWN`` lets the guard deny the access and the user discover
    the misconfiguration when they query the result.
    """
    if isinstance(value, ExecutionContext):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        for member in ExecutionContext:
            if member.value == lowered:
                return member
    return ExecutionContext.UNKNOWN


def is_frozen_allowed(context: ExecutionContext) -> bool:
    """``True`` iff ``context`` is allowed to read frozen records.

    With the governance rule in effect, this is True for exactly
    ``ExecutionContext.FINAL_JUDGEMENT``.
    """
    return context in FROZEN_ALLOWED_CONTEXTS


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def select_clips_for_context(
    manifest: Manifest,
    context: object,
) -> list[ClipRecord]:
    """Return clips appropriate for ``context``.

    Frozen-tier access is filtered out for non-allowed contexts —
    including ``UNKNOWN``. If the caller wants explicit frozen access,
    use :func:`get_frozen_clips` (which raises on denial) so the
    "I asked for frozen and I got it" intent is captured at the call
    site.

    Args:
        manifest: A well-formed :class:`Manifest`. The function does
            not re-validate; the upstream :func:`validate_manifest`
            is responsible for "vault isolation" / leakage. This
            function adds a second layer against accidental context
            misuse.
        context: Any value :func:`coerce_execution_context` accepts.

    Returns:
        The full list of clips for allowed contexts; the manifest
        minus ``frozen_unseen_test`` rows otherwise.
    """
    resolved = coerce_execution_context(context)
    if is_frozen_allowed(resolved):
        return list(manifest.clips)
    return [c for c in manifest.clips if c.role is not ClipRole.FROZEN_UNSEEN_TEST]


def get_frozen_clips(
    manifest: Manifest,
    context: object,
) -> list[ClipRecord]:
    """Explicit frozen-tier accessor.

    Differentiated from :func:`select_clips_for_context` so a training
    loop that accidentally calls a frozen-aware function fails LOUD
    (with a clear denial error) rather than silently receiving the
    non-frozen subset.

    Raises:
        FrozenAccessError: if ``context`` is not in
            :data:`FROZEN_ALLOWED_CONTEXTS` — under the governance
            rule, this fires for every context except
            :attr:`ExecutionContext.FINAL_JUDGEMENT`.
    """
    resolved = coerce_execution_context(context)
    if not is_frozen_allowed(resolved):
        allowed_labels = sorted(c.value for c in FROZEN_ALLOWED_CONTEXTS)
        raise FrozenAccessError(
            f"ExecutionContext {resolved.value!r} cannot read frozen_unseen_test records. "
            f"Only {allowed_labels} may. Refusing to return frozen clips."
        )
    # Defensive copy so callers cannot mutate manifest state through
    # the returned list — by_role already returns a fresh list but we
    # repeat the protection to make the freeze explicit.
    return list(manifest.by_role(ClipRole.FROZEN_UNSEEN_TEST))


def frozen_clips_present(manifest: Manifest) -> bool:
    """``True`` iff ``manifest`` carries at least one frozen clip.

    Convenience helper for diagnostics: lets a caller assert
    "this manifest actually has vault records" before deciding that
    a frozen-aware call is meaningful.
    """
    return any(c.role is ClipRole.FROZEN_UNSEEN_TEST for c in manifest.clips)


def enforce_no_frozen_in_iterable(
    clips: Iterable[ClipRecord],
    context: object,
) -> None:
    """Raise if ``clips`` carries a frozen record and ``context`` is not allowed.

    Use this when a code path accepted a list of clips without going
    through :func:`select_clips_for_context` — e.g. a test fixture, a
    manual notebook shuffle — and you want to assert at the boundary
    that no frozen leakage slipped in.

    Raises:
        FrozenAccessError: if a frozen clip is found and ``context``
            is not in :data:`FROZEN_ALLOWED_CONTEXTS` — under the
            governance rule, this fires for every context except
            :attr:`ExecutionContext.FINAL_JUDGEMENT`.
    """
    resolved = coerce_execution_context(context)
    if is_frozen_allowed(resolved):
        return
    for clip in clips:
        if clip.role is ClipRole.FROZEN_UNSEEN_TEST:
            raise FrozenAccessError(
                f"ExecutionContext {resolved.value!r} cannot read frozen_unseen_test record "
                f"{clip.clip_id!r}. Refusing."
            )


__all__: tuple[str, ...] = (
    "ExecutionContext",
    "FROZEN_ALLOWED_CONTEXTS",
    "FrozenAccessError",
    "coerce_execution_context",
    "enforce_no_frozen_in_iterable",
    "frozen_clips_present",
    "get_frozen_clips",
    "is_frozen_allowed",
    "select_clips_for_context",
)

