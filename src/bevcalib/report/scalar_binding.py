"""Exact display bindings shared by tables and figures after whole-set audit."""

from collections.abc import Mapping
from decimal import Decimal

from bevcalib.analysis.claims import ClaimV1


def bind_scalar(
    document: str,
    pointer: str,
    value: int | float | None,
    document_sha256: str,
    scalar_claims: Mapping[tuple[str, str], ClaimV1],
) -> str | None:
    """Bind a source-snapshot scalar; callers still audit paths and whole-set identity.

    Null has no numerical claim. Decimal string equality preserves the existing
    table contract without introducing display rounding as an acceptance tolerance.
    """
    if value is None:
        return None
    key = (document, pointer)
    if key not in scalar_claims:
        raise ValueError(f"unclaimed displayed formal scalar: {document}{pointer}")
    claim = scalar_claims[key]
    binding = claim.report_binding
    if (
        claim.status != "verified"
        or claim.metric_path != pointer
        or binding is None
        or binding.expected_summary_sha256 != document_sha256
        or Decimal(str(binding.expected_value)) != Decimal(str(value))
    ):
        raise ValueError("formal report scalar differs from verified source snapshot")
    return claim.claim_id
