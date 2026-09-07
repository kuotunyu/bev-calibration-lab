"""Small inline SVG figures derived solely from claim-bound validity values."""

from html import escape


def validity_bar(invalid_rate: float, claim_id: str) -> str:
    return f'<svg viewBox="0 0 120 12" width="120" height="12" role="img" aria-label="{escape(claim_id)}"><rect width="120" height="12" rx="3" fill="#dce5eb"/><rect width="{120 * invalid_rate}" height="12" rx="3" fill="#b24b35"/></svg>'
