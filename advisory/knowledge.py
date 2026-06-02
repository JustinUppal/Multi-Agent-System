"""The analyst's in-house knowledge store and the lookup tool over it.

A curated in-memory dataset of asset-class facts and model portfolios, standing
in for what would be a vector DB or house-research API in a real firm.
"""

from __future__ import annotations

# House view on broad asset classes. Figures are illustrative long-run averages,
# not advice, just enough for the analyst to reason about risk/return tradeoffs.
_ASSET_CLASSES: dict[str, dict[str, object]] = {
    "equities": {
        "summary": "Ownership stakes in companies; primary growth engine of a portfolio.",
        "expected_real_return_pct": 6.5,
        "volatility": "high",
        "typical_vehicles": ["broad index ETF (e.g. total-market)", "S&P 500 fund"],
        "notes": "Concentration in a single stock adds idiosyncratic risk without "
        "extra expected return; broad diversification is preferred.",
    },
    "bonds": {
        "summary": "Loans to governments/companies; income and ballast against equity drawdowns.",
        "expected_real_return_pct": 1.5,
        "volatility": "low-to-moderate",
        "typical_vehicles": ["aggregate bond ETF", "Treasury funds", "TIPS"],
        "notes": "Cushions volatility; allocation often scales up as the time horizon shortens.",
    },
    "cash": {
        "summary": "Money-market funds, high-yield savings, T-bills.",
        "expected_real_return_pct": 0.5,
        "volatility": "very low",
        "typical_vehicles": ["money-market fund", "high-yield savings", "T-bills"],
        "notes": "Good for an emergency fund and near-term needs; loses to inflation over long horizons.",
    },
    "real_estate": {
        "summary": "Property exposure, commonly via REITs for liquidity.",
        "expected_real_return_pct": 4.0,
        "volatility": "moderate-to-high",
        "typical_vehicles": ["REIT index fund"],
        "notes": "Diversifies an equity/bond mix; sensitive to interest rates.",
    },
}

# Rule-of-thumb model portfolios keyed by risk appetite. These mirror the
# qualitative labels used on ClientProfile.risk_aversion.
_MODEL_PORTFOLIOS: dict[str, dict[str, object]] = {
    "conservative": {
        "allocation": {"equities": 30, "bonds": 55, "cash": 15},
        "rationale": "Capital preservation first; modest growth.",
    },
    "moderate": {
        "allocation": {"equities": 60, "bonds": 30, "cash": 5, "real_estate": 5},
        "rationale": "Balanced growth and stability for a multi-decade horizon.",
    },
    "aggressive": {
        "allocation": {"equities": 85, "bonds": 10, "real_estate": 5},
        "rationale": "Maximize long-run growth; tolerate large drawdowns.",
    },
}


def lookup_investments(topic: str) -> dict:
    """Look up the firm's in-house research on an asset class or a model portfolio.

    Use this before reaching for web search when the question is about general
    investment principles, expected risk/return of an asset class, or a
    recommended allocation for a risk profile. It is faster and authoritative.

    Args:
        topic: An asset class (``equities``, ``bonds``, ``cash``, ``real_estate``)
            or a risk profile for a model portfolio (``conservative``,
            ``moderate``, ``aggressive``). Case-insensitive.

    Returns:
        A dict with a ``found`` flag. When found, it carries the matched
        ``asset_class`` or ``model_portfolio`` entry. When not found, it lists the
        available ``topics`` so the caller can retry with a valid one.
    """
    key = topic.strip().lower().replace(" ", "_")
    if key in _ASSET_CLASSES:
        return {"found": True, "kind": "asset_class", "topic": key, "data": _ASSET_CLASSES[key]}
    if key in _MODEL_PORTFOLIOS:
        return {"found": True, "kind": "model_portfolio", "topic": key, "data": _MODEL_PORTFOLIOS[key]}
    return {
        "found": False,
        "topics": {
            "asset_classes": sorted(_ASSET_CLASSES),
            "model_portfolios": sorted(_MODEL_PORTFOLIOS),
        },
    }
