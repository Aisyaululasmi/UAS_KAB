import numpy as np
import pandas as pd

from .config import COMPANY_NAMES, SECTORS, TOTAL_CAPITAL_IDR
from .utils import min_max


def _capped_normalize(scores: pd.Series, max_weight: float = 0.25, min_weight: float = 0.05) -> pd.Series:
    weights = scores / scores.sum()
    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()
    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        under = ~over
        if not under.any() or weights[under].sum() <= 0:
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def rank_stocks(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["norm_expected_return"] = min_max(out["expected_return_6m"])
    if "expected_excess_return_6m" not in out.columns:
        out["expected_excess_return_6m"] = out["expected_return_6m"] - out["expected_return_6m"].mean()
    out["norm_expected_excess_return"] = min_max(out["expected_excess_return_6m"])
    out["norm_directional_accuracy"] = min_max(out["directional_accuracy"])
    if "catboost_positive_prob" not in out.columns:
        out["catboost_positive_prob"] = 0.5
    out["norm_catboost_positive_prob"] = min_max(out["catboost_positive_prob"])
    out["norm_liquidity_score"] = min_max(np.log1p(out["avg_volume_60d"]))
    out["norm_volatility"] = min_max(out["volatility_60d"])
    drawdown_abs = out.get("max_drawdown", out["drawdown"]).abs()
    out["norm_max_drawdown"] = min_max(drawdown_abs)
    out["risk_score"] = 0.50 * out["norm_volatility"] + 0.50 * out["norm_max_drawdown"]
    if "price_position_252d" not in out.columns:
        out["price_position_252d"] = 0.5
    out["price_position"] = out["price_position_252d"].clip(0, 1).fillna(0.5)
    out["buy_lowest_score"] = 1 - out["price_position"]
    out["hold_highest_score"] = out["price_position"]
    out["risk_adjusted_score"] = out["expected_return_6m"] / out["volatility_60d"].replace(0, np.nan)
    out["ranking_score"] = (
        0.30 * out["norm_expected_return"]
        + 0.25 * out["norm_expected_excess_return"]
        + 0.20 * out["norm_directional_accuracy"]
        + 0.15 * out["norm_catboost_positive_prob"]
        + 0.15 * out["norm_liquidity_score"]
        - 0.05 * out["norm_volatility"]
    )
    out["signal"] = np.where(
        (
            (out["expected_return_6m"] > 0.05)
            & (out["expected_excess_return_6m"] > 0)
            & (out["drawdown"] > -0.25)
            & (out["catboost_positive_prob"] >= 0.50)
        ),
        "BUY",
        "HOLD",
    )
    buy = out[out["signal"] == "BUY"].sort_values(
        ["expected_return_6m", "ranking_score"], ascending=False
    )
    positive_hold = out[(out["signal"] != "BUY") & (out["expected_return_6m"] > 0)].sort_values(
        ["expected_return_6m", "ranking_score"], ascending=False
    )
    fallback_hold = out[(out["signal"] != "BUY") & (out["expected_return_6m"] <= 0)].sort_values(
        ["expected_return_6m", "ranking_score"], ascending=False
    )
    selected_tickers = pd.concat([buy, positive_hold, fallback_hold]).head(5)["ticker"].tolist()
    out = out.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["selected"] = out["ticker"].isin(selected_tickers)
    return out


def allocate_portfolio(ranking: pd.DataFrame) -> pd.DataFrame:
    selected = ranking[ranking["selected"]].sort_values(
        ["signal", "expected_return_6m", "ranking_score"], ascending=[True, False, False]
    ).copy()
    selected["allocation_score"] = (
        0.30 * selected["norm_expected_return"]
        + 0.25 * selected["norm_expected_excess_return"]
        + 0.25 * selected["buy_lowest_score"]
        + 0.15 * selected["hold_highest_score"]
        + 0.15 * selected["catboost_positive_prob"]
        - 0.20 * selected["risk_score"]
    ).clip(lower=0.01)
    selected["sector"] = selected["ticker"].map(SECTORS).fillna("Other")
    selected["final_weight"] = _capped_normalize(selected["allocation_score"], max_weight=0.25, min_weight=0.05).values
    selected["allocated_amount_idr"] = selected["final_weight"] * TOTAL_CAPITAL_IDR
    selected["expected_profit_idr"] = selected["allocated_amount_idr"] * selected["expected_return_6m"]
    return selected[[
        "rank", "ticker", "signal", "expected_return_6m", "final_weight",
        "allocated_amount_idr", "expected_profit_idr", "ranking_score",
        "expected_excess_return_6m", "catboost_positive_prob", "sector", "price_position", "buy_lowest_score",
        "hold_highest_score", "risk_score", "allocation_score",
    ]]


def allocate_portfolio_aggressive(ranking: pd.DataFrame) -> pd.DataFrame:
    buy = ranking[ranking["signal"] == "BUY"].sort_values(
        ["expected_return_6m", "catboost_positive_prob"], ascending=False
    )
    positive_hold = ranking[(ranking["signal"] != "BUY") & (ranking["expected_return_6m"] > 0)].sort_values(
        ["expected_return_6m", "catboost_positive_prob"], ascending=False
    )
    fallback_hold = ranking[(ranking["signal"] != "BUY") & (ranking["expected_return_6m"] <= 0)].sort_values(
        ["expected_return_6m", "catboost_positive_prob"], ascending=False
    )
    selected = pd.concat([buy, positive_hold, fallback_hold]).head(5).copy()
    selected["allocation_score"] = (
        0.45 * selected["norm_expected_return"]
        + 0.25 * selected["norm_expected_excess_return"]
        + 0.20 * selected["catboost_positive_prob"]
        + 0.10 * selected["buy_lowest_score"]
        + 0.05 * selected["hold_highest_score"]
        - 0.10 * selected["risk_score"]
    ).clip(lower=0.01)
    selected["sector"] = selected["ticker"].map(SECTORS).fillna("Other")
    selected["final_weight"] = _capped_normalize(selected["allocation_score"], max_weight=0.30, min_weight=0.05).values
    selected["allocated_amount_idr"] = selected["final_weight"] * TOTAL_CAPITAL_IDR
    selected["expected_profit_idr"] = selected["allocated_amount_idr"] * selected["expected_return_6m"]
    return selected[[
        "rank", "ticker", "signal", "expected_return_6m", "final_weight",
        "allocated_amount_idr", "expected_profit_idr", "ranking_score",
        "expected_excess_return_6m", "catboost_positive_prob", "sector", "price_position", "buy_lowest_score",
        "hold_highest_score", "risk_score", "allocation_score",
    ]]


def final_recommendation(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in portfolio.sort_values("final_weight", ascending=False).iterrows():
        rows.append({
            "rank": int(row["rank"]),
            "ticker": row["ticker"],
            "company_name": COMPANY_NAMES.get(row["ticker"], row["ticker"]),
            "signal": row["signal"],
            "expected_return_6m": row["expected_return_6m"],
            "expected_excess_return_6m": row.get("expected_excess_return_6m", np.nan),
            "final_weight": row["final_weight"],
            "sector": row.get("sector", ""),
            "catboost_positive_prob": row.get("catboost_positive_prob", np.nan),
            "allocation_score": row.get("allocation_score", np.nan),
            "reason": f"Allocation score {row.get('allocation_score', np.nan):.3f}, sinyal {row['signal']}, return 6 bulan {row['expected_return_6m']:.2%}, excess return vs benchmark internal {row.get('expected_excess_return_6m', np.nan):.2%}, probabilitas positif CatBoost {row.get('catboost_positive_prob', np.nan):.2%}.",
        })
    return pd.DataFrame(rows).sort_values("rank")
