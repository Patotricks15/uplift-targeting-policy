import numpy as np
import pandas as pd


def top_fraction_mask(score, fraction: float) -> np.ndarray:
    """Budget cut: treat exactly round(fraction * n) customers with the highest scores."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    score = np.asarray(score, dtype=float)
    k = int(round(fraction * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    if k > 0:
        mask[np.argsort(-score, kind="mergesort")[:k]] = True
    return mask


def cumulative_net_gain(y, treatment, score, cost: float) -> pd.DataFrame:
    """Estimated net gain of treating the top k customers by score, for every k.

    Net gain = k * (uplift among the top k) - k * cost, with uplift estimated as the treated
    rate minus the control rate in that group. A k with an empty arm is undefined and set to -inf.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, t, s = y[order], t[order], score[order]

    n = np.arange(1, len(y) + 1)
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = yt / n_t - yc / n_c
    net = np.where((n_t > 0) & (n_c > 0), n * uplift - n * cost, -np.inf)

    return pd.DataFrame(
        {
            "k": n,
            "fraction": n / len(y),
            "cutoff_score": s,
            "net_gain": net,
            "net_gain_per_customer": net / len(y),
        }
    )


def choose_cutoff(y, treatment, score, cost: float) -> dict:
    """Pick the treated share that maximises net gain on the data given (use validation only).

    Returns `fraction=0` and `cutoff_score=inf` (treat nobody) when no positive net gain exists.
    """
    table = cumulative_net_gain(y, treatment, score, cost)
    best = table.loc[table["net_gain"].idxmax()]
    if not np.isfinite(best["net_gain"]) or best["net_gain"] <= 0:
        return {"fraction": 0.0, "cutoff_score": np.inf, "net_gain": 0.0}
    return {
        "fraction": float(best["fraction"]),
        "cutoff_score": float(best["cutoff_score"]),
        "net_gain": float(best["net_gain"]),
    }


def apply_cutoff(score, cutoff_score: float) -> np.ndarray:
    """Treat customers whose score is at or above the cutoff (inf treats nobody)."""
    return np.asarray(score, dtype=float) >= cutoff_score


def evaluate_policy(y, treatment, mask, cost: float) -> dict:
    """Net gain of a fixed targeting mask, estimated on the split it is applied to."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"n_selected": 0, "uplift_selected": np.nan, "net_gain": 0.0, "net_gain_per_customer": 0.0}
    ys, ts = y[mask], t[mask]
    if ts.sum() == 0 or (1 - ts).sum() == 0:
        raise ValueError("Selected customers must include both treated and control rows.")
    uplift = ys[ts == 1].mean() - ys[ts == 0].mean()
    net = n_sel * (uplift - cost)
    return {
        "n_selected": n_sel,
        "uplift_selected": float(uplift),
        "net_gain": float(net),
        "net_gain_per_customer": float(net / len(y)),
    }
