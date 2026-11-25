import numpy as np
import pandas as pd


def qini_curve(y, treatment, score):
    """Cumulative incremental responders when customers are targeted in descending score order.

    Returns (x, gain) with x the population fraction targeted, both starting at 0.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    # Scale control responders by the treated/control ratio seen so far; undefined until a control is targeted.
    ratio = np.divide(cum_t, cum_c, out=np.zeros_like(cum_t), where=cum_c > 0)
    gain = cum_yt - cum_yc * ratio

    n = len(y)
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    return x, np.concatenate([[0.0], gain])


def qini_coefficient(y, treatment, score):
    """Area between the Qini curve and random targeting, in responders x population fraction."""
    x, gain = qini_curve(y, treatment, score)
    random_gain = x * gain[-1]
    return float(np.trapezoid(gain - random_gain, x))


def uplift_by_decile(y, treatment, score, n_bins: int = 10, min_group: int = 30) -> pd.DataFrame:
    """Treated and control response rates per score bin, highest-score bin first.

    Bins with fewer than `min_group` treated or control rows are flagged `reliable=False`.
    """
    df = pd.DataFrame({"y": np.asarray(y), "t": np.asarray(treatment), "score": np.asarray(score)})
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(df)
    df["bin"] = (np.arange(n) * n_bins) // n + 1

    rows = []
    for b, g in df.groupby("bin", sort=True):
        treated, control = g[g.t == 1], g[g.t == 0]
        rate_t = treated.y.mean() if len(treated) else np.nan
        rate_c = control.y.mean() if len(control) else np.nan
        rows.append(
            {
                "bin": b,
                "n_treated": len(treated),
                "n_control": len(control),
                "rate_treated": rate_t,
                "rate_control": rate_c,
                "uplift": rate_t - rate_c,
                "reliable": min(len(treated), len(control)) >= min_group,
            }
        )
    return pd.DataFrame(rows)
