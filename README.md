# uplift-targeting-policy (UP03)

**Question:** Which fraction of customers should be treated to maximize net gain, once the cost of treating each customer is counted?

A T-learner estimates `τ(x) = μ₁(x) − μ₀(x)` for each customer. A policy treats a customer when `τ(x) − c > 0`, where `c` is the stated cost per treated customer, expressed in conversion units. The Qini curve and uplift by decile show how well the ranking works, and the treated share is chosen against expected net gain.

## Decision supported

How many customers to treat, and from which score cutoff, when each treatment has a cost `c`.

## Data

- **Dataset:** Criteo uplift benchmark, Kaggle slug [`arashnic/uplift-modeling`](https://www.kaggle.com/datasets/arashnic/uplift-modeling). Real randomized incrementality-test data with 12 anonymized features, a binary `treatment`, and a binary `conversion` outcome.
- **License:** Kaggle metadata says CC0. The upstream Criteo terms say CC BY-NC-SA 4.0. This conflict is unresolved, so no figures are committed. See [data/README.md](data/README.md).
- **Read path:** `/kaggle/input/uplift-modeling/` inside a Kaggle Notebook. Nothing is downloaded or committed.
- **Split:** no dates, so a random train/validation/test split (50/20/30 of rows) stratified by `treatment`. Seed 42.
- **Cutoff:** the T-learner is fit on train, the cutoff and treated share are chosen on validation, and the chosen policy is reported once on test.

## Method

| Element | Choice |
|---|---|
| Scores | `τ(x)` from `TLearner` (`HistGradientBoostingClassifier(random_state=42)` per arm) |
| Policy | Treat if `τ(x) ≥ cutoff`; the cutoff maximises `n·(uplift − c)` on validation |
| Net gain | For the treated set, `n · (treated rate − control rate − c)` |
| Budget cut | `top_fraction_mask` treats exactly the top share of scores |

## Metrics

- **Qini coefficient** on the test split, with random targeting as the baseline.
- **Uplift by decile** with treated and control counts and rates per bin. Bins below 30 treated or control rows are flagged as unreliable.
- The notebook writes a Qini-curve figure, an uplift-by-decile chart, and a treated-share versus expected-gain figure to `reports/figures/`.

## Results

**Not yet run on the real dataset.** The notebook has not been executed locally. Results and the chosen cutoff will be filled in after a Kaggle run.

## Tests

The tests use tiny hand-built fixtures. They check the budget cut, the per-k net gain hand calculation, cutoff choice (including the case where no treatment is profitable), and the Qini curve and coefficient.

## Limitation

The net gain depends on the cost `c`, which is assumed, not estimated from the data. The conversion rate is also a proxy for value, so a different cost or value per conversion would move the chosen cutoff.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then open `notebooks/03_targeting_policy.ipynb` in a Kaggle Notebook with `arashnic/uplift-modeling` attached and choose **Run All**.

## Layout

```text
src/uplift_policy/      data split, T-learner scores, Qini and decile metrics, targeting policy
notebooks/              03_targeting_policy.ipynb (Run All on Kaggle)
tests/                  budget cut, net gain, cutoff choice, Qini hand calculation
reports/figures/        exported figures (empty until a licensed run is published)
data/                   git-ignored except data/README.md
```
