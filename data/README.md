# Data

Raw data is not committed. Only this README is tracked; everything else in `data/` is git-ignored.

## Dataset

- **Kaggle slug:** `arashnic/uplift-modeling` ("Uplift Modeling, Marketing Campaign Data", Criteo AI Lab benchmark mirror)
- **Kaggle URL:** https://www.kaggle.com/datasets/arashnic/uplift-modeling
- **Read date:** 2026-11-25 (Kaggle dataset page metadata, read from the public web page; no download)
- **License, Kaggle metadata:** CC0: Public Domain
- **License, upstream terms:** The Criteo uplift dataset page (https://ailab.criteo.com/criteo-uplift-prediction-dataset/) states **CC BY-NC-SA 4.0** for the data. This conflicts with the CC0 label on the Kaggle mirror. Until this is resolved, derived figures must not be published. `reports/figures/` is therefore empty in this repository.
- **Real data:** yes. The Kaggle description states it is a collection of randomized control trial (incrementality test) data, and that it is sub-sampled and features are anonymized. It is not described as synthetic.
- **Dataset size (per Kaggle):** about 13M rows (the upstream page quotes 25M rows for the full benchmark).

## Columns

| Column | Type | Meaning |
|---|---|---|
| `f0` to `f11` | float | Anonymized dense features (model inputs) |
| `treatment` | binary | 1 = treated (ad exposure allowed), 0 = control (held out) |
| `conversion` | binary | Outcome: conversion in the test window (used for the policy) |
| `visit` | binary | Secondary outcome, not used here |
| `exposure` | binary | Effective exposure, not used here |

Treatment ratio per Kaggle: about 84.6% treated. Assignment is randomized (incrementality test), so the treated/control comparison is unconfounded.

## Access

The data is read in place inside a Kaggle Notebook at `/kaggle/input/uplift-modeling/` (the notebook globs `*.csv` in that folder). It is not downloaded to this machine and not committed.
