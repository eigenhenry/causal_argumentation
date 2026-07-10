# Chapter 7 — Scaling the SMD Method to 10 Datasets

**Goal.** Extend the model–data alignment study (RQ1) from the current 4 datasets to **10**,
demonstrating that the Standardized Mean Difference (SMD) baseline generalizes across
domains (survival, medical, credit-risk, fairness) and dataset sizes.

**Engine.** No changes needed to `PotentialOutcomesSMD` / `EntropySplitter` /
`StandardizedMeanDifferenceAnalyzer`. The work is data acquisition, feature typing,
scalability-aware preprocessing, and (Phase 2) the agreement layer.

---

## The 10 datasets

| # | Dataset | Source (programmatic) | n (approx) | Target | Status |
|---|---|---|---|---|---|
| 1 | Titanic | `seaborn.load_dataset('titanic')` | 891 | `survived` | ✅ wired |
| 2 | Adult Census Income | OpenML `adult` (v2) | 48,842 | `income` (>50K) | ✅ wired |
| 3 | Pima Diabetes | local `diabetes.csv` | 768 | `Outcome` | ✅ wired |
| 4 | COMPAS Recidivism | ProPublica CSV (URL) | ~6,000 | `two_year_recid` | ✅ wired |
| 5 | Breast Cancer Wisconsin | `sklearn.load_breast_cancer` | 569 | malignant/benign | ⬜ new |
| 6 | German Credit | OpenML `credit-g` | 1,000 | good/bad | ⬜ new |
| 7 | Bank Marketing | OpenML `bank-marketing` | ~45,000 | `y` (subscribed) | ⬜ new |
| 8 | Heart Disease | OpenML `heart-statlog` (or UCI Cleveland) | 270–303 | disease present | ⬜ new |
| 9 | Student Performance | UCI `student-*.csv` | 395 / 649 | pass (`G3 ≥ 10`) | ⬜ new |
| 10 | Credit Card Default (Taiwan) | OpenML `default-of-credit-card-clients` | 30,000 | default next month | ⬜ new (replaces HELOC) |

> **HELOC dropped.** It is gated behind FICO registration (no open URL), which breaks
> reproducibility. Slot #10 is filled by the Taiwan **Default of Credit Card Clients**
> dataset — openly downloadable, binary target, and keeps a credit-risk case in the mix.

---

## Phase 1 — Data gathering, preprocessing, and running the method

### Step 1.1 — Data gathering (reproducible loaders)

Write one `load_<name>() -> pd.DataFrame` per dataset. Rules:

- Prefer **programmatic sources** (`sklearn`, `fetch_openml`, stable URLs) over manual downloads.
- `fetch_openml` already caches to `~/scikit_learn_data`; **additionally cache a processed
  CSV** under `Chapter 7 /data/<name>.csv` on first run so results are frozen and offline-runnable.
- Each loader returns a clean frame with the **binary target already encoded 0/1**.

New-dataset acquisition notes:

- **Breast Cancer** — `load_breast_cancer(as_frame=True)`; 30 continuous features, target
  `target` (0 = malignant, 1 = benign). No categoricals.
- **German Credit** — `fetch_openml('credit-g', version=1, as_frame=True)`; map `class`
  good→1 / bad→0. 7 numeric + 13 categorical columns.
- **Bank Marketing** — `fetch_openml('bank-marketing', version=1, as_frame=True)`; map
  `y` yes→1 / no→0. Imbalanced (~11% positive).
- **Heart Disease** — start with `heart-statlog` (270 rows, already binary). If the UCI
  Cleveland version is preferred, its target is 0–4 → **binarize `>0`** = disease present.
- **Student Performance** — UCI zip (`student-mat.csv` = math, `student-por.csv` =
  Portuguese; **semicolon-delimited**). Regression target `G3` (0–20) → **binarize
  `G3 ≥ 10`** = pass. Use the Portuguese file (larger, n=649) as primary; math as a variant.
- **Taiwan Default** — `fetch_openml('default-of-credit-card-clients', version=1)` (or UCI
  `.xls`); target `Y` / `default payment next month` already 0/1. 23 features.

### Step 1.2 — Preprocessing to control scalability

The engine cost is dominated by (a) the entropy splitter (sorts unique values per feature)
and (b) **interaction discovery**, which is `O(pairs × bins_a × bins_b)` and produces a
heatmap grid that blows up with many features / high-cardinality categoricals. Preprocessing
must actively contain this:

1. **Drop leakage / non-features up front.**
   - Bank Marketing: drop **`duration`** (known only after the call — leakage).
   - Student: drop **`G1`, `G2`** (prior-period grades, near-deterministic of `G3`).
   - Adult: drop `fnlwgt` (already handled). Taiwan: drop `ID`.
2. **Collapse high-cardinality categoricals** (as the Adult "collapsed" variant already
   does). Targets: German `purpose` (10), Bank `job` (12) / `month` (12), Student
   `Mjob`/`Fjob` (5 each). Map to 3–4 semantic groups. This keeps indicator counts — and
   therefore interaction pairs and subgroup sizes — manageable.
3. **Cap interaction analysis for high-dimensional sets.** Breast Cancer (30 features →
   435 pairs) and any wide set: rank features by **univariate SMD first**, then run
   interactions only on the **top-k** (e.g. k=8). Univariate SMD (the core contribution)
   still uses all features; only the exploratory interaction layer is capped.
4. **Scale `min_subgroup_size` with n** so interaction subgroups stay statistically
   meaningful without vanishing on small data (see hyperparameter table below).
5. **Standardize the feature-typing contract**: every dataset config declares
   `raw` / `to_bin` / `cat` explicitly — no automatic inference — so the analysis is
   auditable and matches the chapter's methodology.

**Per-dataset hyperparameters (starting points):**

| Dataset | n | `max_bins` | `min_subgroup_size` | Interaction top-k |
|---|---|---|---|---|
| Breast Cancer | 569 | 2 | 20 | 8 (cap) |
| German Credit | 1,000 | 2 | 30 | all |
| Bank Marketing | 45,000 | 2 | 300 | 8 (cap) |
| Heart Disease | ~300 | 2 | 20 | all |
| Student Performance | 649 | 2 | 25 | all |
| Taiwan Default | 30,000 | 2 | 250 | 8 (cap) |

### Step 1.3 — Refactor to a dataset registry (enables 4 → 10 cleanly)

Replace per-dataset copy-paste with a config-driven driver:

```python
DATASETS = {
    "titanic": dict(
        loader=load_titanic, target="survived",
        raw=[], to_bin=["fare", "age"], cat=["sex", "pclass"],
        max_bins=3, min_subgroup=30, top_k=None,
    ),
    "breast_cancer": dict(
        loader=load_breast_cancer_df, target="target",
        raw=[], to_bin=ALL_CONTINUOUS, cat=[],
        max_bins=2, min_subgroup=20, top_k=8,
    ),
    # ... one entry per dataset ...
}

for name, cfg in DATASETS.items():
    df = cfg["loader"]()
    analyzer = PotentialOutcomesSMD(
        df=df, target_col=cfg["target"],
        raw_features=cfg["raw"], features_to_bin=cfg["to_bin"],
        categorical_features=cfg["cat"],
        max_bins=cfg["max_bins"], min_subgroup_size=cfg["min_subgroup"],
        output_dir=f"./{name}_plots",
    )
    results[name] = analyzer.run()
```

- Output plots go to a per-dataset subfolder **inside the `Chapter 7 /` folder**
  (`Chapter 7 /<name>_plots/`, e.g. `titanic_plots/`, `breast_cancer_plots/`), matching the
  existing `titanic_plots/` convention. Create the 6 new subdirectories there.
- The 4 existing datasets fold into the same registry (behavior unchanged).

### Step 1.4 — Run and sanity-check

For each dataset confirm: exactly 2 target classes, no all-NaN indicator columns,
class balance printed, base SMD table + plots saved. Spot-check that top SMD features are
domain-plausible (e.g. Breast Cancer → `worst concave points` / `worst perimeter`;
Bank → `poutcome`/`pdays`; Student → `failures`/`higher`).

**Phase 1 deliverable:** all 10 datasets loading reproducibly and producing base-SMD +
interaction outputs, with scalability contained on the three large sets.

---

## Phase 2 — Agreement (model–data alignment)

This is Chapter 7's actual contribution and is **not yet in the notebook** — it must be
added on top of Phase 1.

> **Scope: univariate (main-effect) features only.** The alignment analysis uses the
> **univariate SMD ranking** exclusively. The pairwise-interaction layer from Phase 1 is
> **excluded** from Phase 2 — interaction terms have no clean counterpart in DT/SHAP
> feature-importance scores, so their alignment cannot be quantified on a common index set.
> Interactions remain an exploratory diagnostic (Phase 1 output), not part of the agreement claim.

For each dataset, produce three rankings over the **same main-effect feature index set** and compare:

1. **SMD ranking** `r^SMD` — from univariate `|Δ|` on the original features (Phase 1 already
   yields this); interaction rows are dropped before ranking.
2. **Decision-tree ranking** `r^DT` — pruned `DecisionTreeClassifier`, features ranked by
   Gini importance.
3. **SHAP ranking** `r^SHAP` — small MLP/neural net, features ranked by mean `|SHAP|`.

### Hypothesis under test (H1)

> **H1.** SMD reliably recovers the model's **most important (top 1–3) features**, even
> when whole-ranking agreement is weak. Formally: top-weighted / top-`k` agreement between
> `r^SMD` and the model rankings is **high and significantly above chance for small `k`**,
> while full-list Spearman's ρ may be **low**. The gap between the two is the quantified claim.

Plain Spearman's ρ weights every rank equally, so a noisy tail drags it down even when the
head agrees perfectly — which is exactly why ρ alone cannot confirm or deny H1. The design
therefore reports ρ as the "all-features" baseline **and** a battery of top-weighted metrics.

### Metrics (per dataset, `r^SMD` vs `r^DT` and vs `r^SHAP`)

1. **Full-list Spearman's ρ** — the all-features baseline (expected *lower*; the foil).
2. **Agreement@k curve** — for `k = 1 … p`, the overlap of the two top-`k` sets:
   - `Overlap@k = |top_k(SMD) ∩ top_k(model)| / k` (a.k.a. precision@k / normalized set overlap).
   - Plot agreement as a **function of k**. H1 predicts a curve that starts high (k=1–3)
     and decays toward the chance line as k grows.
3. **Rank-Biased Overlap (RBO)** — a single top-weighted similarity in [0,1] with a tunable
   persistence `p_rbo` (e.g. 0.9) that emphasizes the head of the list. Purpose-built for
   "do the tops agree?"; report alongside ρ so the contrast (`RBO high, ρ low`) is explicit.
4. **Weighted Kendall's τ** — `scipy.stats.weightedtau` (hyperbolic weighting) — a second
   top-weighted single number, robust cross-check on RBO.
5. **Top-1 / Top-3 hit rate** — does SMD's #1 land in the model's top-1 / top-3? The most
   direct, reader-legible statement of H1; aggregate as a fraction across the 10 datasets.

### Chance baselines & significance

- **Chance line for Overlap@k** is `k/p` (random top-`k` set) — draw it on every curve so
  "above chance" is visible.
- **Permutation test:** shuffle one ranking `N=10⁴` times to get a null distribution for
  top-3 overlap / RBO per dataset; report an empirical p-value. This turns "feels high" into
  a significance claim.

### Aggregation across the 10 datasets (the headline result)

- **Mean Agreement@k curve** across datasets with error bands (SMD-vs-SHAP and SMD-vs-DT),
  overlaid on the `k/p` chance line — one figure that visually *is* H1.
- **Summary table:** one row per dataset with `ρ_all`, `RBO`, `top-1 hit`, `top-3 hit`
  (vs DT and vs SHAP). The expected pattern — **low `ρ_all`, high `RBO`/top-3** — is the
  quantified confirmation of H1.

### Design notes

- Rank at the **original-feature granularity** (aggregate binned indicators back to their
  source feature, e.g. max or sum of `|Δ|` per source column) so all three rankings share
  one index set. Note: small-`p` datasets (Titanic ~7, Diabetes 8) make top-3 a large
  fraction of features — report `p` next to each result so `k` is read relative to `p`.
- Fix a random seed; report the model's test accuracy alongside the metrics so a low `ρ_all`
  reads as a diagnostic (model exploiting interactions) rather than a broken model.
- Reuse the collapsed-categorical preprocessing from Phase 1 so DT/SHAP see the same
  feature space as SMD.
- RBO has no standard scipy implementation — use a small vetted helper (~15 lines) or the
  `rbo` package; pin whichever for reproducibility.

**Phase 2 deliverable:** (a) the mean Agreement@k figure with chance line, (b) a 10-row
summary table (`ρ_all`, `RBO`, top-1, top-3 vs DT and SHAP), and (c) per-dataset permutation
p-values for top-3 overlap — together confirming or refuting H1, ready to drop into
`chapter7.tex`.

---

## Risks & open decisions

- **Heart Disease version.** `heart-statlog` (clean binary, n=270) vs UCI Cleveland
  (n=303, needs `>0` binarization). Pick one for the final chapter; statlog is simpler.
- **Student math vs Portuguese.** Use Portuguese (n=649) as primary; note math as a
  robustness variant if space allows.
- **Interaction layer on wide/large data** is exploratory only — if runtime is a problem,
  disable interactions there and keep the univariate SMD (the core signal).
- **Taiwan vs another open credit set.** Taiwan Default is the default substitute for
  HELOC; swappable for "Give Me Some Credit" if preferred.
