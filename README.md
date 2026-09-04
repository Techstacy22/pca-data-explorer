# PCA Data Explorer

A from-scratch implementation of Principal Component Analysis via
singular value decomposition — centering, decomposition, explained
variance, projection, reconstruction error, and a deterministic sign
convention are all implemented directly (`src/pca_explorer/`), validated
against `scikit-learn`, and wrapped in an interactive CSV-upload web app.

## Results

Validated against `sklearn.decomposition.PCA` (`svd_solver="full"`) on
five datasets:

| Dataset | Shape | Components max abs diff | Scores max abs diff |
|---|---|---:|---:|
| Iris | 150×4 | 4.4e-16 | 2.9e-15 |
| Wine | 178×13 | 2.6e-14 | 5.4e-13 |
| Breast cancer | 569×30 | 4.2e-14 | 3.9e-11 |
| Random gaussian | 200×20 | 2.4e-14 | 3.5e-12 |
| Random gaussian (p > n) | 30×100 | see below | 1.6e-13 |

All agree to float64 noise level — this is expected, not luck: the
decomposition (SVD), variance accounting, and **sign convention**
(`sklearn.utils.extmath.svd_flip`, `u_based_decision=False`) were
deliberately reimplemented to match scikit-learn's exactly, so the two
are directly, numerically comparable. Reconstruction error curves agree
to 3.6e-15.

**A genuine finding, not hidden:** on the 30-sample × 100-feature case,
one component (the 30th, out of 30) disagreed by ~0.33 — a real,
different direction, not noise. Diagnosis: centering 30 samples caps the
data's rank at 29, so that component's explained variance is ~1e-29 in
*both* implementations — numerically zero. When a component carries zero
variance, its direction is mathematically undefined (any vector spanning
the leftover null space is equally valid), so two independent, entirely
correct SVD implementations can land on different — but equally
correct — answers there. Full detail in `src/pca_explorer/pca.py`'s
docstring and `results/validation_summary.txt`.

## What's implemented from scratch vs. used only for validation

| Component | Status |
|---|---|
| Centering / standardization | Exact reimplementation |
| PCA via SVD (components, explained variance, scores) | Exact reimplementation |
| Sign convention | Exact reimplementation of scikit-learn's `svd_flip` |
| Reconstruction / reconstruction error | Exact reimplementation, cross-checked against a closed-form identity |

`scikit-learn` appears only in `scripts/validate_against_sklearn.py` (as
an external comparison target) and `scripts/prepare_sample_datasets.py`
(to fetch the bundled example CSVs) — never inside `src/pca_explorer/` to
perform the decomposition itself.

## Project structure

```text
pca-data-explorer/
├── README.md / requirements.txt / LICENSE / pytest.ini / render.yaml
├── src/pca_explorer/
│   ├── preprocessing.py   <- centering, standardization
│   ├── pca.py             <- PCA via SVD, sign convention, reconstruction
│   └── io.py               <- CSV loading + validation
├── tests/                  <- 22 tests (hand-calculated + closed-form checks)
├── data/
│   ├── wine.csv             <- 178 wines, 13 features, 3 cultivars
│   └── breast_cancer.csv    <- 569 tumors, 30 features, benign/malignant
├── scripts/
│   ├── prepare_sample_datasets.py
│   └── validate_against_sklearn.py
├── app/
│   ├── server.py            <- Flask app
│   └── templates/index.html <- upload form + scatter/scree/reconstruction charts
└── results/validation_summary.txt
```

## Reproducing everything

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

pytest tests/ -v                              # 1. run the 22 tests
python scripts/prepare_sample_datasets.py     # 2. generate bundled sample CSVs
python scripts/validate_against_sklearn.py    # 3. validate vs. scikit-learn

python app/server.py                          # 4. web UI at http://127.0.0.1:5000
```

## The interface

Pick a bundled dataset (Wine, Breast Cancer) or upload your own CSV
(optionally naming a label column for coloring), toggle standardization,
and get back: a PC1-vs-PC2 scatter plot, a scree plot with cumulative
explained variance, a reconstruction-error-vs-components-kept curve, and
each component's top contributing features.

## Tests

```
pytest tests/ -v
```

22 tests: preprocessing (hand-calculated centering/standardization),
PCA against a hand-calculated rank-one example (worked out by hand: a
covariance matrix `[[4,4],[4,4]]` with eigenvalues 8 and 0), orthonormal
components, explained-variance-ratio summing to 1, PC scores' variance
matching `explained_variance_` exactly, exact full reconstruction, and
reconstruction error checked against its closed-form identity (residual
variance after keeping k components must equal exactly the variance of
the discarded components).

## Author

Ecstacy Williams. Independent portfolio project (mini project #2 of 2).

## Dataset citations

> Aeberhard, S., Coomans, D., de Vel, O. (1992). "Comparison of
> Classifiers in High Dimensional Settings." (Wine dataset, via
> `sklearn.datasets.load_wine`.)

> Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993). "Nuclear feature
> extraction for breast tumor diagnosis." (Wisconsin Diagnostic Breast
> Cancer dataset, via `sklearn.datasets.load_breast_cancer`.)
