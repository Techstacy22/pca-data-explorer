"""
Validation report: our from-scratch PCA vs. `sklearn.decomposition.PCA`.

**Reference implementation:** scikit-learn 1.9.0
**Independent implementation:** this project's `src/pca_explorer/`

scikit-learn is used here ONLY as an external comparison target -- never
inside `src/pca_explorer/` to perform the decomposition itself.

Usage:
    python scripts/validate_against_sklearn.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.decomposition import PCA as SklearnPCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca_explorer.pca import fit_pca, reconstruction_error_curve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def compare_on_dataset(name: str, X: np.ndarray) -> dict:
    sk = SklearnPCA(svd_solver="full")
    sk.fit(X)

    ours = fit_pca(X)

    n_components = min(X.shape)
    max_component_abs_diff = np.max(np.abs(ours.components_ - sk.components_))
    max_variance_abs_diff = np.max(np.abs(ours.explained_variance_ - sk.explained_variance_))
    max_ratio_abs_diff = np.max(np.abs(ours.explained_variance_ratio_ - sk.explained_variance_ratio_))

    sk_scores = sk.transform(X)
    max_score_abs_diff = np.max(np.abs(ours.scores_ - sk_scores))

    return {
        "dataset": name,
        "shape": X.shape,
        "max_component_abs_diff": max_component_abs_diff,
        "max_variance_abs_diff": max_variance_abs_diff,
        "max_ratio_abs_diff": max_ratio_abs_diff,
        "max_score_abs_diff": max_score_abs_diff,
    }


def main():
    datasets = {
        "iris (4 features)": load_iris().data,
        "wine (13 features)": load_wine().data,
        "breast_cancer (30 features)": load_breast_cancer().data,
    }

    rng = np.random.default_rng(42)
    datasets["random gaussian (200x20)"] = rng.normal(size=(200, 20)) * rng.uniform(0.1, 50, size=20)
    datasets["random gaussian (30x100, p>n)"] = rng.normal(size=(30, 100))

    lines = ["VALIDATION REPORT", "==================", "",
             "Reference implementation: scikit-learn 1.9.0 (sklearn.decomposition.PCA, svd_solver='full')",
             "Independent implementation: this project's src/pca_explorer/", ""]

    all_results = []
    for name, X in datasets.items():
        r = compare_on_dataset(name, X)
        all_results.append(r)
        lines.append(f"Dataset: {name}  shape={r['shape']}")
        lines.append(f"  components max abs diff:         {r['max_component_abs_diff']:.2e}")
        lines.append(f"  explained_variance max abs diff: {r['max_variance_abs_diff']:.2e}")
        lines.append(f"  explained_variance_ratio max abs diff: {r['max_ratio_abs_diff']:.2e}")
        lines.append(f"  transformed scores max abs diff: {r['max_score_abs_diff']:.2e}")
        lines.append("")

    # Reconstruction error sanity check against sklearn's own inverse_transform, on wine.
    X = load_wine().data
    ours = fit_pca(X)
    our_curve = reconstruction_error_curve(ours, X)

    sk = SklearnPCA(svd_solver="full")
    sk.fit(X)
    n = X.shape[0]
    sk_curve = []
    for k in range(1, X.shape[1] + 1):
        Xk = sk.transform(X)[:, :k]
        Vk = sk.components_[:k]
        X_hat = Xk @ Vk + sk.mean_
        sk_curve.append(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
    sk_curve = np.array(sk_curve)
    max_recon_diff = np.max(np.abs(our_curve - sk_curve))
    lines.append(f"Reconstruction error curve (wine, all k) max abs diff vs. sklearn: {max_recon_diff:.2e}")

    summary = "\n".join(lines)
    print(summary)
    with open(RESULTS / "validation_summary.txt", "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
