"""
Flask web interface for the PCA explorer.

Upload a CSV (or pick a bundled sample dataset), optionally designate a
label column for coloring, optionally standardize features, and get back
the first two principal components, the explained-variance (scree) curve,
the reconstruction-error-vs-k curve, and each PC's top contributing
features -- all computed by the from-scratch implementation in
`src/pca_explorer/`.

Local usage:
    python app/server.py
    (then open http://127.0.0.1:5000)

Production usage (see render.yaml):
    gunicorn app.server:app --chdir . --bind 0.0.0.0:$PORT
"""
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pca_explorer.io import load_csv, validate_features, DataValidationError
from pca_explorer.preprocessing import standardize
from pca_explorer.pca import fit_pca, reconstruction_error_curve

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SAMPLE_DATASETS = {
    "wine": {"file": "wine.csv", "label_column": "cultivar",
             "description": "178 wines, 13 chemical measurements, 3 cultivars"},
    "breast_cancer": {"file": "breast_cancer.csv", "label_column": "diagnosis",
                       "description": "569 tumor samples, 30 measurements, benign/malignant"},
}

app = Flask(__name__, template_folder="templates")


def run_pca_pipeline(df, labels, standardize_features: bool, top_n_loadings: int = 5):
    features_df = validate_features(df)
    feature_names = list(features_df.columns)
    X = features_df.to_numpy(dtype=float)

    if standardize_features:
        X_input = standardize(X).standardized
    else:
        X_input = X

    result = fit_pca(X_input)
    n_components_available = result.components_.shape[0]
    k_show = min(2, n_components_available)

    errors = reconstruction_error_curve(result, X_input)

    def top_loadings(component_idx):
        loadings = result.components_[component_idx]
        order = sorted(range(len(loadings)), key=lambda i: -abs(loadings[i]))[:top_n_loadings]
        return [{"feature": feature_names[i], "loading": round(float(loadings[i]), 4)} for i in order]

    return {
        "n_samples": result.n_samples_,
        "n_features": result.n_features_,
        "feature_names": feature_names,
        "labels": list(labels) if labels is not None else None,
        "pc1": result.scores_[:, 0].round(4).tolist(),
        "pc2": (result.scores_[:, 1].round(4).tolist() if k_show > 1 else [0.0] * result.n_samples_),
        "explained_variance_ratio": result.explained_variance_ratio_.round(6).tolist(),
        "cumulative_variance_ratio": result.explained_variance_ratio_.cumsum().round(6).tolist(),
        "reconstruction_error": errors.round(6).tolist(),
        "top_loadings_pc1": top_loadings(0),
        "top_loadings_pc2": (top_loadings(1) if k_show > 1 else []),
    }


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_DATASETS)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    standardize_features = request.form.get("standardize", "true") == "true"
    label_column = request.form.get("label_column", "").strip() or None

    try:
        if "file" in request.files and request.files["file"].filename:
            df, labels = load_csv(request.files["file"], label_column=label_column)
        else:
            sample_key = request.form.get("sample", "wine")
            if sample_key not in SAMPLE_DATASETS:
                return jsonify({"error": f"Unknown sample dataset '{sample_key}'"}), 400
            spec = SAMPLE_DATASETS[sample_key]
            df, labels = load_csv(DATA_DIR / spec["file"], label_column=spec["label_column"])

        payload = run_pca_pipeline(df, labels, standardize_features)
        return jsonify(payload)

    except DataValidationError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
