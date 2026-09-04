"""
Prepares bundled example CSVs so the app has something real to show at
first load, before a user uploads their own data.

Usage:
    python scripts/prepare_sample_datasets.py
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def save(loader, filename: str, label_name: str):
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[label_name] = [bunch.target_names[i] for i in bunch.target]
    df.to_csv(DATA / filename, index=False)
    print(f"Saved {filename}: {df.shape[0]} rows x {df.shape[1]} columns "
          f"({df.shape[1]-1} numeric features + '{label_name}')")


if __name__ == "__main__":
    save(load_wine, "wine.csv", "cultivar")
    save(load_breast_cancer, "breast_cancer.csv", "diagnosis")
