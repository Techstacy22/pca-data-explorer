"""
pca_explorer
============

A from-scratch implementation of Principal Component Analysis via
singular value decomposition (`pca.py`), with supporting preprocessing
(`preprocessing.py`) and CSV loading/validation (`io.py`). Only
`numpy.linalg.svd` is used as a low-level numerical primitive -- the
centering, variance accounting, projection, reconstruction, and sign
convention are all implemented directly here, not delegated to
`sklearn.decomposition.PCA` or any other decomposition library.

`scikit-learn` is used elsewhere in this project
(`scripts/validate_against_sklearn.py`) only as an external validation
reference.
"""

__version__ = "0.1.0"
