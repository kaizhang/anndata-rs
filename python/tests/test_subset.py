from itertools import product

import numpy as np
from numpy import ma
import pandas as pd
import pytest
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse, random

from anndata_rs import AnnData

def h5ad():
    return str(uuid.uuid4()) + ".h5ad"

def test_subset():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    adata = AnnData(
        X=X,
        obs=dict(Obs=["A", "B", "C"]),
        var=dict(Feat=["a", "b", "c"]),
        obsm=dict(X_pca=np.array([[1, 2], [3, 4], [5, 6]])),
        filename=h5ad(),
    )

    adata.subset([0, 1])

    np.testing.assert_array_equal(adata.obsm["X_pca"][...], np.array([[1, 2], [3, 4]]))

    X = random(5000, 50, 0.1, format="csr")
    adata = AnnData(
        X=X,
        obsm=dict(X_pca=X),
        filename=h5ad(),
    )
    idx = np.random.randint(0, 5000, 1000)
    adata.subset(idx)
    np.testing.assert_array_equal(adata.X[...].todense(), X[idx,:].todense())
    np.testing.assert_array_equal(adata.obsm["X_pca"][...].todense(), X[idx,:].todense())

    X = random(5000, 50, 0.1, format="csr")
    adata = AnnData(
        X=X,
        obsm=dict(X_pca=X),
        filename=h5ad(),
    )
    idx = np.random.choice([True, False], size=5000, p=[0.5, 0.5])
    adata.subset(idx)
    np.testing.assert_array_equal(adata.X[...].todense(), X[idx,:].todense())
    np.testing.assert_array_equal(adata.obsm["X_pca"][...].todense(), X[idx,:].todense())

def test_chunk():
    X = random(5000, 50, 0.1, format="csr", dtype=np.int64)
    adata = AnnData(
        X=X,
        filename=h5ad(),
    )
    s = X.sum(axis = 0)
    s_ = np.zeros_like(s)
    for m in adata.X.chunked(47):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)

    s_ = np.zeros_like(s)
    for m in adata.X.chunked(500000):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)