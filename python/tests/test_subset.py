from anndata_rs.pyanndata import AnnData, AnnDataSet

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse, random

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_subset(tmp_path):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    adata = AnnData(
        X=X,
        obs=dict(Obs=["A", "B", "C"]),
        var=dict(Feat=["a", "b", "c"]),
        obsm=dict(X_pca=np.array([[1, 2], [3, 4], [5, 6]])),
        filename=h5ad(tmp_path),
    )

    adata.subset([0, 1])

    np.testing.assert_array_equal(adata.obsm["X_pca"][...], np.array([[1, 2], [3, 4]]))

    X = random(5000, 50, 0.1, format="csr")
    adata = AnnData(
        X=X,
        obsm=dict(X_pca=X),
        filename=h5ad(tmp_path),
    )
    idx = np.random.randint(0, 5000, 1000)
    adata.subset(idx)
    np.testing.assert_array_equal(adata.X[...].todense(), X[idx,:].todense())
    np.testing.assert_array_equal(adata.obsm["X_pca"][...].todense(), X[idx,:].todense())

    X = random(5000, 50, 0.1, format="csr")
    adata = AnnData(
        X=X,
        obsm=dict(X_pca=X),
        filename=h5ad(tmp_path),
    )
    idx = np.random.choice([True, False], size=5000, p=[0.5, 0.5])
    adata.subset(idx)
    np.testing.assert_array_equal(adata.X[...].todense(), X[idx,:].todense())
    np.testing.assert_array_equal(adata.obsm["X_pca"][...].todense(), X[idx,:].todense())

def test_chunk(tmp_path):
    X = random(5000, 50, 0.1, format="csr", dtype=np.int64)
    adata = AnnData(
        X=X,
        filename=h5ad(tmp_path),
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

def test_anndataset(tmp_path):
    data1 = np.random.randint(10000, size=(15, 500))
    data2 = np.random.randint(10000, size=(99, 500))
    data3 = np.random.randint(10000, size=(87, 500))
    merged = np.concatenate([data1, data2, data3], axis=0)

    adata1 = AnnData(X=data1, filename=h5ad(tmp_path))
    adata2 = AnnData(X=data2, filename=h5ad(tmp_path))
    adata3 = AnnData(X=data3, filename=h5ad(tmp_path))

    dataset = AnnDataSet([("1", adata1), ("2", adata2), ("3", adata3)], h5ad(tmp_path))

    idx = np.random.randint(0, 100, 50)
    np.testing.assert_array_equal(merged[idx, :], dataset.X.get_rows(idx))
 