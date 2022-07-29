from anndata_rs import AnnData, AnnDataSet, read

import math
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pathlib import Path
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse, random
from hypothesis import given, example, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

@given(x=arrays(
    integer_dtypes(endianness='=') | floating_dtypes(endianness='=', sizes=(32, 64)) |
    unsigned_integer_dtypes(endianness = '='),
    array_shapes(min_dims=2, max_dims=2, min_side=0, max_side=5),
))
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_assign_arrays(x, tmp_path):
    adata = AnnData(filename = h5ad(tmp_path))
    adata.uns['x'] = x
    x_ = adata.uns['x']
    np.testing.assert_array_equal(x_, x)

@given(x=st.floats())
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_assign_floats(x, tmp_path):
    adata = AnnData(filename = h5ad(tmp_path))
    adata.uns['x'] = x
    x_ = adata.uns['x']
    assert (x_ == x or (math.isnan(x) and math.isnan(x_)))

def test_creation(tmp_path):
    adata = AnnData(filename=h5ad(tmp_path))
    assert adata.n_obs == 0
    adata.obsm =dict(X_pca=np.array([[1, 2], [3, 4]]))
    assert adata.n_obs == 2
    adata.uns['df'] = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    adata.uns['df'] = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])

def test_type(tmp_path):
    adata = AnnData(filename = h5ad(tmp_path), X = np.array([[1, 2], [3, 4]]))

    x = "test"
    adata.uns["str"] = x
    assert adata.uns["str"] == x

    x = {
        "a": 1,
        "b": 2.0,
        "c": {"1": 2, "2": 5},
        "d": "test",
    }
    adata.uns["dict"] = x
    assert adata.uns["dict"] == x

@given(
    x1 = arrays(np.int64, (15, 179)),
    x2 = arrays(np.int64, (47, 179)),
    x3 = arrays(np.int64, (77, 179)),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_create_anndataset(x1, x2, x3, tmp_path):
    # empty dataset
    adata1 = AnnData(filename=h5ad(tmp_path))
    adata2 = AnnData( filename=h5ad(tmp_path))
    adata3 = AnnData(filename=h5ad(tmp_path))
    dataset = AnnDataSet(
        [("1", adata1), ("2", adata2), ("3", adata3)], h5ad(tmp_path), "batch"
    )

    # dense array
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path))
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path))
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path))
    merged = np.concatenate([x1, x2, x3], axis=0)
    dataset = AnnDataSet(
        [("1", adata1), ("2", adata2), ("3", adata3)], h5ad(tmp_path), "batch"
    )
    np.testing.assert_array_equal(merged, dataset.X[:])

    # sparse array
    adata1 = AnnData(X=csr_matrix(x1), filename=h5ad(tmp_path))
    adata2 = AnnData(X=csr_matrix(x2), filename=h5ad(tmp_path))
    adata3 = AnnData(X=csr_matrix(x3), filename=h5ad(tmp_path))
    dataset = AnnDataSet(
        [("1", adata1), ("2", adata2), ("3", adata3)], h5ad(tmp_path), "batch"
    )
    np.testing.assert_array_equal(merged, dataset.X[:].todense())