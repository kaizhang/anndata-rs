from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from anndata_rs import AnnData, AnnDataSet

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import uuid
from scipy.sparse import csr_matrix

BACKENDS = ["hdf5"]

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

@pytest.mark.parametrize("backend", BACKENDS)
@given(
    x = arrays(integer_dtypes(endianness='='), (47, 79)),
    ridx = st.lists(st.integers(min_value=0, max_value=46), min_size=0, max_size=50),
    cidx = st.lists(st.integers(min_value=0, max_value=78), min_size=0, max_size=100),
    mask = st.lists(st.booleans(), min_size=79, max_size=79),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_index(x, ridx, cidx, mask, tmp_path, backend):
    x_ = csr_matrix(x)
    adata = AnnData(
        X=x,
        obsm=dict(x=x_),
        filename = h5ad(tmp_path),
        backend=backend,
    )

    np.testing.assert_array_equal(adata.X[:10, 4:50], x[:10, 4:50])
    np.testing.assert_array_equal(adata.X[ridx, cidx], x[np.ix_(ridx, cidx)])
    np.testing.assert_array_equal(adata.X[ridx, :], x[ridx, :])
    np.testing.assert_array_equal(adata.X[pl.Series(ridx), :], x[ridx, :])
    np.testing.assert_array_equal(adata.X[:, cidx], x[:, cidx])
    np.testing.assert_array_equal(adata.X[:, pl.Series(cidx)], x[:, cidx])
    np.testing.assert_array_equal(adata.X[:, mask], x[:, mask])
    np.testing.assert_array_equal(adata.X[:, np.array(mask)], x[:, np.array(mask)])
    np.testing.assert_array_equal(adata.X[:, pl.Series(mask)], x[:, np.array(mask)])

    np.testing.assert_array_equal(
        adata.obsm.el('x')[ridx, cidx].todense(),
        x[np.ix_(ridx, cidx)],
    )
    np.testing.assert_array_equal(adata.obsm.el('x')[ridx, :].todense(), x[ridx, :])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, cidx].todense(), x[:, cidx])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, mask].todense(), x[:, mask])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, np.array(mask)].todense(), x[:, np.array(mask)])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, pl.Series(mask)].todense(), x[:, np.array(mask)])

@pytest.mark.parametrize("backend", BACKENDS)
@given(
    x1 = arrays(np.int64, (15, 179)),
    x2 = arrays(np.int64, (47, 179)),
    x3 = arrays(np.int64, (77, 179)),
    row_idx = st.lists(st.integers(min_value=0, max_value=15+47+76), min_size=1, max_size=500),
    col_idx = st.lists(st.integers(min_value=0, max_value=178), min_size=1, max_size=200),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_index_anndataset(x1, x2, x3, row_idx, col_idx, tmp_path, backend):
    merged = np.concatenate([x1, x2, x3], axis=0)

    # dense array
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path), backend=backend)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    np.testing.assert_array_equal(merged, dataset.X[:])
    np.testing.assert_array_equal(merged[row_idx, :], dataset.X[row_idx, :])
    np.testing.assert_array_equal(merged[:, col_idx], dataset.X[:, col_idx])
    np.testing.assert_array_equal(merged[np.ix_(row_idx, col_idx)], dataset.X[row_idx, col_idx])

    # sparse array
    adata1 = AnnData(X=csr_matrix(x1), filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(X=csr_matrix(x2), filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(X=csr_matrix(x3), filename=h5ad(tmp_path), backend=backend)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    np.testing.assert_array_equal(merged, dataset.X[:].todense())
    np.testing.assert_array_equal(merged[:, col_idx], dataset.X[:, col_idx].todense())
    np.testing.assert_array_equal(merged[:, col_idx], dataset.X[:, pl.Series(col_idx)].todense())
    np.testing.assert_array_equal(merged[np.ix_(row_idx, col_idx)], dataset.X[row_idx, col_idx].todense())