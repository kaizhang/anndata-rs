from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
import pytest
from anndata_rs import AnnData, AnnDataSet, read_dataset
import os

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse, random, vstack

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

@given(
    x = arrays(integer_dtypes(endianness='='), (47, 79)),
    indices = st.lists(st.integers(min_value=0, max_value=46), min_size=0, max_size=50),
    indices2 = st.lists(st.integers(min_value=0, max_value=78), min_size=0, max_size=100),
    obs = st.lists(st.integers(min_value=0, max_value=100000), min_size=47, max_size=47),
    obsm = arrays(integer_dtypes(endianness='='), (47, 139)),
    obsp = arrays(integer_dtypes(endianness='='), (47, 47)),
    varm = arrays(integer_dtypes(endianness='='), (79, 39)),
    varp = arrays(integer_dtypes(endianness='='), (79, 79)),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_subset(x, obs, obsm, obsp, varm, varp, indices, indices2, tmp_path):
    ident = list(map(lambda x: str(x), range(len(obs))))
    adata = AnnData(
        X=x,
        obs = dict(ident=ident, txt=obs),
        #obs = dict(txt=obs),
        obsm = dict(x=obsm, y=csr_matrix(obsm)),
        obsp = dict(x=obsp, y=csr_matrix(obsp)),
        varm = dict(x=varm, y=csr_matrix(varm)),
        varp = dict(x=varp, y=csr_matrix(varp)),
        filename = h5ad(tmp_path),
    )

    adata_subset = adata.subset(indices, indices2, out = h5ad(tmp_path))
    np.testing.assert_array_equal(adata_subset.X[:], x[np.ix_(indices, indices2)])
    np.testing.assert_array_equal(adata_subset.obs["txt"], np.array(list(obs[i] for i in indices)))
    np.testing.assert_array_equal(adata_subset.obsm["x"], obsm[indices, :])
    np.testing.assert_array_equal(adata_subset.obsm["y"].todense(), obsm[indices, :])
    np.testing.assert_array_equal(adata_subset.obsp["x"], obsp[np.ix_(indices, indices)])
    np.testing.assert_array_equal(adata_subset.obsp["y"].todense(), obsp[np.ix_(indices, indices)])
    np.testing.assert_array_equal(adata_subset.varm["x"], varm[indices2, :])
    np.testing.assert_array_equal(adata_subset.varm["y"].todense(), varm[indices2, :])
    np.testing.assert_array_equal(adata_subset.varp["x"], varp[np.ix_(indices2, indices2)])
    np.testing.assert_array_equal(adata_subset.varp["y"].todense(), varp[np.ix_(indices2, indices2)])

    adata_subset = adata.subset([str(x) for x in indices], out = h5ad(tmp_path))
    np.testing.assert_array_equal(adata_subset.X[:], x[indices, :])
    np.testing.assert_array_equal(adata_subset.obs["txt"], np.array(list(obs[i] for i in indices)))
    np.testing.assert_array_equal(adata_subset.obsm["x"], obsm[indices, :])
    np.testing.assert_array_equal(adata_subset.obsm["y"].todense(), obsm[indices, :])

    adata_subset = adata.subset(pl.Series([str(x) for x in indices]), out = h5ad(tmp_path))
    np.testing.assert_array_equal(adata_subset.X[:], x[indices, :])

    adata.subset(indices)
    np.testing.assert_array_equal(adata.X[:], x[indices, :])
    np.testing.assert_array_equal(adata.obs["txt"], np.array(list(obs[i] for i in indices)))
    np.testing.assert_array_equal(adata.obsm["x"], obsm[indices, :])
    np.testing.assert_array_equal(adata.obsm["y"].todense(), obsm[indices, :])

def test_chunk(tmp_path):
    X = random(5000, 50, 0.1, format="csr", dtype=np.int64)
    adata = AnnData(
        X=X,
        filename=h5ad(tmp_path),
    )
    s = X.sum(axis = 0)
    s_ = np.zeros_like(s)
    for m, _, _ in adata.X.chunked(47):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)
    s_ = np.zeros_like(s)
    for m, _, _ in adata.X.chunked(500000):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)

    x1 = random(2321, 50, 0.1, format="csr", dtype=np.int64)
    x2 = random(2921, 50, 0.1, format="csr", dtype=np.int64)
    x3 = random(1340, 50, 0.1, format="csr", dtype=np.int64)
    merged = vstack([x1, x2, x3])
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path))
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path))
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path))
    adata = AnnDataSet(
        [("1", adata1), ("2", adata2), ("3", adata3)],
        h5ad(tmp_path),
        "batch"
    )

    s = merged.sum(axis = 0)
    s_ = np.zeros_like(s)
    for m, _, _ in adata.X.chunked(47):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)
    s_ = np.zeros_like(s)
    for m, _, _ in adata.X.chunked(500000):
        s_ += m.sum(axis = 0)
    np.testing.assert_array_equal(s, s_)

@given(
    x1 = arrays(np.int64, (15, 179)),
    x2 = arrays(np.int64, (47, 179)),
    x3 = arrays(np.int64, (77, 179)),
    idx1 = st.lists(st.integers(min_value=0, max_value=14), min_size=0, max_size=50),
    idx2 = st.lists(st.integers(min_value=15, max_value=15+46), min_size=0, max_size=50),
    idx3 = st.lists(st.integers(min_value=15+47, max_value=15+47+76), min_size=0, max_size=50),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_anndataset_subset(x1, x2, x3, idx1, idx2, idx3, tmp_path):
    # Setup
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path))
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path))
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path))
    merged = np.concatenate([x1, x2, x3], axis=0)
    dataset = AnnDataSet(
        [("1", adata1), ("2", adata2), ("3", adata3)], h5ad(tmp_path), "batch"
    )
    dataset.obs_names = dataset.obs['batch'].to_numpy() + "-" + np.array(dataset.obs_names)
    obs_names = np.array(dataset.obs_names)
    indices = idx1 + idx2 + idx3
    s = set(indices)
    boolean_mask = list(i in s for i in range(dataset.n_obs))
    shuffled_indices = np.copy(indices).astype(np.int64)
    np.random.shuffle(shuffled_indices)
    s = set(shuffled_indices)
    shuffled_boolean_mask = list(i in s for i in range(dataset.n_obs))

    # Non-inplace operations
    ## fancy indexing
    dataset_subset, _ = dataset.subset(indices, out = h5ad(tmp_path))
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[indices, :], mat)
    np.testing.assert_array_equal(obs_names[indices].tolist(), dataset_subset.obs_names)

    ## Boolean mask
    s = set(indices)
    boolean_mask = list(i in s for i in range(dataset.n_obs))
    dataset_subset, _ = dataset.subset(boolean_mask, out = h5ad(tmp_path))
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[boolean_mask, :], mat)
    np.testing.assert_array_equal(obs_names[boolean_mask].tolist(), dataset_subset.obs_names)

    # Index shuffling
    dataset_subset, reorder = dataset.subset(shuffled_indices, out = h5ad(tmp_path))
    reordered_indices = shuffled_indices[reorder]
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[reordered_indices, :], mat)
    np.testing.assert_array_equal(obs_names[reordered_indices].tolist(), dataset_subset.obs_names)

    dataset_subset, reorder = dataset.subset(shuffled_boolean_mask, out = h5ad(tmp_path))
    assert reorder is None
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[shuffled_boolean_mask, :], mat)
    np.testing.assert_array_equal(obs_names[shuffled_boolean_mask].tolist(), dataset_subset.obs_names)

    # Check if open is OK
    os.chdir(tmp_path)
    dataset_subset.subset([], out = "a_copy")

    '''
    # Inplace operations
    # fancy indexing
    dataset_subset = dataset.copy(h5ad(tmp_path))
    dataset_subset.subset(indices)
    np.testing.assert_array_equal(merged[indices, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[indices].tolist(), dataset_subset.obs_names)

    # Index shuffling
    dataset_subset = dataset.copy(h5ad(tmp_path))
    dataset_subset.subset(shuffled_indices)
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[indices, :], mat)
    np.testing.assert_array_equal(obs_names[indices].tolist(), dataset_subset.obs_names)
    '''


    '''
    dataset_subset, index = dataset.subset(shuffled_indices, out = h5ad(tmp_path))
    mat = merged[[], :] if dataset_subset.X is None else dataset_subset.X[:]
    np.testing.assert_array_equal(merged[indices, :], mat)
    np.testing.assert_array_equal(obs[indices], dataset_subset.obs['batch'])
    np.testing.assert_array_equal(obs[[shuffled_indices[i] for i in index]], dataset_subset.obs['batch'])
    '''