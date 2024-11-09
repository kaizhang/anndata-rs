from hypothesis import Phase, given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from anndata_rs import AnnData, AnnDataSet
import os
import pytest

import polars as pl
import numpy as np
from pathlib import Path
import uuid
from scipy.sparse import csr_matrix, random, vstack

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

@pytest.mark.parametrize("backend", ["hdf5", "zarr"])
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
@settings(
    deadline=None,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
    suppress_health_check = [HealthCheck.function_scoped_fixture],
)
def test_subset(x, obs, obsm, obsp, varm, varp, indices, indices2, tmp_path, backend):
    ident = list(map(lambda x: str(x), range(len(obs))))
    adata = AnnData(
        X=x,
        obs = dict(ident=ident, txt=obs),
        obsm = dict(x=obsm, y=csr_matrix(obsm)),
        varm = dict(x=varm, y=csr_matrix(varm)),
        filename = h5ad(tmp_path),
        backend=backend,
    )
    adata.obsp = dict(x=obsp, y=csr_matrix(obsp))
    adata.varp = dict(x=varp, y=csr_matrix(varp))
    adata.layers["raw"] = x

    for adata_subset in [ adata.subset(indices, indices2, out=h5ad(tmp_path), inplace=False, backend=backend),
                         adata.subset(indices, indices2, inplace=False)]:
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
        np.testing.assert_array_equal(adata_subset.layers["raw"], x[np.ix_(indices, indices2)])

    for adata_subset in [ adata.subset([str(x) for x in indices], out=h5ad(tmp_path), inplace=False, backend=backend),
                         adata.subset([str(x) for x in indices], inplace=False) ]:
        np.testing.assert_array_equal(adata_subset.X[:], x[indices, :])
        np.testing.assert_array_equal(adata_subset.obs["txt"], np.array(list(obs[i] for i in indices)))
        np.testing.assert_array_equal(adata_subset.obsm["x"], obsm[indices, :])
        np.testing.assert_array_equal(adata_subset.obsm["y"].todense(), obsm[indices, :])
        np.testing.assert_array_equal(adata_subset.layers["raw"], x[indices, :])

    for adata_subset in [ adata.subset(pl.Series([str(x) for x in indices]), out=h5ad(tmp_path), inplace=False, backend=backend),
                         adata.subset(pl.Series([str(x) for x in indices]), inplace=False) ]:
        np.testing.assert_array_equal(adata_subset.X[:], x[indices, :])
        np.testing.assert_array_equal(adata_subset.layers["raw"], x[indices, :])

    adata.subset(indices)
    np.testing.assert_array_equal(adata.X[:], x[indices, :])
    np.testing.assert_array_equal(adata.obs["txt"], np.array(list(obs[i] for i in indices)))
    np.testing.assert_array_equal(adata.obsm["x"], obsm[indices, :])
    np.testing.assert_array_equal(adata.obsm["y"].todense(), obsm[indices, :])
    np.testing.assert_array_equal(adata_subset.layers["raw"], x[indices, :])

@pytest.mark.parametrize("backend", ["hdf5", "zarr"])
def test_chunk(tmp_path, backend):
    X = random(5000, 50, 0.1, format="csr", dtype=np.int64)
    adata = AnnData(
        X=X,
        filename=h5ad(tmp_path),
        backend=backend,
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
    adata1 = AnnData(X=x1, filename=h5ad(tmp_path), backend=backend)
    adata2 = AnnData(X=x2, filename=h5ad(tmp_path), backend=backend)
    adata3 = AnnData(X=x3, filename=h5ad(tmp_path), backend=backend)
    adata = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
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

@pytest.mark.parametrize("backend", ["hdf5"])
@given(
    x1 = arrays(np.int64, (15, 179)),
    x2 = arrays(np.int64, (47, 179)),
    x3 = arrays(np.int64, (77, 179)),
    idx1 = st.lists(st.integers(min_value=0, max_value=14), min_size=0, max_size=50),
    idx2 = st.lists(st.integers(min_value=15, max_value=15+46), min_size=0, max_size=50),
    idx3 = st.lists(st.integers(min_value=15+47, max_value=15+47+76), min_size=0, max_size=50),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_anndataset_subset(x1, x2, x3, idx1, idx2, idx3, tmp_path, backend):
    # Setup
    n = x1.shape[0]
    obs_names = list(map(lambda x: str(x), range(n)))
    adata1 = AnnData(
        X=x1,
        var = dict(ident=list(map(lambda x: str(x), range(x1.shape[1])))),
        filename=h5ad(tmp_path),
        backend=backend,
    )
    adata1.obs_names = obs_names
    obs_names = list(map(lambda x: str(x), range(n, n + x2.shape[0])))
    adata2 = AnnData(
        X=x2,
        var = dict(ident=list(map(lambda x: str(x), range(x1.shape[1])))),
        filename=h5ad(tmp_path),
        backend=backend,
    )
    adata2.obs_names = obs_names
    n += x2.shape[0]
    obs_names = list(map(lambda x: str(x), range(n, n + x3.shape[0])))
    adata3 = AnnData(
        X=x3,
        var = dict(ident=list(map(lambda x: str(x), range(x1.shape[1])))),
        filename=h5ad(tmp_path),
        backend=backend,
    )
    adata3.obs_names = obs_names
    merged = np.concatenate([x1, x2, x3], axis=0)
    dataset = AnnDataSet(
        adatas=[("1", adata1), ("2", adata2), ("3", adata3)],
        filename=h5ad(tmp_path),
        add_key="batch",
        backend=backend,
    )
    obs_names = np.array(dataset.obs_names)
    indices = idx1 + idx2 + idx3
    s = set(indices)
    boolean_mask = list(i in s for i in range(dataset.n_obs))
    shuffled_indices = np.copy(indices).astype(np.int64)
    np.random.shuffle(shuffled_indices)
    s = set(shuffled_indices)
    shuffled_boolean_mask = list(i in s for i in range(dataset.n_obs))

    ## fancy indexing
    dataset_subset, reorder = dataset.subset(indices, out=h5ad(tmp_path), backend=backend)
    assert reorder is None
    np.testing.assert_array_equal(merged[indices, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[indices].tolist(), dataset_subset.obs_names)

    ## fancy indexing by names
    dataset_subset, reorder = dataset.subset([str(i) for i in indices], out=h5ad(tmp_path), backend=backend)
    assert reorder is None
    np.testing.assert_array_equal(merged[indices, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[indices].tolist(), dataset_subset.obs_names)

    ## Boolean mask
    s = set(indices)
    boolean_mask = list(i in s for i in range(dataset.n_obs))
    dataset_subset, reorder = dataset.subset(boolean_mask, out=h5ad(tmp_path), backend=backend)
    assert reorder is None
    np.testing.assert_array_equal(merged[boolean_mask, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[boolean_mask].tolist(), dataset_subset.obs_names)

    # Index shuffling
    dataset_subset, reorder = dataset.subset(shuffled_indices, out=h5ad(tmp_path), backend=backend)
    if reorder is not None:
        reordered_indices = shuffled_indices[reorder]
    else:
        reordered_indices = shuffled_indices
    np.testing.assert_array_equal(merged[reordered_indices, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[reordered_indices].tolist(), dataset_subset.obs_names)

    dataset_subset, reorder = dataset.subset(shuffled_boolean_mask, out = h5ad(tmp_path), backend=backend)
    assert reorder is None
    np.testing.assert_array_equal(merged[shuffled_boolean_mask, :], dataset_subset.X[:])
    np.testing.assert_array_equal(obs_names[shuffled_boolean_mask].tolist(), dataset_subset.obs_names)

    # Check if open is OK
    os.chdir(tmp_path)
    dataset_subset.subset([], out = "a_copy", backend=backend)