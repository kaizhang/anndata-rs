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

BACKENDS = ["hdf5"]

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

@pytest.mark.parametrize("backend", BACKENDS)
def test_split(tmp_path, backend):
    adata = AnnData(filename = h5ad(tmp_path), backend=backend)
    adata.obs_names = [str(i) for i in range(100)]
    adata.var_names = [str(i) for i in range(50)]

    adata.X = random(100, 50, 0.1, format="csr", dtype=np.int64)

    groups = ["A"] * 30 + ["B"] * 50 + ["C"] * 20
    adatas = adata.split_obs_by(groups, out_dir=tmp_path / "split1")
    np.testing.assert_array_equal(adatas['A'].X[:].todense(), adata.X[0:30, :].todense())
    np.testing.assert_array_equal(adatas['B'].X[:].todense(), adata.X[30:80, :].todense())
    np.testing.assert_array_equal(adatas['C'].X[:].todense(), adata.X[80:100, :].todense())

    groups = [None] * 30 + ["B"] * 50 + ["C"] * 20
    adatas = adata.split_obs_by(groups, out_dir=tmp_path / "split2")
    assert len(adatas) == 2
    np.testing.assert_array_equal(adatas['B'].X[:].todense(), adata.X[30:80, :].todense())
    np.testing.assert_array_equal(adatas['C'].X[:].todense(), adata.X[80:100, :].todense())

@pytest.mark.parametrize("backend", BACKENDS)
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
    np.testing.assert_array_equal(adata.obs["txt"], obs)

    # Subsetting using indices
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

    # Subsetting using names
    for adata_subset in [ adata.subset([str(x) for x in indices], out=h5ad(tmp_path), inplace=False, backend=backend),
                         adata.subset([str(x) for x in indices], inplace=False) ]:
        np.testing.assert_array_equal(adata_subset.X[:], x[indices, :])
        np.testing.assert_array_equal(adata_subset.obs["txt"], np.array(list(obs[i] for i in indices)))
        np.testing.assert_array_equal(adata_subset.obsm["x"], obsm[indices, :])
        np.testing.assert_array_equal(adata_subset.obsm["y"].todense(), obsm[indices, :])
        np.testing.assert_array_equal(adata_subset.layers["raw"], x[indices, :])

    # Subsetting using polars series
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

@pytest.mark.parametrize("backend", BACKENDS)
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