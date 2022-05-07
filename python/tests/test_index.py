from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
import pytest
from anndata_rs import AnnData, AnnDataSet

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
    mask = st.lists(st.booleans(), min_size=79, max_size=79),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_index(x, indices, indices2, mask, tmp_path):
    x_ = csr_matrix(x)
    adata = AnnData(
        X=x,
        obsm=dict(x=x_),
        filename = h5ad(tmp_path),
    )

    np.testing.assert_array_equal(
        adata.X[indices, indices2],
        x[np.ix_(indices, indices2)],
    )
    np.testing.assert_array_equal(adata.X[indices, :], x[indices, :])
    np.testing.assert_array_equal(adata.X[:, indices2], x[:, indices2])
    np.testing.assert_array_equal(adata.X[:, mask], x[:, mask])
    np.testing.assert_array_equal(adata.X[:, np.array(mask)], x[:, np.array(mask)])

    np.testing.assert_array_equal(
        adata.obsm.el('x')[indices, indices2].todense(),
        x[np.ix_(indices, indices2)],
    )
    np.testing.assert_array_equal(adata.obsm.el('x')[indices, :].todense(), x[indices, :])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, indices2].todense(), x[:, indices2])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, mask].todense(), x[:, mask])
    np.testing.assert_array_equal(adata.obsm.el('x')[:, np.array(mask)].todense(), x[:, np.array(mask)])