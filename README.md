anndata-rs
==========

A Rust port of the AnnData package.

Key differences:

1. AnnData is always opened in backed mode and synced.
2. All elements are lazily loaded ("zero" memory usage).
3. No views. Subsetting the AnnData will modify the data inplace or make a copy.

Example
-------

```python
>>> data = snap.read_h5ad("data.h5ad")
>>> data
AnnData object with n_obs x n_vars = 1590 x 6176550 backed at 'data.h5ad'
    obs: Cell, tsse, n_fragment, frac_dup, frac_mito, doublet_score, is_doublet, leiden
    var: Feature_ID, selected
    obsm: X_umap, insertion, X_spectral
    obsp: distances
    uns: reference_sequences, spectral_eigenvalue, scrublet_threshold, scrublet_sim_doublet_score
```

Cache can be enabled to speed up repetitive readings. We enable caching for `obs` and 
`var` by default.
The cache will be filled when the data is requested the first time.

```python
>>> data.var
DataFrameElem, cache_enabled: yes, cached: no
>>> data.var[:]
shape: (6176550, 2)
┌────────────────────────┬──────────┐
│ Feature_ID             ┆ selected │
│ ---                    ┆ ---      │
│ str                    ┆ bool     │
╞════════════════════════╪══════════╡
│ chr1:0-500             ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chr1:500-1000          ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chr1:1000-1500         ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chr1:1500-2000         ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ ...                    ┆ ...      │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chrY:57225500-57226000 ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chrY:57226000-57226500 ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chrY:57226500-57227000 ┆ false    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ chrY:57227000-57227415 ┆ false    │
└────────────────────────┴──────────┘
>>> data.var
DataFrameElem, cache_enabled: yes, cached: yes
```

`obsm`

```python
>>> data.obsm
AxisArrays (row) with keys: X_umap, insertion, X_spectral
>>> data.obsm['X_umap']
array([[13.279691  , -3.1859393 ],
       [12.367847  , -1.9303571 ],
       [11.376464  ,  0.36262953],
       ...,
       [12.1357565 , -2.777369  ],
       [12.9115095 , -1.9225913 ],
       [13.247231  , -4.200884  ]], dtype=float32)
```

Cache of individual element can be turned on or off manually.

```python
>>> data.X
1590 x 6176550 MatrixElem with CsrMatrix(Unsigned(U4)), cache_enabled: no, cached: no
>>> data.X.enable_cache()
>>> data.X
1590 x 6176550 MatrixElem with CsrMatrix(Unsigned(U4)), cache_enabled: yes, cached: no
>>> data.X.disable_cache()
>>> data.X
1590 x 6176550 MatrixElem with CsrMatrix(Unsigned(U4)), cache_enabled: no, cached: no
```