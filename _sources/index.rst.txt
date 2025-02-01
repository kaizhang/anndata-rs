anndata-rs: A Rust port of the AnnData package
==============================================

This package provides an alternative implementation of the anndata data format.
Currently only a subset of features are implemented.

Key differences:

1. AnnData is always opened in backed mode and synced.
2. All elements are lazily loaded ("zero" memory usage).
3. No views. Subsetting the AnnData will modify the data inplace or make a copy.

For details about the `anndata` specification, please go to:
https://anndata.readthedocs.io/en/latest/.

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   api