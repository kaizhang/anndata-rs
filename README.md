anndata-rs: A Rust/Python package for reading data in the h5ad format
=====================================================================

Motivation
----------

The goal of this library is to complement the [anndata](https://anndata.readthedocs.io/en/latest/) package by providing an out-of-core AnnData implementation.

Unlike the backed mode in the [anndata](https://anndata.readthedocs.io/en/latest/) package,
`anndata-rs`'s AnnData object is fully backed and always stays in sync with the data stored in the hard drive.

Here are the key features of this implementation:

- AnnData is fully backed by the underlying hdf5 file. Any operations on the AnnData object
  will be reflected on the hdf5 file.
- All elements are lazily loaded. No matter how large is the file, opening it
  consume almost zero memory. Matrix data can be accessed and processed by chunks,
  which keeps the memory usage to the minimum.
- In-memory cache can be turned on to speed up the repetitive access of elements.
- An AnnDataSet object to lazily concatenate multiple AnnData objects.

Limitations:

- Only a subset of the h5ad specifications are implemented. For example, the `.layer`
  and `.raw` is not supported. To request a missing feature, please open a new issue.
- No views. Subsetting the AnnData will modify the data inplace or make a copy.

Installation
------------

We do not provide installation instructions here.
Right now this package is bundled with the [SnapATAC2](https://github.com/kaizhang/SnapATAC2) package.
Please install [SnapATAC2](https://github.com/kaizhang/SnapATAC2) to get these features.

Tutorials
---------

Click [here](https://kzhang.org/epigenomics-analysis/anndata.html) to read tutorials.