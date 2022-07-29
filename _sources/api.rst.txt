.. module:: anndata_rs

API
===

Import anndata_rs as::

    import anndata_rs as ad

Core AnnData classes
--------------------

.. autosummary::
    :toctree: _autosummary

    AnnData
    AnnDataSet
    StackedAnnData

Element classes
---------------

.. autosummary::
    :toctree: _autosummary

    PyElemCollection
    PyAxisArrays
    PyMatrixElem
    PyDataFrameElem
    PyStackedAxisArrays
    PyStackedMatrixElem
    PyStackedDataFrame

IO
---

.. autosummary::
    :toctree: _autosummary

    read
    read_mtx
    read_dataset
    create_dataset
