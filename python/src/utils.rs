mod error;
use error::PyPolarsErr;

use polars::{
    prelude::ArrowField,
    frame::DataFrame,
};
use polars_core::{
    prelude::*,
    frame::ArrowChunk,
    utils::arrow::{array::ArrayRef, ffi},
    utils::accumulate_dataframes_vertical,
};
use pyo3::{
    ffi::Py_uintptr_t,
    prelude::*,
};

/// Arrow array to Python.
fn to_py_array(array: ArrayRef, py: Python, pyarrow: &PyModule) -> PyResult<PyObject> {
    let array_ptr = Box::new(ffi::ArrowArray::empty());
    let schema_ptr = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = Box::into_raw(array_ptr);
    let schema_ptr = Box::into_raw(schema_ptr);

    unsafe {
        ffi::export_field_to_c(
            &ArrowField::new("", array.data_type().clone(), true),
            schema_ptr,
        );
        ffi::export_array_to_c(array, array_ptr);
    };

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        Box::from_raw(array_ptr);
        Box::from_raw(schema_ptr);
    };

    Ok(array.to_object(py))
}

/// RecordBatch to Python.
fn to_py_rb(
    rb: &ArrowChunk,
    names: &[&str],
    py: Python,
    pyarrow: &PyModule,
) -> PyResult<PyObject> {
    let mut arrays = Vec::with_capacity(rb.len());

    for array in rb.columns() {
        let array_object = to_py_array(array.clone(), py, pyarrow)?;
        arrays.push(array_object);
    }

    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (arrays, names.to_vec()))?;

    Ok(record.to_object(py))
}

pub(crate) fn to_py_df(mut df: DataFrame) -> PyResult<PyObject> {
    df.rechunk();
    let gil = Python::acquire_gil();
    let py = gil.python();
    let pyarrow = py.import("pyarrow")?;
    let names = df.get_column_names();

    let rbs: Vec<PyObject> = df
        .iter_chunks()
        .map(|rb| to_py_rb(&rb, &names, py, pyarrow))
        .collect::<PyResult<_>>()?;

    let table = pyarrow.getattr("Table")?
        .call_method1("from_batches", (rbs,))?;

    let polars = py.import("polars")?;
    let df = polars.getattr("from_arrow")?.call1((table,))?;
    Ok(df.to_object(py))
}

pub fn array_to_rust(obj: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)?;
        let array = ffi::import_array_from_c(array, field.data_type).map_err(PyPolarsErr::from)?;
        Ok(array.into())
    }
}

pub(crate) fn to_rust_df(pydf: &PyAny) -> PyResult<DataFrame> {
    let rb: Vec<&PyAny> = pydf.call_method0("to_arrow")?
        .call_method0("to_batches")?.extract()?;
    let schema = rb
        .get(0)
        .ok_or_else(|| PyPolarsErr::Other("empty table".into()))?
        .getattr("schema")?;
    let names = schema.getattr("names")?.extract::<Vec<String>>()?;

    let dfs = rb
        .iter()
        .map(|rb| {
            let columns = (0..names.len())
                .map(|i| {
                    let array = rb.call_method1("column", (i,))?;
                    let arr = array_to_rust(array)?;
                    let s =
                        Series::try_from((names[i].as_str(), arr)).map_err(PyPolarsErr::from)?;
                    Ok(s)
                })
                .collect::<PyResult<_>>()?;
            Ok(DataFrame::new(columns).map_err(PyPolarsErr::from)?)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(accumulate_dataframes_vertical(dfs).map_err(PyPolarsErr::from)?)
}