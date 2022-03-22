use polars::prelude::ArrowField;
use polars_core::frame::ArrowChunk;
use polars_core::utils::arrow::{array::ArrayRef, ffi};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use polars::frame::DataFrame;

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

pub(crate) fn df_to_py(mut df: DataFrame) -> PyResult<PyObject> {
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