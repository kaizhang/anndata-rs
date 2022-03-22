mod utils;
use utils::df_to_py;

use pyo3::prelude::*;
use pyo3::types::PyIterator;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, PyArray1, PyReadonlyArray, Ix1, Ix2, PyArray, IntoPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use nalgebra_sparse::csr::CsrMatrix;
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize;
use hdf5::types::FloatSize;
use std::collections::HashMap;
use ndarray::ArrayD;
use polars::frame::DataFrame;

use anndata_rs::anndata_trait::DataType;
use anndata_rs::backed::{AnnData, Elem2dView};
use anndata_rs::anndata_trait::DataSubset2D;

#[pyclass]
#[repr(transparent)]
pub struct PyAnnData(AnnData);

#[pymethods]
impl PyAnnData {
    fn get_x(&self) -> PyResult<PyElem2dView> {
        Ok(PyElem2dView(self.0.x.clone()))
    }

    fn get_obs(&self) -> PyResult<PyElem2dView> {
        Ok(PyElem2dView(self.0.obs.clone()))
    }

    fn get_obsm(&self) -> PyResult<HashMap<String, PyElem2dView>> {
        let obsm = self.0.obsm.iter()
            .map(|(k, x)| (k.clone(), PyElem2dView(x.clone())))
            .collect();
        Ok(obsm)
    }

    fn get_var(&self) -> PyResult<PyElem2dView> {
        Ok(PyElem2dView(self.0.var.clone()))
    }

    fn get_varm(&self) -> PyResult<HashMap<String, PyElem2dView>> {
        let varm = self.0.varm.iter()
            .map(|(k, x)| (k.clone(), PyElem2dView(x.clone())))
            .collect();
        Ok(varm)
    }

    fn subset_rows(&self, idx: Vec<usize>) -> PyResult<PyAnnData> {
        Ok(PyAnnData(self.0.subset_obs(idx.as_slice())))
    }

    fn write(&self, filename: &str) -> PyResult<()> {
        self.0.write(filename).unwrap();
        Ok(())
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyElem2dView(Elem2dView);

#[pymethods]
impl PyElem2dView {
    fn get_data(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| data_to_py(py, self.0.read_elem().unwrap()))
    }
}

#[pyfunction]
fn read_anndata(filename: &str, mode: &str) -> PyResult<PyAnnData> {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let anndata = AnnData::read(file).unwrap();
    Ok(PyAnnData(anndata))
}

fn data_to_py<'py>(
    py: Python<'py>,
    data: Box<dyn DataSubset2D>,
) -> PyResult<PyObject>
{
    match data.as_ref().get_dtype() {
        DataType::CsrMatrix(Unsigned(IntSize::U4)) =>
            csr_to_scipy::<u32>(py, *data.into_any().downcast().unwrap()),
        DataType::CsrMatrix(Unsigned(IntSize::U8)) =>
            csr_to_scipy::<u64>(py, *data.into_any().downcast().unwrap()),
        DataType::CsrMatrix(Float(FloatSize::U4)) =>
            csr_to_scipy::<f32>(py, *data.into_any().downcast().unwrap()),
        DataType::CsrMatrix(Float(FloatSize::U8)) =>
            csr_to_scipy::<f64>(py, *data.into_any().downcast().unwrap()),

        DataType::Array(Float(FloatSize::U4)) => Ok((
            &*data.into_any().downcast::<ArrayD<f32>>().unwrap().into_pyarray(py)
        ).to_object(py)),
        DataType::Array(Float(FloatSize::U8)) => Ok((
            &*data.into_any().downcast::<ArrayD<f64>>().unwrap().into_pyarray(py)
        ).to_object(py)),

        DataType::DataFrame =>
            df_to_py(*data.into_any().downcast::<DataFrame>().unwrap()),

        _ => todo!(),
    }
}

fn csr_to_scipy<'py, T>(
    py: Python<'py>,
    mat: CsrMatrix<T>
) -> PyResult<PyObject>
where T: numpy::Element
{
    let n = mat.nrows();
    let m = mat.ncols();
    let (intptr, indices, data) = mat.disassemble();

    let scipy = PyModule::import(py, "scipy.sparse")?;
    Ok(scipy.getattr("csr_matrix")?.call1((
        (data.into_pyarray(py), indices.into_pyarray(py), intptr.into_pyarray(py)),
        (n, m),
    ))?.to_object(py))
}

#[pymodule]
fn _anndata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnnData>().unwrap();
    m.add_class::<PyElem2dView>().unwrap();

    m.add_function(wrap_pyfunction!(read_anndata, m)?)?;

    Ok(())
}