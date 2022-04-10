use crate::{
    anndata::AnnData,
    anndata_trait::*,
    element::{ElemTrait, MatrixElem, DataFrameElem, ElemCollection, AxisArrays, Axis},
    iterator::IndexedCsrIterator,
};

use itertools::Itertools;
use std::sync::{Arc, Mutex};
use hdf5::{File, Result}; 
use std::ops::Deref;

impl AnnData {
    pub fn read(file: File) -> Result<Self>
    {
        let n_obs = Arc::new(Mutex::new(0));
        let n_vars = Arc::new(Mutex::new(0));

        // Read X
        let x = if file.link_exists("X") {
            let x = MatrixElem::new(DataContainer::open(&file, "X")?)?;
            *n_obs.lock().unwrap() = x.nrows();
            *n_vars.lock().unwrap() = x.ncols();
            Arc::new(Mutex::new(Some(x)))
        } else {
            Arc::new(Mutex::new(None))
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::new(DataContainer::open(&file, "obs")?)?;
            let n = *n_obs.lock().unwrap().deref();
            if n == 0 {
                *n_obs.lock().unwrap() = obs.nrows();
            } else {
                assert!(n == obs.nrows(),
                    "Inconsistent number of observations: {} (X) != {} (obs)",
                    n, obs.nrows(),
                );
            }
            Arc::new(Mutex::new(Some(obs)))
        } else {
            Arc::new(Mutex::new(None))
        };

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new(DataContainer::open(&file, "var")?)?;
            let n = *n_vars.lock().unwrap().deref();
            if n == 0 {
                *n_vars.lock().unwrap() = var.ncols();
            } else {
                assert!(n == var.ncols(),
                    "Inconsistent number of variables: {} (X) != {} (var)",
                    n, var.ncols(),
                );
            }
            Arc::new(Mutex::new(Some(var)))
        } else {
            Arc::new(Mutex::new(None))
        };

        /// define get_* functions
        macro_rules! def_item {
            ($field:ident, $closure:expr) => {
                let $field = {
                    let field = stringify!($field);
                    let data = file.group(field)
                        .map_or(file.create_group(field).ok(), Some)
                        .map($closure);
                    Arc::new(Mutex::new(data))
                };
            }
        }
        def_item!(obsm, |x| AxisArrays::new(x, Axis::Row, n_obs.clone()));
        def_item!(obsp, |x| AxisArrays::new(x, Axis::Both, n_obs.clone()));
        def_item!(varm, |x| AxisArrays::new(x, Axis::Column, n_vars.clone()));
        def_item!(varp, |x| AxisArrays::new(x, Axis::Both, n_vars.clone()));
        def_item!(uns, |x| ElemCollection::new(x));

        Ok(Self { file, n_obs, n_vars, x, obs, obsm, obsp, var, varm, varp, uns })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    {
        let file = File::create(filename)?;

        self.x.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file, "X"))?;
        self.obs.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file, "obs"))?;
        self.var.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file, "var"))?;
        self.obsm.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsm")?))?;
        self.obsp.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsp")?))?;
        self.varm.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varm")?))?;
        self.varp.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varp")?))?;
        self.uns.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file.create_group("uns")?))?;
        Ok(())
    }

    pub fn read_matrix_market<R>(&self,
        reader: &mut R,
        sorted: bool,
    ) -> Result<()>
    where
        R: std::io::BufRead,
    {
        if sorted {
            let (sym_mode, data_type) = crate::utils::io::read_header(reader).unwrap();
            if sym_mode != crate::utils::io::SymmetryMode::General {
                panic!("symmetry matrix is not supported");
            }
            match data_type {
                crate::utils::io::DataType::Integer => {
                    let (num_rows, num_cols, iter) = crate::utils::io::
                        read_sorted_mm_body_from_bufread::<R, i64>(reader);
                    self.set_x_from_row_iter(IndexedCsrIterator {
                        iterator: iter.group_by(|x| x.0).into_iter().map(|(k, grp)|
                            (k, grp.map(|(_, j, v)| (j, v)).collect())
                        ),
                        num_rows,
                        num_cols,
                    })
                },
                _ => {
                    let (num_rows, num_cols, iter) = crate::utils::io::
                        read_sorted_mm_body_from_bufread::<R, f64>(reader);
                    self.set_x_from_row_iter(IndexedCsrIterator {
                        iterator: iter.group_by(|x| x.0).into_iter().map(|(k, grp)|
                            (k, grp.map(|(_, j, v)| (j, v)).collect())
                        ),
                        num_rows,
                        num_cols,
                    })
                },
            }
        } else {
            self.set_x(Some(
                &crate::utils::io::read_matrix_market_from_bufread(reader).unwrap()
            ))
        }
    }
}