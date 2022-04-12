use crate::{
    anndata::AnnData,
    anndata_trait::*,
    element::*,
    iterator::IndexedCsrIterator,
};

use itertools::Itertools;
use std::sync::Arc;
use parking_lot::Mutex;
use hdf5::{File, Result}; 
use std::ops::Deref;

impl AnnData {
    pub fn read(file: File) -> Result<Self>
    {
        let n_obs = Arc::new(Mutex::new(0));
        let n_vars = Arc::new(Mutex::new(0));

        // Read X
        let x = if file.link_exists("X") {
            let x = RawMatrixElem::new(DataContainer::open(&file, "X")?)?;
            *n_obs.lock() = x.nrows();
            *n_vars.lock() = x.ncols();
            Slot::new(x)
        } else {
            Slot::empty()
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::new_elem(DataContainer::open(&file, "obs")?)?;
            let n = *n_obs.lock().deref();
            if n == 0 {
                *n_obs.lock() = obs.nrows();
            } else {
                assert!(n == obs.nrows(),
                    "Inconsistent number of observations: {} (X) != {} (obs)",
                    n, obs.nrows(),
                );
            }
            obs
        } else {
            Slot::empty()
        };

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new_elem(DataContainer::open(&file, "var")?)?;
            let n = *n_vars.lock().deref();
            if n == 0 {
                *n_vars.lock() = var.ncols();
            } else {
                assert!(n == var.ncols(),
                    "Inconsistent number of variables: {} (X) != {} (var)",
                    n, var.ncols(),
                );
            }
            var
        } else {
            Slot::empty()
        };

        /// define get_* functions
        macro_rules! def_item {
            ($field:ident, $closure:expr) => {
                let $field = {
                    let field = stringify!($field);
                    let data = file.group(field)
                        .map_or(file.create_group(field).ok(), Some)
                        .map($closure);
                    Slot(Arc::new(Mutex::new(data)))
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

        self.get_x().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file, "X"))?;
        self.get_obs().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file, "obs"))?;
        self.get_var().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file, "var"))?;
        self.get_obsm().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsm")?))?;
        self.get_obsp().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsp")?))?;
        self.get_varm().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varm")?))?;
        self.get_varp().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varp")?))?;
        self.get_uns().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("uns")?))?;
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