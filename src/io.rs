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
                *n_obs.lock().unwrap() = obs.nrows().unwrap();
            } else {
                assert!(n == obs.nrows().unwrap(),
                    "Inconsistent number of observations: {} (X) != {} (obs)",
                    n, obs.nrows().unwrap(),
                );
            }
            obs
        } else {
            DataFrameElem::empty()
        };

        // Read obsm
        let obsm = AxisArrays::new(
            match file.group("obsm") {
                Ok(g) => g,
                _ => file.create_group("obsm")?,
            },
            Axis::Row,
            n_obs.clone(),
        );
        
        // Read obsp
        let obsp = AxisArrays::new(
            match file.group("obsp") {
                Ok(g) => g,
                _ => file.create_group("obsp")?,
            },
            Axis::Both,
            n_obs.clone(),
        );

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new(DataContainer::open(&file, "var")?)?;
            let n = *n_vars.lock().unwrap().deref();
            if n == 0 {
                *n_vars.lock().unwrap() = var.ncols().unwrap();
            } else {
                assert!(n == var.ncols().unwrap(),
                    "Inconsistent number of variables: {} (X) != {} (var)",
                    n, var.ncols().unwrap(),
                );
            }
            var
        } else {
            DataFrameElem::empty()
        };

        // Read varm
        let varm = AxisArrays::new(
            match file.group("varm") {
                Ok(g) => g,
                _ => file.create_group("varm")?,
            },
            Axis::Column,
            n_vars.clone(),
        );

        // Read varp
        let varp = AxisArrays::new(
            match file.group("varp") {
                Ok(g) => g,
                _ => file.create_group("varp")?,
            },
            Axis::Both,
            n_vars.clone(),
        );

        // Read uns
        let uns = ElemCollection::new(
            match file.group("uns") {
                Ok(g) => g,
                _ => file.create_group("uns")?,
            }
        );

        Ok(Self { file, n_obs, n_vars, x, obs, obsm, obsp, var, varm, varp, uns })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    {
        let file = File::create(filename)?;

        self.x.lock().unwrap().as_ref().map_or(Ok(()), |x| x.write(&file, "X"))?;
        self.obs.write(&file, "obs")?;
        self.var.write(&file, "var")?;
        self.obsm.write(&file.create_group("obsm")?)?;
        self.obsp.write(&file.create_group("obsp")?)?;
        self.varm.write(&file.create_group("varm")?)?;
        self.varp.write(&file.create_group("varp")?)?;
        self.uns.write(&file.create_group("uns")?)?;
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