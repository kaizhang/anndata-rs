use crate::{
    base::AnnData,
    anndata_trait::*,
    element::{Elem, MatrixElem, MatrixElemOptional, DataFrameElem},
    iterator::IndexedCsrIterator,
};

use itertools::Itertools;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use hdf5::{File, Result, Group}; 

impl AnnData {
    pub fn read(file: File) -> Result<Self>
    {
        let mut n_obs = None;
        let mut n_vars = None;

        // Read X
        let x = if file.link_exists("X") {
            let x = MatrixElemOptional::new(DataContainer::open(&file, "X")?)?;
            n_obs = x.nrows();
            n_vars = x.ncols();
            x
        } else {
            MatrixElemOptional::empty()
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::new(DataContainer::open(&file, "obs")?)?;
            if n_obs.is_none() { n_obs = obs.nrows(); }
            assert!(n_obs == obs.nrows(),
                "Inconsistent number of observations: {} (X) != {} (obs)",
                n_obs.unwrap(), obs.nrows().unwrap(),
            );
            obs
        } else {
            DataFrameElem::empty()
        };

        // Read obsm
        let obsm = file.group("obsm").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in obsm.iter() {
            if n_obs.is_none() { n_obs = Some(v.nrows()); }
            assert!(n_obs.unwrap() == v.nrows(), 
                "Inconsistent number of observations: {} (X) != {} ({})",
                n_obs.unwrap(), v.nrows(), k,
            );
        }

        // Read obsp
        let obsp = file.group("obsp").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in obsp.iter() {
            if n_obs.is_none() { n_obs = Some(v.nrows()); }
            assert!(n_obs.unwrap() == v.nrows(), 
                "Inconsistent number of observations: {} (X) != {} ({})",
                n_obs.unwrap(), v.nrows(), k,
            );
            assert!(v.ncols() == v.nrows(), 
                "Not a square matrix: {}", k
            );
        }

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new(DataContainer::open(&file, "var")?)?;
            if n_vars.is_none() { n_vars = var.ncols(); }
            assert!(n_vars == var.ncols(),
                "Inconsistent number of variables: {} (X) != {} (var)",
                n_vars.unwrap(), var.ncols().unwrap(),
            );
            var
        } else {
            DataFrameElem::empty()
        };

        // Read varm
        let varm = file.group("varm").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in varm.iter() {
            if n_vars.is_none() { n_vars = Some(v.ncols()); }
            assert!(n_vars.unwrap() == v.ncols(), 
                "Inconsistent number of variables: {} (X) != {} ({})",
                n_vars.unwrap(), v.ncols(), k,
            );
        }

        // Read varp
        let varp = file.group("varp").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in varp.iter() {
            if n_vars.is_none() { n_vars = Some(v.ncols()); }
            assert!(n_vars.unwrap() == v.ncols(), 
                "Inconsistent number of variables: {} (X) != {} ({})",
                n_vars.unwrap(), v.ncols(), k,
            );
            assert!(v.ncols() == v.nrows(), 
                "Not a square matrix: {}", k
            );
        }

        // Read uns
        let uns = file.group("uns").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, Elem::new(v).unwrap())).collect()
        );

        Ok(Self {
            file,
            n_obs: Arc::new(Mutex::new(n_obs.unwrap_or(0))),
            n_vars: Arc::new(Mutex::new(n_vars.unwrap_or(0))),
            x, obs, obsm, obsp, var, varm, varp, uns,
        })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    {
        let file = File::create(filename)?;

        self.x.write(&file, "X")?;
        self.obs.write(&file, "obs")?;
        self.var.write(&file, "var")?;
        let obsm = file.create_group("obsm")?;
        for (key, val) in self.obsm.iter() {
            val.write(&obsm, key)?;
        }
        let varm = file.create_group("varm")?;
        for (key, val) in self.varm.iter() {
            val.write(&varm, key)?;
        }
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
                    let (_, num_cols, iter) = crate::utils::io::
                        read_sorted_mm_body_from_bufread::<R, i64>(reader);
                    self.set_x_from_row_iter(IndexedCsrIterator {
                        iterator: iter.group_by(|x| x.0).into_iter().map(|(k, grp)|
                            (k, grp.map(|(_, j, v)| (j, v)).collect())
                        ),
                        num_cols,
                    })
                },
                _ => {
                    let (_, num_cols, iter) = crate::utils::io::
                        read_sorted_mm_body_from_bufread::<R, f64>(reader);
                    self.set_x_from_row_iter(IndexedCsrIterator {
                        iterator: iter.group_by(|x| x.0).into_iter().map(|(k, grp)|
                            (k, grp.map(|(_, j, v)| (j, v)).collect())
                        ),
                        num_cols,
                    })
                },
            }
        } else {
            self.set_x(&crate::utils::io::read_matrix_market_from_bufread(reader).unwrap())
        }
    }
}

fn get_all_data(group: &Group) -> impl Iterator<Item=(String, DataContainer)> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    group.groups().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Group(x))
    ).chain(group.datasets().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Dataset(x))
    ))
}


