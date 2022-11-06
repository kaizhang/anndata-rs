use crate::{anndata::{AnnData, AnnDataOp}, data::*, element::*, iterator::IndexedCsrIterator};

use itertools::Itertools;
use parking_lot::Mutex;
use hdf5::File; 
use anyhow::Result;
use std::{path::PathBuf, sync::Arc};
use polars::{
    frame::DataFrame, series::Series, prelude::NamedFrom, prelude::SerReader,
};
use std::path::Path;

impl AnnData {
    pub fn write<P: AsRef<Path>>(&self, filename: P) -> Result<()>
    {
        let file = File::create(filename)?;

        self.get_x().write(None, None, &file, "X")?;
        self.get_obs().write(&file, "obs")?;
        self.get_var().write(&file, "var")?;
        self.get_obsm().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsm")?))?;
        self.get_obsp().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("obsp")?))?;
        self.get_varm().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varm")?))?;
        self.get_varp().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("varp")?))?;
        self.get_uns().inner().0.as_ref().map_or(Ok(()), |x| x.write(&file.create_group("uns")?))?;
        file.close()?;
        Ok(())
    }

    // TODO: refactoring
    pub fn write_subset<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        filename: P,
    ) -> Result<()>
    {
        match obs_idx {
            Some(i) => {
                let file = File::create(filename)?;
                self.obs.write_rows(i, &file, "obs")?;
                self.obsm.inner().0.as_mut()
                    .map(|x| x.write_subset(i, &file.create_group("obsm")?));
                self.obsp.inner().0.as_mut()
                    .map(|x| x.write_subset(i, &file.create_group("obsp")?));
                match var_idx {
                    Some(j) => {
                        self.x.write(Some(i), Some(j), &file, "X")?;
                        self.var.write_rows(j, &file, "var")?;
                        self.varm.inner().0.as_mut()
                            .map(|x| x.write_subset(j, &file.create_group("varm")?));
                        self.varp.inner().0.as_mut()
                            .map(|x| x.write_subset(j, &file.create_group("varp")?));
                    },
                    None => {
                        self.x.write(Some(i), None, &file, "X")?;
                        self.var.write(&file, "var")?;
                        self.varm.inner().0.as_mut()
                            .map(|x| x.write(&file.create_group("varm")?));
                        self.varp.inner().0.as_mut()
                            .map(|x| x.write(&file.create_group("varp")?));
                    },
                }
                self.get_uns().inner().0.as_ref()
                    .map_or(Ok(()), |x| x.write(&file.create_group("uns")?))?;
                file.close()?;
            },
            None => match var_idx {
                Some(j) => {
                    let file = File::create(filename)?;
                    self.obs.write(&file, "obs")?;
                    self.obsm.inner().0.as_mut()
                        .map(|x| x.write(&file.create_group("obsm")?));
                    self.obsp.inner().0.as_mut()
                        .map(|x| x.write(&file.create_group("obsp")?));
                    self.x.write(None, Some(j), &file, "X")?;
                    self.var.write_rows(j, &file, "var")?;
                    self.varm.inner().0.as_mut()
                        .map(|x| x.write_subset(j, &file.create_group("varm")?));
                    self.varp.inner().0.as_mut()
                        .map(|x| x.write_subset(j, &file.create_group("varp")?));
                    self.get_uns().inner().0.as_ref()
                        .map_or(Ok(()), |x| x.write(&file.create_group("uns")?))?;
                    file.close()?;
                },
                None => { self.write(filename)?; },
            },
        }
        Ok(())
    }

    pub fn copy<P: AsRef<Path>>(&self, filename: P) -> Result<Self> {
        self.write(filename.as_ref().clone())?;
        Self::read(File::open_rw(filename)?)
    }

    pub fn copy_subset<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        filename: P,
    ) -> Result<Self>
    {
        self.write_subset(obs_idx, var_idx, filename.as_ref().clone())?;
        Self::read(File::open_rw(filename)?)
    }

    pub fn read(file: File) -> Result<Self>
    {
        let n_obs = Arc::new(Mutex::new(0));
        let n_vars = Arc::new(Mutex::new(0));

        // Read X
        let x = if file.link_exists("X") {
            let x = MatrixElem::try_from(DataContainer::open(&file, "X")?)?;
            *n_obs.lock() = x.nrows();
            *n_vars.lock() = x.ncols();
            x
        } else {
            Slot::empty()
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::try_from(DataContainer::open(&file, "obs")?)?;
            let n = *n_obs.lock();
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
            let var = DataFrameElem::try_from(DataContainer::open(&file, "var")?)?;
            let n = *n_vars.lock();
            if n == 0 {
                *n_vars.lock() = var.nrows();
            } else {
                assert!(n == var.nrows(),
                    "Inconsistent number of variables: {} (X) != {} (var)",
                    n, var.nrows(),
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
        def_item!(varm, |x| AxisArrays::new(x, Axis::Row, n_vars.clone()));
        def_item!(varp, |x| AxisArrays::new(x, Axis::Both, n_vars.clone()));
        def_item!(uns, |x| ElemCollection::try_from(x).unwrap());

        Ok(Self { file, n_obs, n_vars, x, obs, obsm, obsp, var, varm, varp, uns })
    }

    pub fn import_matrix_market<R>(&self,
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
                    self.write_x_from_row_iter(IndexedCsrIterator {
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
                    self.write_x_from_row_iter(IndexedCsrIterator {
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

    pub fn import_csv<P>(&self,
        path: P,
        has_header: bool,
        index_column: Option<usize>,
        delimiter: u8,
    ) -> Result<()>
    where
        P: Into<PathBuf>,
    {
        let mut df = polars::prelude::CsvReader::from_path(path)?
            .has_header(has_header)
            .with_delimiter(delimiter)
            .finish()?;
        let mut colnames = df.get_column_names_owned();
        if let Some(idx_col) = index_column {
            let series = df.drop_in_place(&colnames.remove(idx_col))?;
            self.write_obs(Some(&DataFrame::new(vec![series])?))?;
        }
        if has_header {
            self.write_var(Some(
                &DataFrame::new(vec![Series::new("Index", colnames)])?
            ))?;
        }
        let data: Box<dyn DataPartialIO> = Box::new(
            df.to_ndarray::<polars::datatypes::Float64Type>()?.into_dyn()
        );
        self.set_x(Some(&data))?;
        Ok(())
    }
}
