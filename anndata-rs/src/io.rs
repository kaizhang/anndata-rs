use crate::{anndata::{AnnData, AnnDataOp}, data::*, element::*, iterator::IndexedCsrIterator};
use crate::element::collection::InnerAxisArrays;

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
    pub fn write<P>(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>, filename: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let file = File::create(filename)?;

        self.get_x().write(obs_idx, var_idx, &file, "X")?;
        self.get_obs().write(obs_idx, &file, "obs")?;
        self.get_var().write(var_idx, &file, "var")?;
        self.get_obsm().write(obs_idx, &file.create_group("obsm")?)?;
        self.get_obsp().write(obs_idx, &file.create_group("obsp")?)?;
        self.get_varm().write(var_idx, &file.create_group("varm")?)?;
        self.get_varp().write(var_idx, &file.create_group("varp")?)?;
        self.get_uns().write(&file.create_group("uns")?)?;
        file.close()?;
        Ok(())
    }

    pub fn copy<P>(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>, filename: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        self.write(obs_idx, var_idx, filename.as_ref().clone())?;
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
        def_item!(obsm, |x| InnerAxisArrays::new(x, Axis::Row, n_obs.clone()));
        def_item!(obsp, |x| InnerAxisArrays::new(x, Axis::Both, n_obs.clone()));
        def_item!(varm, |x| InnerAxisArrays::new(x, Axis::Row, n_vars.clone()));
        def_item!(varp, |x| InnerAxisArrays::new(x, Axis::Both, n_vars.clone()));
        let uns = ElemCollection::try_from(file.group("uns").or(file.create_group("uns"))?)?;

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

    // TODO: fix dataframe index
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
            self.set_obs(Some(&DataFrame::new(vec![series])?))?;
        }
        if has_header {
            self.set_var(Some(
                &DataFrame::new(vec![Series::new("Index", colnames)])?
            ))?;
        }
        let data: Box<dyn MatrixData> = Box::new(
            df.to_ndarray::<polars::datatypes::Float64Type>()?.into_dyn()
        );
        self.set_x(Some(&data))?;
        Ok(())
    }
}
