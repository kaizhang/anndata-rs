use crate::backend::ScalarType;
use crate::data::utils::{array_major_minor_index_default, cs_major_minor_index2};
use crate::data::{DataFrameIndex, DynCsrMatrix};
use crate::{AnnDataOp, ArrayElemOp};
use anyhow::{ensure, Result};
use indexmap::IndexSet;
use itertools::Itertools;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use polars::frame::DataFrame;
use polars::prelude::{AnyValue, CategoricalChunkedBuilder, DataType, IntoLazy, NamedFrom};
use polars::series::{IntoSeries, Series};

use crate::data::{ArrayData, DynArray};

#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Outer,
}

pub fn concat<A, O, S>(
    adatas: &[A],
    join: JoinType,
    label: Option<&str>,
    keys: Option<&[S]>,
    out: &O,
) -> Result<()>
where
    A: AnnDataOp,
    O: AnnDataOp,
    S: ToString,
{
    // Concatenate var_names
    let common_vars = adatas
        .iter()
        .map(|x| x.var_names().into_iter().collect::<IndexSet<_>>());
    let common_vars: IndexSet<String> = match join {
        JoinType::Inner => common_vars.reduce(|a, b| a.intersection(&b).cloned().collect()),
        JoinType::Outer => common_vars.reduce(|a, b| a.union(&b).cloned().collect()),
    }
    .unwrap();
    out.set_var_names(common_vars.iter().cloned().collect())?;

    // Concatenate vars
    {
        let df_var = adatas
            .iter()
            .map(|adata| {
                let var = adata.read_var().unwrap();
                let var_names = adata.var_names();
                // Creating the series
                let columns = var
                    .get_columns()
                    .iter()
                    .map(|s| align_series(s, &var_names, &common_vars))
                    .collect::<Result<Vec<_>>>()?;
                Ok(DataFrame::new(columns)?)
            })
            .reduce(|a, b| {
                let mut a = a?;
                merge_df(&mut a, &b?)?;
                anyhow::Ok(a)
            })
            .unwrap()?;
        out.set_var(df_var)?;
    }

    // Concatenate obs
    {
        let obs_names = adatas.iter().flat_map(|adata| adata.obs_names()).collect();
        out.set_obs_names(obs_names)?;

        let mut dfs = adatas
            .iter()
            .map(|adata| adata.read_obs().unwrap())
            .collect::<Vec<_>>();
        if let Some(keys) = keys {
            dfs.iter_mut().zip_eq(keys.iter()).for_each(|(df, key)| {
                let s = Series::new(
                    label.unwrap_or("label").into(),
                    vec![key.to_string(); df.height()],
                );
                df.insert_column(0, s).unwrap();
            });
        }
        let dfs = dfs.into_iter().map(|df| df.lazy()).collect::<Vec<_>>();
        let mut args = polars::prelude::UnionArgs::default();
        match join {
            JoinType::Inner => args.diagonal = false,
            JoinType::Outer => args.diagonal = true,
        };
        let df_obs = polars::prelude::concat(&dfs, args)?.collect()?;
        out.set_obs(df_obs)?;
    }

    // Concatenate X
    {
        if adatas.iter().any(|adata| !adata.x().is_none()) {
            let dtype = adatas
                .iter()
                .flat_map(|x| x.x().dtype().and_then(|d| d.scalar_type()))
                .next()
                .unwrap();
            let x_arr = adatas.iter().map(|adata| {
                let n_obs = adata.n_obs();
                let n_vars = adata.n_vars();
                let var_names = adata.var_names();

                macro_rules! fun {
                    ($variant:ident) => {
                        CsrMatrix::<$variant>::zeros(n_obs, n_vars).into()
                    };
                }

                adata
                    .x()
                    .get()
                    .unwrap()
                    .map(|arr| {
                        index_array(
                            arr,
                            &(0..adata.n_obs())
                                .into_iter()
                                .map(|x| Some(x))
                                .collect::<Vec<_>>(),
                            &common_vars
                                .iter()
                                .map(|x| var_names.get_index(x))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .unwrap_or_else(|| crate::macros::dyn_match!(dtype, ScalarType, fun))
            });
            out.set_x_from_iter(x_arr)?;
        }
    }

    Ok(())
}

fn merge_df(this: &mut DataFrame, other: &DataFrame) -> Result<()> {
    if other.is_empty() {
        return Ok(());
    }
    ensure!(
        this.height() == other.height(),
        "DataFrames must have the same number of rows"
    );
    other.get_columns().iter().try_for_each(|other_s| {
        let name = other_s.name();
        if let Some(i) = this.get_column_index(name) {
            let this_s = this.column(name)?;
            let new_column = this_s
                .iter()
                .zip(other_s.iter())
                .map(|(this_v, other_v)| {
                    if other_v.is_null() {
                        this_v.clone()
                    } else {
                        other_v.clone()
                    }
                })
                .collect::<Vec<_>>();
            let dtype = match (this_s.dtype(), other_s.dtype()) {
                (DataType::Categorical(_, _), _) => this_s.dtype(),
                (_, DataType::Categorical(_, _)) => other_s.dtype(),
                _ => this_s.dtype(),
            };
            let new_column = match dtype {
                DataType::Categorical(_, ord) => {
                    CategoricalChunkedBuilder::new(name.clone(), this.height(), ord.clone())
                        .drain_iter_and_finish(new_column.iter().map(|x| x.get_str()))
                        .into_series()
                }
                dtype => {
                    Series::from_any_values_and_dtype(name.clone(), &new_column, dtype, false)?
                }
            };
            this.replace_column(i, new_column)?;
        } else {
            this.insert_column(this.width(), other_s.clone())?;
        }
        anyhow::Ok(())
    })?;
    Ok(())
}

/// Reorganize a series to match the new row names, filling in missing values with `None`.
fn align_series(
    series: &Series,
    row_names: &DataFrameIndex,
    new_row_names: &IndexSet<String>,
) -> Result<Series> {
    let name = series.name();
    let new_series = match series.dtype() {
        DataType::Categorical(_, ord) => {
            let builder =
                CategoricalChunkedBuilder::new(name.clone(), new_row_names.len(), ord.clone());
            let values: Vec<_> = new_row_names
                .iter()
                .map(|key| row_names.get_index(key).map(|i| series.get(i).unwrap()))
                .collect();
            builder
                .drain_iter_and_finish(values.iter().map(|v| {
                    if let Some(v) = v {
                        v.get_str()
                    } else {
                        None
                    }
                }))
                .into_series()
        }
        dtype => {
            let values: Result<Vec<_>> = new_row_names
                .iter()
                .map(|key| {
                    if let Some(i) = row_names.get_index(key) {
                        Ok(series.get(i)?)
                    } else {
                        Ok(AnyValue::Null)
                    }
                })
                .collect();
            Series::from_any_values_and_dtype(name.clone(), &values?, &dtype, false)?
        }
    };
    Ok(new_series)
}

fn index_array(
    arr: ArrayData,
    row_indices: &[Option<usize>],
    col_indices: &[Option<usize>],
) -> ArrayData {
    macro_rules! fun_array {
        ($variant:ident, $value:expr) => {
            array_major_minor_index_default(
                row_indices,
                col_indices,
                &$value.into_dimensionality().unwrap(),
            )
            .into()
        };
    }

    macro_rules! fun_csr {
        ($variant:ident, $value:expr) => {{
            let (offsets, indices, data) = $value.csr_data();
            let (new_row_offsets, new_col_indices, new_data) = cs_major_minor_index2(
                row_indices,
                col_indices,
                $value.ncols(),
                offsets,
                indices,
                data,
            );
            let pattern = unsafe {
                SparsityPattern::from_offset_and_indices_unchecked(
                    row_indices.len(),
                    col_indices.len(),
                    new_row_offsets,
                    new_col_indices,
                )
            };
            CsrMatrix::try_from_pattern_and_values(pattern, new_data)
                .unwrap()
                .into()
        }};
    }

    match arr {
        ArrayData::Array(x) => crate::macros::dyn_map!(x, DynArray, fun_array),
        ArrayData::CsrMatrix(x) => crate::macros::dyn_map!(x, DynCsrMatrix, fun_csr),
        _ => todo!(),
    }
}