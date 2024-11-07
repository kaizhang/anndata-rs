use std::collections::HashMap;

use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_index, cs_major_minor_index, cs_major_slice},
    array::DynScalar,
    data_traits::*,
    slice::{SelectInfoElem, Shape},
    SelectInfoBounds, SelectInfoElemBounds,
};

use anyhow::{bail, Result};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::Ix1;

use super::super::slice::SliceBounds;

impl<T> HasShape for CscMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: BackendData + Clone> Indexable for CscMatrix<T> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        if index.len() != 2 {
            panic!("index must have length 2");
        }
        todo!()
        //self.get_entry(index[0], index[1]).map(|x| DynScalar::from(x.into_value()))
    }
}

impl<T: BackendData + Clone> Selectable for CscMatrix<T> {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let info = SelectInfoBounds::new(&info, &self.shape());
        if info.ndim() != 2 {
            panic!("index must have length 2");
        }
        let row_idx = &info.as_ref()[0];
        let col_idx = &info.as_ref()[1];
        let (col_offsets, row_indices, data) = self.csc_data();
        let (new_col_offsets, new_row_indices, new_data) = if row_idx.is_full(info.in_shape()[0]) {
            match col_idx {
                &SelectInfoElemBounds::Slice(SliceBounds { step, start, end }) => {
                    if step == 1 {
                        let (offsets, indices, data) =
                            cs_major_slice(start, end, col_offsets, row_indices, data);
                        (
                            offsets,
                            indices.iter().copied().collect(),
                            data.iter().cloned().collect(),
                        )
                    } else if step < 0 {
                        cs_major_index(
                            (start..end).step_by(step.abs() as usize).rev(),
                            col_offsets,
                            row_indices,
                            data,
                        )
                    } else {
                        cs_major_index(
                            (start..end).step_by(step as usize),
                            col_offsets,
                            row_indices,
                            data,
                        )
                    }
                }
                SelectInfoElemBounds::Index(idx) => {
                    cs_major_index(idx.iter().copied(), col_offsets, row_indices, data)
                }
            }
        } else {
            // row_idx not full
            match col_idx {
                &SelectInfoElemBounds::Slice(SliceBounds {
                    start: col_start,
                    end: col_end,
                    step: col_step,
                }) => {
                    if col_step < 0 {
                        // col_idx is major, row_idx is minor
                        match row_idx {
                            &SelectInfoElemBounds::Slice(row) => {
                                if row.step < 0 {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                        (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                        (row.start..row.end).step_by(row.step as usize),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                idx.iter().copied(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            ),
                        }
                    } else {
                        // col_step >0, col_idx is major, row_idx is minor
                        match row_idx {
                            &SelectInfoElemBounds::Slice(row) => {
                                if row.step < 0 {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step as usize),
                                        (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step as usize),
                                        (row.start..row.end).step_by(row.step as usize),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (col_start..col_end).step_by(col_step as usize),
                                idx.iter().copied(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            ),
                        }
                    }
                }
                SelectInfoElemBounds::Index(i) => match row_idx {
                    &SelectInfoElemBounds::Slice(row) => {
                        if row.step < 0 {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            )
                        } else {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (row.start..row.end).step_by(row.step as usize),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            )
                        }
                    }
                    SelectInfoElemBounds::Index(j) => cs_major_minor_index(
                        i.iter().copied(),
                        j.iter().copied(),
                        self.nrows(),
                        col_offsets,
                        row_indices,
                        data,
                    ),
                },
            }
        };
        let out_shape = info.out_shape();
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                out_shape[1],
                out_shape[0],
                new_col_offsets,
                new_row_indices,
            )
        };
        CscMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
    }
}

impl<T: BackendData> Element for CscMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CscMatrix(T::DTYPE)
    }

    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), self.shape().into());
        MetaData::new("csc_matrix", "0.1.0", Some(metadata))
    }
}

impl<T: BackendData> Writable for CscMatrix<T> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        let shape = self.shape();

        self.metadata().save_metadata(&mut group)?;
        group.new_array_dataset("data", self.values().into(), Default::default())?;

        let num_rows = shape[0];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_rows.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .col_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.new_array_dataset("indptr", indptr_i32.into(), Default::default())?;
                group.new_array_dataset(
                    "indices",
                    self.row_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            } else {
                group.new_array_dataset(
                    "indptr",
                    self.col_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
                group.new_array_dataset(
                    "indices",
                    self.row_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_rows.saturating_sub(1)).is_ok() {
            group.new_array_dataset(
                "indptr",
                self.col_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
            group.new_array_dataset(
                "indices",
                self.row_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
        } else {
            panic!(
                "The number of rows ({}) is too large to be stored as i64",
                num_rows
            );
        }

        Ok(DataContainer::Group(group))
    }
}

impl<T: BackendData> Readable for CscMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data_type = container.encoding_type()?;
        if let DataType::CscMatrix(_) = data_type {
            let group = container.as_group()?;
            let shape: Vec<u64> = group.get_attr("shape")?;
            let data = group
                .open_dataset("data")?
                .read_array::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            let indptr: Vec<usize> = group
                .open_dataset("indptr")?
                .read_array_cast::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            let indices: Vec<usize> = group
                .open_dataset("indices")?
                .read_array_cast::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            CscMatrix::try_from_csc_data(
                shape[0] as usize,
                shape[1] as usize,
                indptr,
                indices,
                data,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))
        } else {
            bail!(
                "cannot read csc matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> ReadableArray for CscMatrix<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    // TODO: efficient implementation for slice
    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let data_type = container.encoding_type()?;
        if let DataType::CscMatrix(_) = data_type {
            if info.as_ref().len() != 2 {
                panic!("index must have length 2");
            }

            if info.iter().all(|s| s.as_ref().is_full()) {
                return Self::read(container);
            }

            let data = if let SelectInfoElem::Slice(s) = info[1].as_ref() {
                let group = container.as_group()?;
                let indptr_slice = if let Some(end) = s.end {
                    SelectInfoElem::from(s.start..end + 1)
                } else {
                    SelectInfoElem::from(s.start..)
                };
                let mut indptr: Vec<usize> = group
                    .open_dataset("indptr")?
                    .read_array_slice_cast(&[indptr_slice])?
                    .to_vec();
                let lo = indptr[0];
                let slice = SelectInfoElem::from(lo..indptr[indptr.len() - 1]);
                let data: Vec<T> = group
                    .open_dataset("data")?
                    .read_array_slice(&[&slice])?
                    .to_vec();
                let indices: Vec<usize> = group
                    .open_dataset("indices")?
                    .read_array_slice_cast(&[&slice])?
                    .to_vec();
                indptr.iter_mut().for_each(|x| *x -= lo);
                CscMatrix::try_from_csc_data(
                    Self::get_shape(container)?[0],
                    indptr.len() - 1,
                    indptr,
                    indices,
                    data,
                )
                .unwrap()
                .select_axis(0, info[0].as_ref()) // selct axis 1, then select axis 0 elements
            } else {
                Self::read(container)?.select(info)
            };
            Ok(data)
        } else {
            bail!(
                "cannot read csc matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> WritableArray for &CscMatrix<T> {}
impl<T: BackendData> WritableArray for CscMatrix<T> {}

#[cfg(test)]
mod csc_matrix_index_tests {
    use super::*;
    use crate::s;
    use nalgebra::base::DMatrix;
    use nalgebra_sparse::CooMatrix;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn csc_select<I1, I2>(csc: &CscMatrix<i64>, row_indices: I1, col_indices: I2) -> CscMatrix<i64>
    where
        I1: Iterator<Item = usize>,
        I2: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_rows(&i).select_columns(&j))
    }

    fn csc_select_rows<I>(csc: &CscMatrix<i64>, row_indices: I) -> CscMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_rows(&i))
    }

    fn csc_select_cols<I>(csc: &CscMatrix<i64>, col_indices: I) -> CscMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_columns(&j))
    }

    #[test]
    fn test_csc() {
        let n: usize = 200;
        let m: usize = 200;
        let nnz: usize = 1000;

        let ridx = Array::random(220, Uniform::new(0, n)).to_vec();
        let cidx = Array::random(100, Uniform::new(0, m)).to_vec();

        let row_indices = Array::random(nnz, Uniform::new(0, n)).to_vec();
        let col_indices = Array::random(nnz, Uniform::new(0, m)).to_vec();
        let values = Array::random(nnz, Uniform::new(-10000, 10000)).to_vec();

        let csc_matrix: CscMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();

        // Row slice
        assert_eq!(
            csc_matrix.select(s![2..177, ..].as_ref()),
            csc_select_rows(&csc_matrix, 2..177),
        );
        assert_eq!(
            csc_matrix.select(s![0..2, ..].as_ref()),
            csc_select_rows(&csc_matrix, 0..2),
        );

        // Row fancy indexing
        assert_eq!(
            csc_matrix.select(s![&ridx, ..].as_ref()),
            csc_select_rows(&csc_matrix, ridx.iter().cloned()),
        );

        // Column slice
        assert_eq!(
            csc_matrix.select(s![.., 77..200].as_ref()),
            csc_select_cols(&csc_matrix, 77..200),
        );

        // Column fancy indexing
        assert_eq!(
            csc_matrix.select(s![.., &cidx].as_ref()),
            csc_select_cols(&csc_matrix, cidx.iter().cloned()),
        );

        // Both
        assert_eq!(
            csc_matrix.select(s![2..49, 0..77].as_ref()),
            csc_select(&csc_matrix, 2..49, 0..77),
        );

        assert_eq!(
            csc_matrix.select(s![2..177, &cidx].as_ref()),
            csc_select(&csc_matrix, 2..177, cidx.iter().cloned()),
        );

        assert_eq!(
            csc_matrix.select(s![&ridx, &cidx].as_ref()),
            csc_select(&csc_matrix, ridx.iter().cloned(), cidx.iter().cloned()),
        );
    }
}
