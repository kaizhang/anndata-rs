use std::collections::HashMap;

use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_index, cs_major_minor_index, cs_major_slice},
    data_traits::*,
    slice::{SelectInfoElem, Shape},
    SelectInfoBounds, SelectInfoElemBounds,
};

use anyhow::{anyhow, bail, Result};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::Ix1;

use super::super::slice::SliceBounds;

impl<T: BackendData> Element for CsrMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CsrMatrix(T::DTYPE)
    }

    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), self.shape().into());
        MetaData::new("csr_matrix", "0.1.0", Some(metadata))
    }
}

impl<T> HasShape for CsrMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: Clone> Selectable for CsrMatrix<T> {
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
        let (row_offsets, col_indices, data) = self.csr_data();
        let (new_row_offsets, new_col_indices, new_data) = if col_idx.is_full(info.in_shape()[1]) {
            match row_idx {
                &SelectInfoElemBounds::Slice(SliceBounds { step, start, end }) => {
                    if step == 1 {
                        let (offsets, indices, data) =
                            cs_major_slice(start, end, row_offsets, col_indices, data);
                        (
                            offsets,
                            indices.iter().copied().collect(),
                            data.iter().cloned().collect(),
                        )
                    } else if step < 0 {
                        cs_major_index(
                            (start..end).step_by(step.abs() as usize).rev(),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    } else {
                        cs_major_index(
                            (start..end).step_by(step as usize),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    }
                }
                SelectInfoElemBounds::Index(idx) => {
                    cs_major_index(idx.iter().copied(), row_offsets, col_indices, data)
                }
            }
        } else {
            match row_idx {
                &SelectInfoElemBounds::Slice(SliceBounds {
                    start: row_start,
                    end: row_end,
                    step: row_step,
                }) => {
                    if row_step < 0 {
                        match col_idx {
                            &SelectInfoElemBounds::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    } else {
                        match col_idx {
                            &SelectInfoElemBounds::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step as usize),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    }
                }
                SelectInfoElemBounds::Index(i) => match col_idx {
                    &SelectInfoElemBounds::Slice(col) => {
                        if col.step < 0 {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        } else {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step as usize),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        }
                    }
                    SelectInfoElemBounds::Index(j) => cs_major_minor_index(
                        i.iter().copied(),
                        j.iter().copied(),
                        self.ncols(),
                        row_offsets,
                        col_indices,
                        data,
                    ),
                },
            }
        };
        let out_shape = info.out_shape();
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                out_shape[0],
                out_shape[1],
                new_row_offsets,
                new_col_indices,
            )
        };
        CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
    }
}

impl<T: Clone> Stackable for CsrMatrix<T> {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        fn vstack_csr<T: Clone>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T> {
            let num_cols = this.ncols();
            let num_rows = this.nrows() + other.nrows();
            let nnz = this.nnz();
            let (mut indptr, mut indices, mut data) = this.disassemble();
            let (indptr2, indices2, data2) = other.csr_data();
            indices.extend_from_slice(indices2);
            data.extend_from_slice(data2);
            indptr2.iter().skip(1).for_each(|&i| indptr.push(i + nnz));

            let pattern = unsafe {
                SparsityPattern::from_offset_and_indices_unchecked(
                    num_rows, num_cols, indptr, indices,
                )
            };
            CsrMatrix::try_from_pattern_and_values(pattern, data).unwrap()
        }

        Ok(iter.reduce(|acc, x| vstack_csr(acc, x)).unwrap())
    }
}

impl<T: BackendData> Writable for CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        let shape = self.shape();

        self.metadata().save(&mut group)?;
        group.new_array_dataset("data", self.values().into(), Default::default())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .row_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.new_array_dataset("indptr", indptr_i32.into(), Default::default())?;
                group.new_array_dataset(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            } else {
                group.new_array_dataset(
                    "indptr",
                    self.row_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
                group.new_array_dataset(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_cols.saturating_sub(1)).is_ok() {
            group.new_array_dataset(
                "indptr",
                self.row_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
            group.new_array_dataset(
                "indices",
                self.col_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
        } else {
            panic!(
                "The number of columns ({}) is too large to be stored as i64",
                num_cols
            );
        }

        Ok(DataContainer::Group(group))
    }
}

impl<T: BackendData> Readable for CsrMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data_type = container.encoding_type()?;
        if let DataType::CsrMatrix(_) = data_type {
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
            CsrMatrix::try_from_csr_data(shape[0] as usize, shape[1] as usize, indptr, indices, data)
                .map_err(|e| anyhow!("cannot read csr matrix: {}", e))
        } else {
            bail!(
                "cannot read csr matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> ReadableArray for CsrMatrix<T> {
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
        if let DataType::CsrMatrix(_) = data_type {
            if info.as_ref().len() != 2 {
                panic!("index must have length 2");
            }

            if info.iter().all(|s| s.as_ref().is_full()) {
                return Self::read(container);
            }

            let data = if let SelectInfoElem::Slice(s) = info[0].as_ref() {
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
                CsrMatrix::try_from_csr_data(
                    indptr.len() - 1,
                    Self::get_shape(container)?[1],
                    indptr,
                    indices,
                    data,
                )
                .unwrap()
                .select_axis(1, info[1].as_ref())
            } else {
                Self::read(container)?.select(info)
            };
            Ok(data)
        } else {
            bail!(
                "cannot read csr matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> WritableArray for &CsrMatrix<T> {}
impl<T: BackendData> WritableArray for CsrMatrix<T> {}

#[cfg(test)]
mod csr_matrix_index_tests {
    use super::*;
    use crate::s;
    use nalgebra::base::DMatrix;
    use nalgebra_sparse::CooMatrix;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn csr_select<I1, I2>(csr: &CsrMatrix<i64>, row_indices: I1, col_indices: I2) -> CsrMatrix<i64>
    where
        I1: Iterator<Item = usize>,
        I2: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i).select_columns(&j))
    }

    fn csr_select_rows<I>(csr: &CsrMatrix<i64>, row_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i))
    }

    fn csr_select_cols<I>(csr: &CsrMatrix<i64>, col_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_columns(&j))
    }

    #[test]
    fn test_c() {
        let dense = DMatrix::from_row_slice(3, 3, &[1, 0, 3, 2, 0, 1, 0, 0, 4]);
        let csr = CsrMatrix::from(&dense);

        // Column fancy indexing
        let cidx = [1, 2, 0, 1, 1, 2];
        let mut expected = DMatrix::from_row_slice(
            3,
            6,
            &[0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4],
        );
        let mut expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., cidx.as_ref()].as_ref()), expected_csr,);

        expected = DMatrix::from_row_slice(3, 2, &[1, 0, 2, 0, 0, 0]);
        expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., 0..2].as_ref()), expected_csr);

        let ridx = [1, 2, 0, 1];
        expected = DMatrix::from_row_slice(
            4,
            6,
            &[
                0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1,
            ],
        );
        expected_csr = CsrMatrix::from(&expected);
        let (new_offsets, new_indices, new_data) = cs_major_minor_index(
            ridx.into_iter(),
            cidx.into_iter(),
            csr.ncols(),
            csr.row_offsets(),
            csr.col_indices(),
            csr.values(),
        );

        assert_eq!(new_offsets.as_slice(), expected_csr.row_offsets());
        assert_eq!(new_indices.as_slice(), expected_csr.col_indices());
        assert_eq!(new_data.as_slice(), expected_csr.values());
    }

    #[test]
    fn test_csr() {
        let n: usize = 200;
        let m: usize = 200;
        let nnz: usize = 1000;

        let ridx = Array::random(220, Uniform::new(0, n)).to_vec();
        let cidx = Array::random(100, Uniform::new(0, m)).to_vec();

        let row_indices = Array::random(nnz, Uniform::new(0, n)).to_vec();
        let col_indices = Array::random(nnz, Uniform::new(0, m)).to_vec();
        let values = Array::random(nnz, Uniform::new(-10000, 10000)).to_vec();

        let csr_matrix: CsrMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();

        // Row slice
        assert_eq!(
            csr_matrix.select(s![2..177, ..].as_ref()),
            csr_select_rows(&csr_matrix, 2..177),
        );

        // Row fancy indexing
        assert_eq!(
            csr_matrix.select(s![&ridx, ..].as_ref()),
            csr_select_rows(&csr_matrix, ridx.iter().cloned()),
        );

        // Column slice
        assert_eq!(
            csr_matrix.select(s![.., 77..200].as_ref()),
            csr_select_cols(&csr_matrix, 77..200),
        );

        // Column fancy indexing
        assert_eq!(
            csr_matrix.select(s![.., &cidx].as_ref()),
            csr_select_cols(&csr_matrix, cidx.iter().cloned()),
        );

        // Both
        assert_eq!(
            csr_matrix.select(s![2..49, 0..77].as_ref()),
            csr_select(&csr_matrix, 2..49, 0..77),
        );

        assert_eq!(
            csr_matrix.select(s![2..177, &cidx].as_ref()),
            csr_select(&csr_matrix, 2..177, cidx.iter().cloned()),
        );

        assert_eq!(
            csr_matrix.select(s![&ridx, &cidx].as_ref()),
            csr_select(&csr_matrix, ridx.iter().cloned(), cidx.iter().cloned()),
        );
    }
}
