#![allow(dead_code, unused)]

use ndarray::{Array1, Array2, Slice, SliceInfo, SliceInfoElem, IxDyn};
use anyhow::{bail, Result};
use itertools::Itertools;
use std::ops::{RangeFull, Range, Index, IndexMut, RangeFrom, RangeTo};
use smallvec::{SmallVec, smallvec};

#[derive(Clone, Debug)]
pub struct Shape(SmallVec<[usize; 3]>);

impl Shape {
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_slice().iter().map(|x| x.to_string()).join(" x "))
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl AsMut<[usize]> for Shape {
    fn as_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(SmallVec::from_vec(shape))
    }
}

impl From<(usize, usize)> for Shape {
    fn from(shape: (usize, usize)) -> Self {
        Self(smallvec![shape.0, shape.1])
    }
}

impl From<&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self(SmallVec::from_slice(shape))
    }
}

impl From<usize> for Shape {
    fn from(shape: usize) -> Self {
        Self(smallvec![shape])
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(SmallVec::from_iter(iter))
    }
}

/// A multi-dimensional selection used for reading and writing to a Container.
#[derive(Debug, PartialEq, Eq)]
pub struct SelectInfo(pub Vec<SelectInfoElem>);

impl AsRef<[SelectInfoElem]> for SelectInfo {
    fn as_ref(&self) -> &[SelectInfoElem] {
        &self.0
    }
}

impl FromIterator<SelectInfoElem> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = SelectInfoElem>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl FromIterator<Slice> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = Slice>>(iter: T) -> Self {
        Self(iter.into_iter().map(SelectInfoElem::Slice).collect())
    }
}

impl<'a> FromIterator<&'a Slice> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = &'a Slice>>(iter: T) -> Self {
        Self(iter.into_iter().map(|x| SelectInfoElem::Slice(x.clone())).collect())
    }
}


impl TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for SelectInfo {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self.0.into_iter().map(|e| match e {
            SelectInfoElem::Slice(s) => Ok(SliceInfoElem::from(s)),
            _ => bail!("Cannot convert SelectInfo to SliceInfo"),
        }).collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}

impl SelectInfo {
    pub fn all(n: usize) -> Self {
        Self(vec![SelectInfoElem::Slice(SLICE_FULL); n])
    }
}



/// A selection used for reading and writing to a Container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectInfoElem {
    Index(Vec<usize>),
    Slice(Slice),
}

impl FromIterator<usize> for SelectInfoElem {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::Index(iter.into_iter().collect())
    }
}

impl From<Slice> for SelectInfoElem {
    fn from(x: Slice) -> Self {
        Self::Slice(x)
    }
}

impl From<usize> for SelectInfoElem {
    fn from(x: usize) -> Self {
        Self::Index(vec![x])
    }
}

impl From<&[usize]> for SelectInfoElem {
    fn from(x: &[usize]) -> Self {
        Self::Index(x.into_iter().map(|x| *x).collect())
    }
}

impl From<Vec<usize>> for SelectInfoElem {
    fn from(x: Vec<usize>) -> Self {
        Self::Index(x)
    }
}

impl From<&Vec<usize>> for SelectInfoElem {
    fn from(x: &Vec<usize>) -> Self {
        x.as_slice().into()
    }
}

impl From<Array1<usize>> for SelectInfoElem {
    fn from(x: Array1<usize>) -> Self {
        Self::Index(x.to_vec())
    }
}

impl From<&Array1<usize>> for SelectInfoElem {
    fn from(x: &Array1<usize>) -> Self {
        Self::Index(x.to_vec())
    }
}

impl From<Range<usize>> for SelectInfoElem {
    fn from(x: Range<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<Range<isize>> for SelectInfoElem {
    fn from(x: Range<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<Range<i32>> for SelectInfoElem {
    fn from(x: Range<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFull> for SelectInfoElem {
    fn from(x: RangeFull) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<usize>> for SelectInfoElem {
    fn from(x: RangeFrom<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<isize>> for SelectInfoElem {
    fn from(x: RangeFrom<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeFrom<i32>> for SelectInfoElem {
    fn from(x: RangeFrom<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<usize>> for SelectInfoElem {
    fn from(x: RangeTo<usize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<isize>> for SelectInfoElem {
    fn from(x: RangeTo<isize>) -> Self {
        Self::Slice(x.into())
    }
}

impl From<RangeTo<i32>> for SelectInfoElem {
    fn from(x: RangeTo<i32>) -> Self {
        Self::Slice(x.into())
    }
}

impl AsRef<SelectInfoElem> for SelectInfoElem {
    fn as_ref(&self) -> &SelectInfoElem {
        self
    }
}

impl SelectInfoElem {
    /// Will panic if the index is out of bounds.
    pub fn bound_check(&self, bound: usize) -> Result<()> {
        match self {
            SelectInfoElem::Index(index) => index.iter().try_for_each(|i|
                if *i >= bound {
                    bail!("index out of bounds: {} >= {}", i, bound)
                } else {
                    Ok(())
                }
            ),
            SelectInfoElem::Slice(slice) => {
                if let Some(end) = slice.end {
                    if end > bound as isize {
                        bail!("slice end out of bounds: {} >= {}", end, bound)
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, SelectInfoElem::Index(_))
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, SelectInfoElem::Slice(_))
    }

    pub fn full() -> Self {
        SelectInfoElem::Slice(Slice {
            start: 0,
            end: None,
            step: 1,
        })
    }

    pub(crate) fn set_axis<'a>(&'a self, axis: usize, ndim: usize, fill: &'a Self) -> SmallVec<[&'a Self; 3]> {
        let mut slice = smallvec![fill; ndim];
        slice[axis] = self;
        slice
    }

    pub fn is_full(&self) -> bool {
        matches!(
            self,
            SelectInfoElem::Slice(Slice {
                start: 0,
                end: None,
                step: 1
            })
        )
    }
}
pub struct BoundedSelectInfo<'a> {
    input_shape: Shape,
    select: Vec<BoundedSelectInfoElem<'a>>,
}

impl<'a> AsRef<[BoundedSelectInfoElem<'a>]> for BoundedSelectInfo<'a> {
    fn as_ref(&self) -> &[BoundedSelectInfoElem<'a>] {
        &self.select
    }
}

impl<'a> TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for BoundedSelectInfo<'a>{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self.select.into_iter().map(|e| match e {
            BoundedSelectInfoElem::Slice(s) => Ok(s.into()),
            _ => bail!("Cannot convert SelectInfo to SliceInfo"),
        }).collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}


impl<'a> BoundedSelectInfo<'a> {
    pub fn new<S, E>(select: &'a S, shape: &Shape) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem> + 'a,
    {
        let res: Vec<_> = select.as_ref().iter().zip(shape.as_ref()).map(|(sel, dim)|
            BoundedSelectInfoElem::new(sel.as_ref(), *dim)
        ).collect();
        Self {
            input_shape: shape.clone(),
            select: res,
        }
    }

    pub fn in_shape(&self) -> Shape {
        self.input_shape.clone()
    }

    pub fn out_shape(&self) -> Shape {
        Shape::from(self.select.iter().map(|x| x.len()).collect::<Vec<_>>())
    }

    pub fn size(&self) -> usize {
        self.out_shape().as_ref().iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.select.len()
    }

    /// Convert to a new slice that contain only the unique indices. A mapping for
    /// getting back the original indices is also returned.
    /*
    pub fn to_unique(&self) -> (Self, Self) {
        let out_shape = self.out_shape();
        let (unique, mapping): (Vec<_>, Vec<_>) = self.select.iter().zip(out_shape.as_ref())
            .map(|(sel, dim)| match sel {
                BoundedSelectInfoElem::Index(x) => {
                    let (unique, mapping) = unique_indices(x, *dim);
                    (unique.into(), mapping.into())
            }
            BoundedSelectInfoElem::Slice(x) => todo!(),
        }).unzip();
        todo!()
    }
    */

    /// Convert into indices. Fail if elements are all slices.
    pub fn try_into_indices(&self) -> Option<Array2<usize>> {
        if self.select.iter().all(|e| matches!(e, BoundedSelectInfoElem::Slice(_))) {
            return None;
        }

        let shape = self.out_shape();
        let ncols = self.ndim();
        let nrows = self.size();

        let mut result: Array2<usize> = Array2::zeros((nrows, ncols));

        for c in 0..ncols {
            let n_repeat = shape.as_ref()[(c+1)..].iter().product();
            match self.select[c] {
                BoundedSelectInfoElem::Index(ref x) => {
                    let mut values = x.iter().flat_map(|x| std::iter::repeat(x).take(n_repeat)).cycle();
                    for r in 0..nrows {
                        result[[r, c]] = *values.next().unwrap();
                    }
                },
                BoundedSelectInfoElem::Slice(BoundedSlice { start, end, step }) => {
                    if step > 0 {
                        let mut values = (start..end).step_by(step as usize).flat_map(|x| std::iter::repeat(x).take(n_repeat)).cycle();
                        for r in 0..nrows {
                            result[[r, c]] = values.next().unwrap();
                        }
                    } else {
                        let mut values = (start..end).step_by(step.abs() as usize).rev().flat_map(|x| std::iter::repeat(x).take(n_repeat)).cycle();
                        for r in 0..nrows {
                            result[[r, c]] = values.next().unwrap();
                        }
                    }
                },
            }
        }
        Some(result)
    }
}

pub enum BoundedSelectInfoElem<'a> {
    Index(&'a [usize]),
    Slice(BoundedSlice),
}

impl<'a> BoundedSelectInfoElem<'a> {
    pub fn new<S: AsRef<SelectInfoElem>>(select: &'a S, bound: usize) -> Self {
        match select.as_ref() {
            SelectInfoElem::Index(idx) => Self::Index(idx.as_slice()),
            SelectInfoElem::Slice(slice) => Self::Slice(BoundedSlice::new(slice, bound)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Index(idx) => idx.len(),
            Self::Slice(slice) => slice.len(),
        }
    }

    pub fn index(&self, i: usize) -> usize {
        match self {
            Self::Index(idx) => idx[i],
            Self::Slice(slice) => slice.index(i),
        }
    }

    pub fn is_full(&self, bound: usize) -> bool {
        match self {
            Self::Slice(slice) => slice.start == 0 && slice.end == bound && slice.step == 1,
            Self::Index(indices) => indices.len() == bound && indices.iter().enumerate().all(|(i, &x)| x == i),
        }
    }

    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Index(idx) => idx.iter().copied().collect(),
            Self::Slice(slice) => if slice.step > 0 {
                (slice.start..slice.end).step_by(slice.step as usize).collect()
            } else {
                (slice.start..slice.end).step_by(slice.step.abs() as usize).rev().collect()
            },
        }
    }

    pub fn iter(&self) -> Box<dyn ExactSizeIterator<Item=usize> + 'a> {
        match self {
            Self::Index(idx) => Box::new(idx.iter().copied()),
            Self::Slice(slice) => if slice.step > 0 {
                Box::new((slice.start..slice.end).step_by(slice.step as usize))
            } else {
                Box::new((slice.start..slice.end).step_by(slice.step.abs() as usize).rev())
            },
        }
    }
}


#[derive(Debug, Copy, Clone)]
pub struct BoundedSlice {
    pub start: usize,
    pub end: usize,
    pub step: isize,
}

impl Into<SliceInfoElem> for BoundedSlice {
    fn into(self) -> SliceInfoElem {
        Slice {
            start: self.start as isize,
            end: Some(self.end as isize),
            step: self.step,
        }.into()
    }
}


impl BoundedSlice {
    pub(crate) fn new(slice: &Slice, bound: usize) -> Self {
        fn convert(x: isize, d: usize) -> usize {
            if x < 0 {
                d.checked_add_signed(x).unwrap()
            } else {
                x as usize
            }
        }

        Self {
            start: convert(slice.start, bound),
            end: slice.end.map_or(bound, |x| convert(x, bound)),
            step: slice.step,
        }
    }

    pub(crate) fn len(&self) -> usize {
        (self.end - self.start).checked_div(self.step.unsigned_abs()).unwrap()
    }

    pub(crate) fn index(&self, i: usize) -> usize {
        if self.step > 0 {
            self.start + i * self.step as usize
        } else {
            self.end.checked_sub(1 + i * (-self.step) as usize).unwrap()
        }
    }
}

pub const SLICE_FULL: Slice = Slice {
    start: 0,
    end: None,
    step: 1,
};

/// find unique indices and return the mapping
/// 
/// Example:
/// 
/// (unique_idx, mapping) = unique_indices_sorted(ori_idx, upper_bound)
/// assert_eq!(ori_idx, mapping.iter().map(|x| unique_idx[*x]).collect::<Vec<usize>>())
pub(crate) fn unique_indices_sorted(indices: &[usize], upper_bound: usize) -> (Vec<usize>, Vec<usize>) {
    let mut mask = vec![upper_bound; upper_bound];
    // Set the mask for the present indices
    for i in indices {
        mask[*i] = *i;
    }
    let unique = mask.iter().filter(|x| **x != upper_bound).map(|x| *x).collect();

    // Find the new order
    mask.iter_mut().fold(0, |acc, x| {
        if *x != upper_bound {
            *x = acc;
            acc + 1
        } else {
            acc
        }
    });

    // Get the mapping
    let mapping = indices.iter().map(|x| mask[*x]).collect();
    (unique, mapping)
}

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices/new-axes, separated by comma,
/// with optional step sizes that are separated from the range by a semicolon.
///
/// # Example
///
/// # Negative *step*
///
/// The behavior of negative *step* arguments is most easily understood with
/// slicing as a two-step process:
///
/// 1. First, perform a slice with *range*.
///
/// 2. If *step* is positive, start with the front of the slice; if *step* is
///    negative, start with the back of the slice. Then, add *step* until
///    reaching the other end of the slice (inclusive).
///
/// An equivalent way to think about step 2 is, "If *step* is negative, reverse
/// the slice. Start at the front of the (possibly reversed) slice, and add
/// *step.abs()* until reaching the back of the slice (inclusive)."
///
/// For example,
///
/// ```
/// # use anndata::s;
/// #
/// # fn main() {
/// println!("{:?}", s![1..3 , ..]);
/// println!("{:?}", s![vec![1, 10, 3], ..]);
/// # }
/// ```
#[macro_export]
macro_rules! s{
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($crate::data::SelectInfoElem::from($x));
            )*
            $crate::data::SelectInfo(temp_vec)
        }

    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_indices(input: Vec<u16>) {
            let max = (*input.iter().max().unwrap_or(&0) as usize) + 1;
            let indices = input.into_iter().map(|x| x as usize).collect::<Vec<_>>();
            let sorted_expected = indices.iter().map(|x| *x).unique().sorted().collect::<Vec<_>>();
            let (sorted, mapping) = unique_indices_sorted(indices.as_slice(), max);
            assert_eq!(sorted, sorted_expected);
            assert_eq!(indices, mapping.iter().map(|x| sorted[*x]).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_basic() {
        assert_eq!(
            unique_indices_sorted(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11),
            (
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
        );
    }
}
