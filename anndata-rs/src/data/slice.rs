use crate::data::Shape;

use ndarray::{Array2, SliceInfoElem, SliceInfo, IxDyn};
use anyhow::{bail, Result};
use itertools::Itertools;

/// A multi-dimensional selection used for reading and writing to a Container.
#[derive(Debug)]
pub struct SelectInfo(Vec<SelectInfoElem>);

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

impl FromIterator<SliceInfoElem> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = SliceInfoElem>>(iter: T) -> Self {
        Self(iter.into_iter().map(SelectInfoElem::Slice).collect())
    }
}

impl<'a> FromIterator<&'a SliceInfoElem> for SelectInfo {
    fn from_iter<T: IntoIterator<Item = &'a SliceInfoElem>>(iter: T) -> Self {
        Self(iter.into_iter().map(|x| SelectInfoElem::Slice(x.clone())).collect())
    }
}


impl TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for SelectInfo {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self.0.into_iter().map(|e| match e {
            SelectInfoElem::Slice(s) => Ok(s),
            _ => bail!("Cannot convert SelectInfo to SliceInfo"),
        }).collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}

impl SelectInfo {
    pub fn all() -> Self {
        Self(vec![SelectInfoElem::Slice(SLICE_FULL)])
    }
}



/// A selection used for reading and writing to a Container.
#[derive(Debug, Clone)]
pub enum SelectInfoElem {
    Index(Vec<usize>),
    Slice(SliceInfoElem),
}

impl AsRef<SelectInfoElem> for SelectInfoElem {
    fn as_ref(&self) -> &SelectInfoElem {
        self
    }
}

impl SelectInfoElem {
    pub fn is_index(&self) -> bool {
        matches!(self, SelectInfoElem::Index(_))
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, SelectInfoElem::Slice(_))
    }

    pub fn is_full(&self) -> bool {
        matches!(
            self,
            SelectInfoElem::Slice(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1
            })
        )
    }
}
pub struct BoundedSelectInfo<'a>(Vec<BoundedSelectInfoElem<'a>>);

impl<'a> TryInto<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> for BoundedSelectInfo<'a>{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
        let elems: Result<Vec<_>> = self.0.into_iter().map(|e| match e {
            BoundedSelectInfoElem::Slice(s) => Ok(s.into()),
            _ => bail!("Cannot convert SelectInfo to SliceInfo"),
        }).collect();
        let slice = SliceInfo::try_from(elems?)?;
        Ok(slice)
    }
}


impl<'a> BoundedSelectInfo<'a> {
    pub fn new<S, E>(select: &'a S, shape: &Shape) -> Option<Self>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem> + 'a,
    {
        let res: Option<Vec<_>> = select.as_ref().iter().zip(shape.as_ref()).map(|(sel, dim)|
            BoundedSelectInfoElem::new(sel.as_ref(), *dim)
        ).collect();
        res.map(Self)
    }

    pub fn shape(&self) -> Shape {
        Shape::from(self.0.iter().map(|x| x.len()).collect::<Vec<_>>())
    }

    pub fn size(&self) -> usize {
        self.shape().as_ref().iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Convert into indices. Fail if elements are all slices.
    pub fn try_into_indices(&self) -> Option<Array2<usize>> {
        if self.0.iter().all(|e| matches!(e, BoundedSelectInfoElem::Slice(_))) {
            return None;
        }

        fn slice_to_vec(start: usize, end: usize, step: isize) -> Vec<usize> {
            if step > 0 {
                (start..end).step_by(step as usize).collect()
            } else {
                (start..end).step_by(-step as usize).rev().collect()
            }
        }

        // TODO: pre-allocate the array to improve performance
        let vec: Vec<_> = self.0.iter().map(|sel| match sel {
            BoundedSelectInfoElem::Slice(BoundedSliceInfoElem { start, end, step }) => 
                slice_to_vec(*start, *end, *step),
            BoundedSelectInfoElem::Index(indices) => indices.iter().map(|x| *x).collect(),
        }).multi_cartesian_product().flatten().collect();
        let indices = Array2::from_shape_vec((vec.len() / self.ndim(), self.ndim()), vec).unwrap();
        Some(indices)
    }
}

pub enum BoundedSelectInfoElem<'a> {
    Index(&'a [usize]),
    Slice(BoundedSliceInfoElem),
}

impl<'a> BoundedSelectInfoElem<'a> {
    pub fn new<S: AsRef<SelectInfoElem>>(select: &'a S, bound: usize) -> Option<Self> {
        let res = match select.as_ref() {
            SelectInfoElem::Index(idx) => Self::Index(idx.as_slice()),
            SelectInfoElem::Slice(slice) => Self::Slice(BoundedSliceInfoElem::new(slice, bound)?),
        };
        Some(res)
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
}



#[derive(Debug, Copy, Clone)]
pub struct BoundedSliceInfoElem {
    start: usize,
    end: usize,
    step: isize,
}

impl Into<SliceInfoElem> for BoundedSliceInfoElem {
    fn into(self) -> SliceInfoElem {
        ndarray::Slice {
            start: self.start as isize,
            end: Some(self.end as isize),
            step: self.step,
        }.into()
    }
}



impl BoundedSliceInfoElem {
    pub(crate) fn new(slice: &SliceInfoElem, bound: usize) -> Option<Self> {
        fn convert(x: isize, d: usize) -> usize {
            if x < 0 {
                d.checked_add_signed(x).unwrap()
            } else {
                x as usize
            }
        }
        match slice {
            SliceInfoElem::Index(x) => Some(Self {
                start: *x as usize,
                end: *x as usize + 1,
                step: 1,
            }),
            SliceInfoElem::Slice { start, end, step } => Some(Self {
                start: convert(*start, bound),
                end: end.map_or(bound, |x| convert(x, bound)),
                step: *step,
            }),
            SliceInfoElem::NewAxis => None,
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

pub const SLICE_FULL: SliceInfoElem = SliceInfoElem::Slice {
    start: 0,
    end: None,
    step: 1,
};


/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices/new-axes, separated by comma,
/// with optional step sizes that are separated from the range by a semicolon.
/// It is converted into a [`SliceInfo`] instance.
///
/// Each range/slice/index uses signed indices, where a negative value is
/// counted from the end of the axis. Step sizes are also signed and may be
/// negative, but must not be zero.
///
/// The syntax is `s![` *[ elem [, elem [ , ... ] ] ]* `]`, where *elem* is any
/// of the following:
///
/// * *index*: an index to use for taking a subview with respect to that axis.
///   (The index is selected. The axis is removed except with
///   [`.slice_collapse()`].)
/// * *range*: a range with step size 1 to use for slicing that axis.
/// * *range* `;` *step*: a range with step size *step* to use for slicing that axis.
/// * *slice*: a [`Slice`] instance to use for slicing that axis.
/// * *slice* `;` *step*: a range constructed from a [`Slice`] instance,
///   multiplying the step size by *step*, to use for slicing that axis.
/// * *new-axis*: a [`NewAxis`] instance that represents the creation of a new axis.
///   (Except for [`.slice_collapse()`], which panics on [`NewAxis`] elements.)
///
/// The number of *elem*, not including *new-axis*, must match the
/// number of axes in the array. *index*, *range*, *slice*, *step*, and
/// *new-axis* can be expressions. *index* must be of type `isize`, `usize`, or
/// `i32`. *range* must be of type `Range<I>`, `RangeTo<I>`, `RangeFrom<I>`, or
/// `RangeFull` where `I` is `isize`, `usize`, or `i32`. *step* must be a type
/// that can be converted to `isize` with the `as` keyword.
///
/// For example, `s![0..4;2, 6, 1..5, NewAxis]` is a slice of the first axis
/// for 0..4 with step size 2, a subview of the second axis at index 6, a slice
/// of the third axis for 1..5 with default step size 1, and a new axis of
/// length 1 at the end of the shape. The input array must have 3 dimensions.
/// The resulting slice would have shape `[2, 4, 1]` for [`.slice()`],
/// [`.slice_mut()`], and [`.slice_move()`], while [`.slice_collapse()`] would
/// panic. Without the `NewAxis`, i.e. `s![0..4;2, 6, 1..5]`,
/// [`.slice_collapse()`] would result in an array of shape `[2, 1, 4]`.
///
/// [`.slice()`]: crate::ArrayBase::slice
/// [`.slice_mut()`]: crate::ArrayBase::slice_mut
/// [`.slice_move()`]: crate::ArrayBase::slice_move
/// [`.slice_collapse()`]: crate::ArrayBase::slice_collapse
///
/// See also [*Slicing*](crate::ArrayBase#slicing).
///
/// # Example
///
/// ```
/// use ndarray::{s, Array2, ArrayView2};
///
/// fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
///     -4. * &v.slice(s![1..-1, 1..-1])
///     + v.slice(s![ ..-2, 1..-1])
///     + v.slice(s![1..-1,  ..-2])
///     + v.slice(s![1..-1, 2..  ])
///     + v.slice(s![2..  , 1..-1])
/// }
/// # fn main() { let _ = laplacian; }
/// ```
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
/// # use ndarray::prelude::*;
/// #
/// # fn main() {
/// let arr = array![0, 1, 2, 3];
/// assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
/// assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! s(
    // convert a..b;c into @convert(a..b, c), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextDim::next_in_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextDim::next_out_dim(&r, $out_dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* $crate::s!(@convert r, $s)],
                        in_dim,
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b into @convert(a..b), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextDim::next_in_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextDim::next_out_dim(&r, $out_dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* $crate::s!(@convert r)],
                        in_dim,
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b;c into @convert(a..b, c), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r;$s]
    };
    // convert a..b into @convert(a..b), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r]
    };
    // convert a..b;c into @convert(a..b, c)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextDim::next_in_dim(&r, $in_dim),
                   $crate::SliceNextDim::next_out_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r, $s),]
                   $($t)*
                ]
            }
        }
    };
    // convert a..b into @convert(a..b)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextDim::next_in_dim(&r, $in_dim),
                   $crate::SliceNextDim::next_out_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r),]
                   $($t)*
                ]
            }
        }
    };
    // empty call, i.e. `s![]`
    (@parse ::core::marker::PhantomData::<$crate::Ix0>, ::core::marker::PhantomData::<$crate::Ix0>, []) => {
        {
            #[allow(unsafe_code)]
            unsafe {
                $crate::SliceInfo::new_unchecked(
                    [],
                    ::core::marker::PhantomData::<$crate::Ix0>,
                    ::core::marker::PhantomData::<$crate::Ix0>,
                )
            }
        }
    };
    // Catch-all clause for syntax errors
    (@parse $($t:tt)*) => { compile_error!("Invalid syntax in s![] call.") };
    // convert range/index/new-axis into SliceInfoElem
    (@convert $r:expr) => {
        <$crate::SliceInfoElem as ::core::convert::From<_>>::from($r)
    };
    // convert range/index/new-axis and step into SliceInfoElem
    (@convert $r:expr, $s:expr) => {
        <$crate::SliceInfoElem as ::core::convert::From<_>>::from(
            <$crate::Slice as ::core::convert::From<_>>::from($r).step_by($s as isize)
        )
    };
    ($($t:tt)*) => {
        $crate::s![@parse
              ::core::marker::PhantomData::<$crate::Ix0>,
              ::core::marker::PhantomData::<$crate::Ix0>,
              []
              $($t)*
        ]
    };
);