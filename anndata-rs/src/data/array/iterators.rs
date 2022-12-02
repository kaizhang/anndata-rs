pub struct CsrIterator<I> {
    iterator: I,
    num_cols: usize,
}

impl<I> CsrIterator<I> {
    pub fn new(iterator: I, num_cols: usize) -> Self {
        Self { iterator, num_cols }
    }
}