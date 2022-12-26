#[derive(Clone, Debug, PartialEq)]
pub struct NamedRange {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl Iterator for NamedRange {
    type Item = (String, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let end = (self.start + self.step).min(self.end);
            let item = (self.name.clone(), self.start, end);
            self.start = end;
            Some(item)
        }
    }
}