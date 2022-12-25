#[derive(Clone, Debug, PartialEq)]
pub struct NamedRange {
    pub name: Option<String>,
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl NamedRange {
    pub fn to_intervals(&self) -> impl Iterator<Item = usize> {
        (self.start..self.end).step_by(self.step)
    }
}