pub trait VecExtensions<T> {
    fn head(&self, count: usize) -> &[T];
    fn tail(&self, count: usize) -> &[T];
}

impl<T> VecExtensions<T> for Vec<T> {
    fn head(&self, count: usize) -> &[T] {
        &self[..count.min(self.len())]
    }

    fn tail(&self, count: usize) -> &[T] {
        &self[self.len().saturating_sub(count)..]
    }
}
