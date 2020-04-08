pub trait SliceStr<'a, T> {
    fn find(&'a self, needle: &'a [T]) -> Option<usize>;
    fn rfind(&'a self, needle: &'a [T]) -> Option<usize>;
}

impl<'a, T: 'a> SliceStr<'a, T> for [T]
where
    &'a [T]: PartialEq,
{
    fn find(&'a self, needle: &'a [T]) -> Option<usize> {
        for (i, window) in self.windows(needle.len()).enumerate() {
            if window == needle {
                return Some(i);
            }
        }
        None
    }

    fn rfind(&'a self, needle: &'a [T]) -> Option<usize> {
        for (i, window) in self.windows(needle.len()).enumerate().rev() {
            if window == needle {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::SliceStr;

    #[test]
    fn find() {
        let haystack = b"1234512345";
        assert!(haystack[..].find(b"x") == None);
        assert!(haystack[..].find(b"34") == Some(2));
        assert!(haystack[3..].find(b"34") == Some(4));
    }

    #[test]
    fn rfind() {
        let haystack = b"1234512345";
        assert!(haystack[..].rfind(b"x") == None);
        assert!(haystack[..].rfind(b"34") == Some(7));
        assert!(haystack[3..].rfind(b"34") == Some(4));
    }
}
