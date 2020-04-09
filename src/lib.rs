//! String-like slice manipulation.

use self::pattern::Pattern;
use self::pattern::{DoubleEndedSearcher, ReverseSearcher, SearchStep, Searcher};

use std::fmt::{self, Write};
use std::iter::{Chain, FlatMap, Flatten};
use std::iter::{Cloned, Filter, FusedIterator, Map};
use std::mem;
use std::option;
use std::slice::{self, SliceIndex, Split as SliceSplit};

pub mod pattern;

/// This macro generates a Clone impl for string pattern API
/// wrapper types of the form X<'a, P>
macro_rules! derive_pattern_clone {
    (clone $t:ident with |$s:ident| $e:expr) => {
        impl<'a, T: Eq, P> Clone for $t<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: Clone,
        {
            fn clone(&self) -> Self {
                let $s = self;
                $e
            }
        }
    };
}

/// This macro generates two public iterator structs
/// wrapping a private internal one that makes use of the `Pattern` API.
///
/// For all patterns `P: Pattern<'a>` the following items will be
/// generated (generics omitted):
///
/// struct $forward_iterator($internal_iterator);
/// struct $reverse_iterator($internal_iterator);
///
/// impl Iterator for $forward_iterator
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// impl DoubleEndedIterator for $forward_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl Iterator for $reverse_iterator
///       where P::Searcher: ReverseSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl DoubleEndedIterator for $reverse_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// The internal one is defined outside the macro, and has almost the same
/// semantic as a DoubleEndedIterator by delegating to `pattern::Searcher` and
/// `pattern::ReverseSearcher` for both forward and reverse iteration.
///
/// "Almost", because a `Searcher` and a `ReverseSearcher` for a given
/// `Pattern` might not return the same elements, so actually implementing
/// `DoubleEndedIterator` for it would be incorrect.
/// (See the docs in `str::pattern` for more details)
///
/// However, the internal struct still represents a single ended iterator from
/// either end, and depending on pattern is also a valid double ended iterator,
/// so the two wrapper structs implement `Iterator`
/// and `DoubleEndedIterator` depending on the concrete pattern type, leading
/// to the complex impls seen above.
macro_rules! generate_pattern_iterators {
    {
        // Forward iterator
        forward:
            $(#[$forward_iterator_attribute:meta])*
            struct $forward_iterator:ident;

        // Reverse iterator
        reverse:
            $(#[$reverse_iterator_attribute:meta])*
            struct $reverse_iterator:ident;

        // Internal almost-iterator that is being delegated to
        internal:
            $internal_iterator:ident yielding ($iterty:ty);

        // Kind of delegation - either single ended or double ended
        delegate $($t:tt)*
    } => {
        $(#[$forward_iterator_attribute])*
        pub struct $forward_iterator<'a, T: Eq, P: Pattern<'a, T>>($internal_iterator<'a, T, P>);

        impl<'a, T: Eq, P> fmt::Debug for $forward_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: fmt::Debug
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($forward_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        impl<'a, T: 'a + Eq, P: Pattern<'a, T>> Iterator for $forward_iterator<'a, T, P> {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }

        impl<'a, T: Eq, P> Clone for $forward_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: Clone
        {
            fn clone(&self) -> Self {
                $forward_iterator(self.0.clone())
            }
        }

        $(#[$reverse_iterator_attribute])*
        pub struct $reverse_iterator<'a, T: Eq, P: Pattern<'a, T>>($internal_iterator<'a, T, P>);

        impl<'a, T: Eq, P> fmt::Debug for $reverse_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: fmt::Debug
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($reverse_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        impl<'a, T: 'a + Eq, P> Iterator for $reverse_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: ReverseSearcher<'a, T>
        {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        impl<'a, T: 'a + Eq, P> Clone for $reverse_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: Clone
        {
            fn clone(&self) -> Self {
                $reverse_iterator(self.0.clone())
            }
        }

        impl<'a, T: 'a + Eq, P: Pattern<'a, T>> FusedIterator for $forward_iterator<'a, T, P> {}

        impl<'a, T: 'a + Eq, P> FusedIterator for $reverse_iterator<'a,T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: ReverseSearcher<'a, T>
        {}

        generate_pattern_iterators!($($t)* with ,
                                                $forward_iterator,
                                                $reverse_iterator, $iterty);
    };
    {
        double ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {
        $(#[$common_stability_attribute])*
        impl<'a, T: 'a + Eq, P> DoubleEndedIterator for $forward_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: DoubleEndedSearcher<'a, T>,
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, T: 'a + Eq, P> DoubleEndedIterator for $reverse_iterator<'a, T, P>
        where
            P: Pattern<'a, T>,
            P::Searcher: DoubleEndedSearcher<'a, T>
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }
    };
    {
        single ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {}
}

derive_pattern_clone! {
    clone MatchesInternal
    with |s| MatchesInternal(s.0.clone())
}

// struct SplitInternal<'a, P: Pattern<'a>> {
//     start: usize,
//     end: usize,
//     matcher: P::Searcher,
//     allow_trailing_empty: bool,
//     finished: bool,
// }

// impl<'a, P> fmt::Debug for SplitInternal<'a, P>
// where
//     P: Pattern<'a, Searcher: fmt::Debug>,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("SplitInternal")
//             .field("start", &self.start)
//             .field("end", &self.end)
//             .field("matcher", &self.matcher)
//             .field("allow_trailing_empty", &self.allow_trailing_empty)
//             .field("finished", &self.finished)
//             .finish()
//     }
// }

// impl<'a, P: Pattern<'a>> SplitInternal<'a, P> {
//     #[inline]
//     fn get_end(&mut self) -> Option<&'a str> {
//         if !self.finished && (self.allow_trailing_empty || self.end - self.start > 0) {
//             self.finished = true;
//             // SAFETY: `self.start` and `self.end` always lie on unicode boundaries.
//             unsafe {
//                 let string = self.matcher.haystack().get_unchecked(self.start..self.end);
//                 Some(string)
//             }
//         } else {
//             None
//         }
//     }

//     #[inline]
//     fn next(&mut self) -> Option<&'a str> {
//         if self.finished {
//             return None;
//         }

//         let haystack = self.matcher.haystack();
//         match self.matcher.next_match() {
//             // SAFETY: `Searcher` guarantees that `a` and `b` lie on unicode boundaries.
//             Some((a, b)) => unsafe {
//                 let elt = haystack.get_unchecked(self.start..a);
//                 self.start = b;
//                 Some(elt)
//             },
//             None => self.get_end(),
//         }
//     }

//     #[inline]
//     fn next_back(&mut self) -> Option<&'a str>
//     where
//         P::Searcher: ReverseSearcher<'a>,
//     {
//         if self.finished {
//             return None;
//         }

//         if !self.allow_trailing_empty {
//             self.allow_trailing_empty = true;
//             match self.next_back() {
//                 Some(elt) if !elt.is_empty() => return Some(elt),
//                 _ => {
//                     if self.finished {
//                         return None;
//                     }
//                 }
//             }
//         }

//         let haystack = self.matcher.haystack();
//         match self.matcher.next_match_back() {
//             // SAFETY: `Searcher` guarantees that `a` and `b` lie on unicode boundaries.
//             Some((a, b)) => unsafe {
//                 let elt = haystack.get_unchecked(b..self.end);
//                 self.end = a;
//                 Some(elt)
//             },
//             // SAFETY: `self.start` and `self.end` always lie on unicode boundaries.
//             None => unsafe {
//                 self.finished = true;
//                 Some(haystack.get_unchecked(self.start..self.end))
//             },
//         }
//     }
// }

// generate_pattern_iterators! {
//     forward:
//         /// Created with the method [`split`].
//         ///
//         /// [`split`]: ../../std/primitive.str.html#method.split
//         struct Split;
//     reverse:
//         /// Created with the method [`rsplit`].
//         ///
//         /// [`rsplit`]: ../../std/primitive.str.html#method.rsplit
//         struct RSplit;
//     stability:
//         #[stable(feature = "rust1", since = "1.0.0")]
//     internal:
//         SplitInternal yielding (&'a str);
//     delegate double ended;
// }

// generate_pattern_iterators! {
//     forward:
//         /// Created with the method [`split_terminator`].
//         ///
//         /// [`split_terminator`]: ../../std/primitive.str.html#method.split_terminator
//         struct SplitTerminator;
//     reverse:
//         /// Created with the method [`rsplit_terminator`].
//         ///
//         /// [`rsplit_terminator`]: ../../std/primitive.str.html#method.rsplit_terminator
//         struct RSplitTerminator;
//     stability:
//         #[stable(feature = "rust1", since = "1.0.0")]
//     internal:
//         SplitInternal yielding (&'a str);
//     delegate double ended;
// }

// derive_pattern_clone! {
//     clone SplitNInternal
//     with |s| SplitNInternal { iter: s.iter.clone(), ..*s }
// }

// struct SplitNInternal<'a, P: Pattern<'a>> {
//     iter: SplitInternal<'a, P>,
//     /// The number of splits remaining
//     count: usize,
// }

// impl<'a, P> fmt::Debug for SplitNInternal<'a, P>
// where
//     P: Pattern<'a, Searcher: fmt::Debug>,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("SplitNInternal")
//             .field("iter", &self.iter)
//             .field("count", &self.count)
//             .finish()
//     }
// }

// impl<'a, P: Pattern<'a>> SplitNInternal<'a, P> {
//     #[inline]
//     fn next(&mut self) -> Option<&'a str> {
//         match self.count {
//             0 => None,
//             1 => {
//                 self.count = 0;
//                 self.iter.get_end()
//             }
//             _ => {
//                 self.count -= 1;
//                 self.iter.next()
//             }
//         }
//     }

//     #[inline]
//     fn next_back(&mut self) -> Option<&'a str>
//     where
//         P::Searcher: ReverseSearcher<'a>,
//     {
//         match self.count {
//             0 => None,
//             1 => {
//                 self.count = 0;
//                 self.iter.get_end()
//             }
//             _ => {
//                 self.count -= 1;
//                 self.iter.next_back()
//             }
//         }
//     }
// }

// generate_pattern_iterators! {
//     forward:
//         /// Created with the method [`splitn`].
//         ///
//         /// [`splitn`]: ../../std/primitive.str.html#method.splitn
//         struct SplitN;
//     reverse:
//         /// Created with the method [`rsplitn`].
//         ///
//         /// [`rsplitn`]: ../../std/primitive.str.html#method.rsplitn
//         struct RSplitN;
//     stability:
//         #[stable(feature = "rust1", since = "1.0.0")]
//     internal:
//         SplitNInternal yielding (&'a str);
//     delegate single ended;
// }

// derive_pattern_clone! {
//     clone MatchIndicesInternal
//     with |s| MatchIndicesInternal(s.0.clone())
// }

// struct MatchIndicesInternal<'a, P: Pattern<'a>>(P::Searcher);

// impl<'a, P> fmt::Debug for MatchIndicesInternal<'a, P>
// where
//     P: Pattern<'a, Searcher: fmt::Debug>,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_tuple("MatchIndicesInternal").field(&self.0).finish()
//     }
// }

// impl<'a, P: Pattern<'a>> MatchIndicesInternal<'a, P> {
//     #[inline]
//     fn next(&mut self) -> Option<(usize, &'a str)> {
//         self.0
//             .next_match()
//             // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
//             .map(|(start, end)| unsafe { (start, self.0.haystack().get_unchecked(start..end)) })
//     }

//     #[inline]
//     fn next_back(&mut self) -> Option<(usize, &'a str)>
//     where
//         P::Searcher: ReverseSearcher<'a>,
//     {
//         self.0
//             .next_match_back()
//             // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
//             .map(|(start, end)| unsafe { (start, self.0.haystack().get_unchecked(start..end)) })
//     }
// }

// generate_pattern_iterators! {
//     forward:
//         /// Created with the method [`match_indices`].
//         ///
//         /// [`match_indices`]: ../../std/primitive.str.html#method.match_indices
//         struct MatchIndices;
//     reverse:
//         /// Created with the method [`rmatch_indices`].
//         ///
//         /// [`rmatch_indices`]: ../../std/primitive.str.html#method.rmatch_indices
//         struct RMatchIndices;
//     stability:
//         #[stable(feature = "str_match_indices", since = "1.5.0")]
//     internal:
//         MatchIndicesInternal yielding ((usize, &'a str));
//     delegate double ended;
// }

pub struct MatchesInternal<'a, T: Eq, P: Pattern<'a, T>>(P::Searcher);

impl<'a, T: Eq, P> std::fmt::Debug for MatchesInternal<'a, T, P>
where
    P: Pattern<'a, T>,
    P::Searcher: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("MatchesInternal").field(&self.0).finish()
    }
}

impl<'a, T: Eq, P: Pattern<'a, T>> MatchesInternal<'a, T, P> {
    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        self.0.next_match().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().get_unchecked(a..b)
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]>
    where
        P::Searcher: ReverseSearcher<'a, T>,
    {
        // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
        self.0.next_match_back().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().get_unchecked(a..b)
        })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`matches`].
        ///
        /// [`matches`]: ../../std/primitive.str.html#method.matches
        struct Matches;
    reverse:
        /// Created with the method [`rmatches`].
        ///
        /// [`rmatches`]: ../../std/primitive.str.html#method.rmatches
        struct RMatches;
    internal:
        MatchesInternal yielding (&'a [T]);
    delegate double ended;
}

// /// An iterator over the lines of a string, as string slices.
// ///
// /// This struct is created with the [`lines`] method on [`str`].
// /// See its documentation for more.
// ///
// /// [`lines`]: ../../std/primitive.str.html#method.lines
// /// [`str`]: ../../std/primitive.str.html
// #[stable(feature = "rust1", since = "1.0.0")]
// #[derive(Clone, Debug)]
// pub struct Lines<'a>(Map<SplitTerminator<'a, char>, LinesAnyMap>);

// #[stable(feature = "rust1", since = "1.0.0")]
// impl<'a> Iterator for Lines<'a> {
//     type Item = &'a str;

//     #[inline]
//     fn next(&mut self) -> Option<&'a str> {
//         self.0.next()
//     }

//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.0.size_hint()
//     }

//     #[inline]
//     fn last(mut self) -> Option<&'a str> {
//         self.next_back()
//     }
// }

// #[stable(feature = "rust1", since = "1.0.0")]
// impl<'a> DoubleEndedIterator for Lines<'a> {
//     #[inline]
//     fn next_back(&mut self) -> Option<&'a str> {
//         self.0.next_back()
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl FusedIterator for Lines<'_> {}

// impl_fn_for_zst! {
//     /// A nameable, cloneable fn type
//     #[derive(Clone)]
//     struct LinesAnyMap impl<'a> Fn = |line: &'a str| -> &'a str {
//         let l = line.len();
//         if l > 0 && line.as_bytes()[l - 1] == b'\r' { &line[0 .. l - 1] }
//         else { line }
//     };
// }

// /*
// Section: UTF-8 validation
// */
// // use truncation to fit u64 into usize
// const NONASCII_MASK: usize = 0x80808080_80808080u64 as usize;

// /// Returns `true` if any byte in the word `x` is nonascii (>= 128).
// #[inline]
// fn contains_nonascii(x: usize) -> bool {
//     (x & NONASCII_MASK) != 0
// }

// /// Walks through `v` checking that it's a valid UTF-8 sequence,
// /// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
// #[inline]
// fn run_utf8_validation(v: &[u8]) -> Result<(), Utf8Error> {
//     let mut index = 0;
//     let len = v.len();

//     let usize_bytes = mem::size_of::<usize>();
//     let ascii_block_size = 2 * usize_bytes;
//     let blocks_end = if len >= ascii_block_size { len - ascii_block_size + 1 } else { 0 };
//     let align = v.as_ptr().align_offset(usize_bytes);

//     while index < len {
//         let old_offset = index;
//         macro_rules! err {
//             ($error_len: expr) => {
//                 return Err(Utf8Error { valid_up_to: old_offset, error_len: $error_len });
//             };
//         }

//         macro_rules! next {
//             () => {{
//                 index += 1;
//                 // we needed data, but there was none: error!
//                 if index >= len {
//                     err!(None)
//                 }
//                 v[index]
//             }};
//         }

//         let first = v[index];
//         if first >= 128 {
//             let w = UTF8_CHAR_WIDTH[first as usize];
//             // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
//             //        first  C2 80        last DF BF
//             // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
//             //        first  E0 A0 80     last EF BF BF
//             //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
//             //               ED A0 80 to       ED BF BF
//             // 4-byte encoding is for codepoints \u{1000}0 to \u{10ff}ff
//             //        first  F0 90 80 80  last F4 8F BF BF
//             //
//             // Use the UTF-8 syntax from the RFC
//             //
//             // https://tools.ietf.org/html/rfc3629
//             // UTF8-1      = %x00-7F
//             // UTF8-2      = %xC2-DF UTF8-tail
//             // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
//             //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
//             // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
//             //               %xF4 %x80-8F 2( UTF8-tail )
//             match w {
//                 2 => {
//                     if next!() & !CONT_MASK != TAG_CONT_U8 {
//                         err!(Some(1))
//                     }
//                 }
//                 3 => {
//                     match (first, next!()) {
//                         (0xE0, 0xA0..=0xBF)
//                         | (0xE1..=0xEC, 0x80..=0xBF)
//                         | (0xED, 0x80..=0x9F)
//                         | (0xEE..=0xEF, 0x80..=0xBF) => {}
//                         _ => err!(Some(1)),
//                     }
//                     if next!() & !CONT_MASK != TAG_CONT_U8 {
//                         err!(Some(2))
//                     }
//                 }
//                 4 => {
//                     match (first, next!()) {
//                         (0xF0, 0x90..=0xBF) | (0xF1..=0xF3, 0x80..=0xBF) | (0xF4, 0x80..=0x8F) => {}
//                         _ => err!(Some(1)),
//                     }
//                     if next!() & !CONT_MASK != TAG_CONT_U8 {
//                         err!(Some(2))
//                     }
//                     if next!() & !CONT_MASK != TAG_CONT_U8 {
//                         err!(Some(3))
//                     }
//                 }
//                 _ => err!(Some(1)),
//             }
//             index += 1;
//         } else {
//             // Ascii case, try to skip forward quickly.
//             // When the pointer is aligned, read 2 words of data per iteration
//             // until we find a word containing a non-ascii byte.
//             if align != usize::max_value() && align.wrapping_sub(index) % usize_bytes == 0 {
//                 let ptr = v.as_ptr();
//                 while index < blocks_end {
//                     // SAFETY: since `align - index` and `ascii_block_size` are
//                     // multiples of `usize_bytes`, `block = ptr.add(index)` is
//                     // always aligned with a `usize` so it's safe to dereference
//                     // both `block` and `block.offset(1)`.
//                     unsafe {
//                         let block = ptr.add(index) as *const usize;
//                         // break if there is a nonascii byte
//                         let zu = contains_nonascii(*block);
//                         let zv = contains_nonascii(*block.offset(1));
//                         if zu | zv {
//                             break;
//                         }
//                     }
//                     index += ascii_block_size;
//                 }
//                 // step from the point where the wordwise loop stopped
//                 while index < len && v[index] < 128 {
//                     index += 1;
//                 }
//             } else {
//                 index += 1;
//             }
//         }
//     }

//     Ok(())
// }

// // https://tools.ietf.org/html/rfc3629
// static UTF8_CHAR_WIDTH: [u8; 256] = [
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, // 0x1F
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, // 0x3F
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, // 0x5F
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, // 0x7F
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, // 0x9F
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, // 0xBF
//     0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//     2, // 0xDF
//     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // 0xEF
//     4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0xFF
// ];

// /// Given a first byte, determines how many bytes are in this UTF-8 character.
// #[unstable(feature = "str_internals", issue = "none")]
// #[inline]
// pub fn utf8_char_width(b: u8) -> usize {
//     UTF8_CHAR_WIDTH[b as usize] as usize
// }

// /// Mask of the value bits of a continuation byte.
// const CONT_MASK: u8 = 0b0011_1111;
// /// Value of the tag bits (tag mask is !CONT_MASK) of a continuation byte.
// const TAG_CONT_U8: u8 = 0b1000_0000;

// /*
// Section: Trait implementations
// */
// mod traits {
//     use crate::cmp::Ordering;
//     use crate::ops;
//     use crate::slice::{self, SliceIndex};

//     /// Implements ordering of strings.
//     ///
//     /// Strings are ordered  lexicographically by their byte values. This orders Unicode code
//     /// points based on their positions in the code charts. This is not necessarily the same as
//     /// "alphabetical" order, which varies by language and locale. Sorting strings according to
//     /// culturally-accepted standards requires locale-specific data that is outside the scope of
//     /// the `str` type.
//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl Ord for str {
//         #[inline]
//         fn cmp(&self, other: &str) -> Ordering {
//             self.as_bytes().cmp(other.as_bytes())
//         }
//     }

//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl PartialEq for str {
//         #[inline]
//         fn eq(&self, other: &str) -> bool {
//             self.as_bytes() == other.as_bytes()
//         }
//         #[inline]
//         fn ne(&self, other: &str) -> bool {
//             !(*self).eq(other)
//         }
//     }

//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl Eq for str {}

//     /// Implements comparison operations on strings.
//     ///
//     /// Strings are compared lexicographically by their byte values. This compares Unicode code
//     /// points based on their positions in the code charts. This is not necessarily the same as
//     /// "alphabetical" order, which varies by language and locale. Comparing strings according to
//     /// culturally-accepted standards requires locale-specific data that is outside the scope of
//     /// the `str` type.
//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl PartialOrd for str {
//         #[inline]
//         fn partial_cmp(&self, other: &str) -> Option<Ordering> {
//             Some(self.cmp(other))
//         }
//     }

//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl<I> ops::Index<I> for str
//     where
//         I: SliceIndex<str>,
//     {
//         type Output = I::Output;

//         #[inline]
//         fn index(&self, index: I) -> &I::Output {
//             index.index(self)
//         }
//     }

//     #[stable(feature = "rust1", since = "1.0.0")]
//     impl<I> ops::IndexMut<I> for str
//     where
//         I: SliceIndex<str>,
//     {
//         #[inline]
//         fn index_mut(&mut self, index: I) -> &mut I::Output {
//             index.index_mut(self)
//         }
//     }

//     #[inline(never)]
//     #[cold]
//     fn str_index_overflow_fail() -> ! {
//         panic!("attempted to index str up to maximum usize");
//     }

//     /// Implements substring slicing with syntax `&self[..]` or `&mut self[..]`.
//     ///
//     /// Returns a slice of the whole string, i.e., returns `&self` or `&mut
//     /// self`. Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`. Unlike
//     /// other indexing operations, this can never panic.
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// Prior to 1.20.0, these indexing operations were still supported by
//     /// direct implementation of `Index` and `IndexMut`.
//     ///
//     /// Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`.
//     #[stable(feature = "str_checked_slicing", since = "1.20.0")]
//     impl SliceIndex<str> for ops::RangeFull {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             Some(slice)
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             Some(slice)
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             slice
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             slice
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             slice
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             slice
//         }
//     }

//     /// Implements substring slicing with syntax `&self[begin .. end]` or `&mut
//     /// self[begin .. end]`.
//     ///
//     /// Returns a slice of the given string from the byte range
//     /// [`begin`, `end`).
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// Prior to 1.20.0, these indexing operations were still supported by
//     /// direct implementation of `Index` and `IndexMut`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `begin` or `end` does not point to the starting byte offset of
//     /// a character (as defined by `is_char_boundary`), if `begin > end`, or if
//     /// `end > len`.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// let s = "Löwe 老虎 Léopard";
//     /// assert_eq!(&s[0 .. 1], "L");
//     ///
//     /// assert_eq!(&s[1 .. 9], "öwe 老");
//     ///
//     /// // these will panic:
//     /// // byte 2 lies within `ö`:
//     /// // &s[2 ..3];
//     ///
//     /// // byte 8 lies within `老`
//     /// // &s[1 .. 8];
//     ///
//     /// // byte 100 is outside the string
//     /// // &s[3 .. 100];
//     /// ```
//     #[stable(feature = "str_checked_slicing", since = "1.20.0")]
//     impl SliceIndex<str> for ops::Range<usize> {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             if self.start <= self.end
//                 && slice.is_char_boundary(self.start)
//                 && slice.is_char_boundary(self.end)
//             {
//                 // SAFETY: just checked that `start` and `end` are on a char boundary.
//                 Some(unsafe { self.get_unchecked(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             if self.start <= self.end
//                 && slice.is_char_boundary(self.start)
//                 && slice.is_char_boundary(self.end)
//             {
//                 // SAFETY: just checked that `start` and `end` are on a char boundary.
//                 Some(unsafe { self.get_unchecked_mut(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             let ptr = slice.as_ptr().add(self.start);
//             let len = self.end - self.start;
//             super::from_utf8_unchecked(slice::from_raw_parts(ptr, len))
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             let ptr = slice.as_mut_ptr().add(self.start);
//             let len = self.end - self.start;
//             super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, len))
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             let (start, end) = (self.start, self.end);
//             self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, start, end))
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             // is_char_boundary checks that the index is in [0, .len()]
//             // cannot reuse `get` as above, because of NLL trouble
//             if self.start <= self.end
//                 && slice.is_char_boundary(self.start)
//                 && slice.is_char_boundary(self.end)
//             {
//                 // SAFETY: just checked that `start` and `end` are on a char boundary.
//                 unsafe { self.get_unchecked_mut(slice) }
//             } else {
//                 super::slice_error_fail(slice, self.start, self.end)
//             }
//         }
//     }

//     /// Implements substring slicing with syntax `&self[.. end]` or `&mut
//     /// self[.. end]`.
//     ///
//     /// Returns a slice of the given string from the byte range [`0`, `end`).
//     /// Equivalent to `&self[0 .. end]` or `&mut self[0 .. end]`.
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// Prior to 1.20.0, these indexing operations were still supported by
//     /// direct implementation of `Index` and `IndexMut`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `end` does not point to the starting byte offset of a
//     /// character (as defined by `is_char_boundary`), or if `end > len`.
//     #[stable(feature = "str_checked_slicing", since = "1.20.0")]
//     impl SliceIndex<str> for ops::RangeTo<usize> {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             if slice.is_char_boundary(self.end) {
//                 // SAFETY: just checked that `end` is on a char boundary.
//                 Some(unsafe { self.get_unchecked(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             if slice.is_char_boundary(self.end) {
//                 // SAFETY: just checked that `end` is on a char boundary.
//                 Some(unsafe { self.get_unchecked_mut(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             let ptr = slice.as_ptr();
//             super::from_utf8_unchecked(slice::from_raw_parts(ptr, self.end))
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             let ptr = slice.as_mut_ptr();
//             super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, self.end))
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             let end = self.end;
//             self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, 0, end))
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             if slice.is_char_boundary(self.end) {
//                 // SAFETY: just checked that `end` is on a char boundary.
//                 unsafe { self.get_unchecked_mut(slice) }
//             } else {
//                 super::slice_error_fail(slice, 0, self.end)
//             }
//         }
//     }

//     /// Implements substring slicing with syntax `&self[begin ..]` or `&mut
//     /// self[begin ..]`.
//     ///
//     /// Returns a slice of the given string from the byte range [`begin`,
//     /// `len`). Equivalent to `&self[begin .. len]` or `&mut self[begin ..
//     /// len]`.
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// Prior to 1.20.0, these indexing operations were still supported by
//     /// direct implementation of `Index` and `IndexMut`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `begin` does not point to the starting byte offset of
//     /// a character (as defined by `is_char_boundary`), or if `begin >= len`.
//     #[stable(feature = "str_checked_slicing", since = "1.20.0")]
//     impl SliceIndex<str> for ops::RangeFrom<usize> {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             if slice.is_char_boundary(self.start) {
//                 // SAFETY: just checked that `start` is on a char boundary.
//                 Some(unsafe { self.get_unchecked(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             if slice.is_char_boundary(self.start) {
//                 // SAFETY: just checked that `start` is on a char boundary.
//                 Some(unsafe { self.get_unchecked_mut(slice) })
//             } else {
//                 None
//             }
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             let ptr = slice.as_ptr().add(self.start);
//             let len = slice.len() - self.start;
//             super::from_utf8_unchecked(slice::from_raw_parts(ptr, len))
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             let ptr = slice.as_mut_ptr().add(self.start);
//             let len = slice.len() - self.start;
//             super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, len))
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             let (start, end) = (self.start, slice.len());
//             self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, start, end))
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             if slice.is_char_boundary(self.start) {
//                 // SAFETY: just checked that `start` is on a char boundary.
//                 unsafe { self.get_unchecked_mut(slice) }
//             } else {
//                 super::slice_error_fail(slice, self.start, slice.len())
//             }
//         }
//     }

//     /// Implements substring slicing with syntax `&self[begin ..= end]` or `&mut
//     /// self[begin ..= end]`.
//     ///
//     /// Returns a slice of the given string from the byte range
//     /// [`begin`, `end`]. Equivalent to `&self [begin .. end + 1]` or `&mut
//     /// self[begin .. end + 1]`, except if `end` has the maximum value for
//     /// `usize`.
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `begin` does not point to the starting byte offset of
//     /// a character (as defined by `is_char_boundary`), if `end` does not point
//     /// to the ending byte offset of a character (`end + 1` is either a starting
//     /// byte offset or equal to `len`), if `begin > end`, or if `end >= len`.
//     #[stable(feature = "inclusive_range", since = "1.26.0")]
//     impl SliceIndex<str> for ops::RangeInclusive<usize> {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             if *self.end() == usize::max_value() {
//                 None
//             } else {
//                 (*self.start()..self.end() + 1).get(slice)
//             }
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             if *self.end() == usize::max_value() {
//                 None
//             } else {
//                 (*self.start()..self.end() + 1).get_mut(slice)
//             }
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             (*self.start()..self.end() + 1).get_unchecked(slice)
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             (*self.start()..self.end() + 1).get_unchecked_mut(slice)
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             if *self.end() == usize::max_value() {
//                 str_index_overflow_fail();
//             }
//             (*self.start()..self.end() + 1).index(slice)
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             if *self.end() == usize::max_value() {
//                 str_index_overflow_fail();
//             }
//             (*self.start()..self.end() + 1).index_mut(slice)
//         }
//     }

//     /// Implements substring slicing with syntax `&self[..= end]` or `&mut
//     /// self[..= end]`.
//     ///
//     /// Returns a slice of the given string from the byte range [0, `end`].
//     /// Equivalent to `&self [0 .. end + 1]`, except if `end` has the maximum
//     /// value for `usize`.
//     ///
//     /// This operation is `O(1)`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `end` does not point to the ending byte offset of a character
//     /// (`end + 1` is either a starting byte offset as defined by
//     /// `is_char_boundary`, or equal to `len`), or if `end >= len`.
//     #[stable(feature = "inclusive_range", since = "1.26.0")]
//     impl SliceIndex<str> for ops::RangeToInclusive<usize> {
//         type Output = str;
//         #[inline]
//         fn get(self, slice: &str) -> Option<&Self::Output> {
//             if self.end == usize::max_value() { None } else { (..self.end + 1).get(slice) }
//         }
//         #[inline]
//         fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
//             if self.end == usize::max_value() { None } else { (..self.end + 1).get_mut(slice) }
//         }
//         #[inline]
//         unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
//             (..self.end + 1).get_unchecked(slice)
//         }
//         #[inline]
//         unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
//             (..self.end + 1).get_unchecked_mut(slice)
//         }
//         #[inline]
//         fn index(self, slice: &str) -> &Self::Output {
//             if self.end == usize::max_value() {
//                 str_index_overflow_fail();
//             }
//             (..self.end + 1).index(slice)
//         }
//         #[inline]
//         fn index_mut(self, slice: &mut str) -> &mut Self::Output {
//             if self.end == usize::max_value() {
//                 str_index_overflow_fail();
//             }
//             (..self.end + 1).index_mut(slice)
//         }
//     }
// }

// // truncate `&str` to length at most equal to `max`
// // return `true` if it were truncated, and the new str.
// fn truncate_to_char_boundary(s: &str, mut max: usize) -> (bool, &str) {
//     if max >= s.len() {
//         (false, s)
//     } else {
//         while !s.is_char_boundary(max) {
//             max -= 1;
//         }
//         (true, &s[..max])
//     }
// }

// #[inline(never)]
// #[cold]
// fn slice_error_fail(s: &str, begin: usize, end: usize) -> ! {
//     const MAX_DISPLAY_LENGTH: usize = 256;
//     let (truncated, s_trunc) = truncate_to_char_boundary(s, MAX_DISPLAY_LENGTH);
//     let ellipsis = if truncated { "[...]" } else { "" };

//     // 1. out of bounds
//     if begin > s.len() || end > s.len() {
//         let oob_index = if begin > s.len() { begin } else { end };
//         panic!("byte index {} is out of bounds of `{}`{}", oob_index, s_trunc, ellipsis);
//     }

//     // 2. begin <= end
//     assert!(
//         begin <= end,
//         "begin <= end ({} <= {}) when slicing `{}`{}",
//         begin,
//         end,
//         s_trunc,
//         ellipsis
//     );

//     // 3. character boundary
//     let index = if !s.is_char_boundary(begin) { begin } else { end };
//     // find the character
//     let mut char_start = index;
//     while !s.is_char_boundary(char_start) {
//         char_start -= 1;
//     }
//     // `char_start` must be less than len and a char boundary
//     let ch = s[char_start..].chars().next().unwrap();
//     let char_range = char_start..char_start + ch.len_utf8();
//     panic!(
//         "byte index {} is not a char boundary; it is inside {:?} (bytes {:?}) of `{}`{}",
//         index, ch, char_range, s_trunc, ellipsis
//     );
// }

pub trait AliasedSliceStr<'a, T: Eq>: SliceStr<'a, T> {
    fn contains_slice<P: Pattern<'a, T>>(&'a self, pat: P) -> bool {
        self.contains(pat)
    }
}

pub trait SliceStr<'a, T: Eq> {
    //     /// Divide one string slice into two at an index.
    //     ///
    //     /// The argument, `mid`, should be a byte offset from the start of the
    //     /// string. It must also be on the boundary of a UTF-8 code point.
    //     ///
    //     /// The two slices returned go from the start of the string slice to `mid`,
    //     /// and from `mid` to the end of the string slice.
    //     ///
    //     /// To get mutable string slices instead, see the [`split_at_mut`]
    //     /// method.
    //     ///
    //     /// [`split_at_mut`]: #method.split_at_mut
    //     ///
    //     /// # Panics
    //     ///
    //     /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    //     /// beyond the last code point of the string slice.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let s = "Per Martin-Löf";
    //     ///
    //     /// let (first, last) = s.split_at(3);
    //     ///
    //     /// assert_eq!("Per", first);
    //     /// assert_eq!(" Martin-Löf", last);
    //     /// ```
    //     #[inline]
    //     #[stable(feature = "str_split_at", since = "1.4.0")]
    //     pub fn split_at(&self, mid: usize) -> (&str, &str) {
    //         // is_char_boundary checks that the index is in [0, .len()]
    //         if self.is_char_boundary(mid) {
    //             // SAFETY: just checked that `mid` is on a char boundary.
    //             unsafe { (self.get_unchecked(0..mid), self.get_unchecked(mid..self.len())) }
    //         } else {
    //             slice_error_fail(self, 0, mid)
    //         }
    //     }

    //     /// Divide one mutable string slice into two at an index.
    //     ///
    //     /// The argument, `mid`, should be a byte offset from the start of the
    //     /// string. It must also be on the boundary of a UTF-8 code point.
    //     ///
    //     /// The two slices returned go from the start of the string slice to `mid`,
    //     /// and from `mid` to the end of the string slice.
    //     ///
    //     /// To get immutable string slices instead, see the [`split_at`] method.
    //     ///
    //     /// [`split_at`]: #method.split_at
    //     ///
    //     /// # Panics
    //     ///
    //     /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    //     /// beyond the last code point of the string slice.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let mut s = "Per Martin-Löf".to_string();
    //     /// {
    //     ///     let (first, last) = s.split_at_mut(3);
    //     ///     first.make_ascii_uppercase();
    //     ///     assert_eq!("PER", first);
    //     ///     assert_eq!(" Martin-Löf", last);
    //     /// }
    //     /// assert_eq!("PER Martin-Löf", s);
    //     /// ```
    //     #[inline]
    //     #[stable(feature = "str_split_at", since = "1.4.0")]
    //     pub fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
    //         // is_char_boundary checks that the index is in [0, .len()]
    //         if self.is_char_boundary(mid) {
    //             let len = self.len();
    //             let ptr = self.as_mut_ptr();
    //             // SAFETY: just checked that `mid` is on a char boundary.
    //             unsafe {
    //                 (
    //                     from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, mid)),
    //                     from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr.add(mid), len - mid)),
    //                 )
    //             }
    //         } else {
    //             slice_error_fail(self, 0, mid)
    //         }
    //     }

    //     /// Returns an iterator over the [`char`]s of a string slice, and their
    //     /// positions.
    //     ///
    //     /// As a string slice consists of valid UTF-8, we can iterate through a
    //     /// string slice by [`char`]. This method returns an iterator of both
    //     /// these [`char`]s, as well as their byte positions.
    //     ///
    //     /// The iterator yields tuples. The position is first, the [`char`] is
    //     /// second.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let word = "goodbye";
    //     ///
    //     /// let count = word.char_indices().count();
    //     /// assert_eq!(7, count);
    //     ///
    //     /// let mut char_indices = word.char_indices();
    //     ///
    //     /// assert_eq!(Some((0, 'g')), char_indices.next());
    //     /// assert_eq!(Some((1, 'o')), char_indices.next());
    //     /// assert_eq!(Some((2, 'o')), char_indices.next());
    //     /// assert_eq!(Some((3, 'd')), char_indices.next());
    //     /// assert_eq!(Some((4, 'b')), char_indices.next());
    //     /// assert_eq!(Some((5, 'y')), char_indices.next());
    //     /// assert_eq!(Some((6, 'e')), char_indices.next());
    //     ///
    //     /// assert_eq!(None, char_indices.next());
    //     /// ```
    //     ///
    //     /// Remember, [`char`]s may not match your human intuition about characters:
    //     ///
    //     /// ```
    //     /// let yes = "y̆es";
    //     ///
    //     /// let mut char_indices = yes.char_indices();
    //     ///
    //     /// assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
    //     /// assert_eq!(Some((1, '\u{0306}')), char_indices.next());
    //     ///
    //     /// // note the 3 here - the last character took up two bytes
    //     /// assert_eq!(Some((3, 'e')), char_indices.next());
    //     /// assert_eq!(Some((4, 's')), char_indices.next());
    //     ///
    //     /// assert_eq!(None, char_indices.next());
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn char_indices(&self) -> CharIndices<'_> {
    //         CharIndices { front_offset: 0, iter: self.chars() }
    //     }

    //     /// Splits a string slice by ASCII whitespace.
    //     ///
    //     /// The iterator returned will return string slices that are sub-slices of
    //     /// the original string slice, separated by any amount of ASCII whitespace.
    //     ///
    //     /// To split by Unicode `Whitespace` instead, use [`split_whitespace`].
    //     ///
    //     /// [`split_whitespace`]: #method.split_whitespace
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let mut iter = "A few words".split_ascii_whitespace();
    //     ///
    //     /// assert_eq!(Some("A"), iter.next());
    //     /// assert_eq!(Some("few"), iter.next());
    //     /// assert_eq!(Some("words"), iter.next());
    //     ///
    //     /// assert_eq!(None, iter.next());
    //     /// ```
    //     ///
    //     /// All kinds of ASCII whitespace are considered:
    //     ///
    //     /// ```
    //     /// let mut iter = " Mary   had\ta little  \n\t lamb".split_ascii_whitespace();
    //     /// assert_eq!(Some("Mary"), iter.next());
    //     /// assert_eq!(Some("had"), iter.next());
    //     /// assert_eq!(Some("a"), iter.next());
    //     /// assert_eq!(Some("little"), iter.next());
    //     /// assert_eq!(Some("lamb"), iter.next());
    //     ///
    //     /// assert_eq!(None, iter.next());
    //     /// ```
    //     #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
    //     #[inline]
    //     pub fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace<'_> {
    //         let inner =
    //             self.as_bytes().split(IsAsciiWhitespace).filter(BytesIsNotEmpty).map(UnsafeBytesToStr);
    //         SplitAsciiWhitespace { inner }
    //     }

    //     /// An iterator over the lines of a string, as string slices.
    //     ///
    //     /// Lines are ended with either a newline (`\n`) or a carriage return with
    //     /// a line feed (`\r\n`).
    //     ///
    //     /// The final line ending is optional.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let text = "foo\r\nbar\n\nbaz\n";
    //     /// let mut lines = text.lines();
    //     ///
    //     /// assert_eq!(Some("foo"), lines.next());
    //     /// assert_eq!(Some("bar"), lines.next());
    //     /// assert_eq!(Some(""), lines.next());
    //     /// assert_eq!(Some("baz"), lines.next());
    //     ///
    //     /// assert_eq!(None, lines.next());
    //     /// ```
    //     ///
    //     /// The final line ending isn't required:
    //     ///
    //     /// ```
    //     /// let text = "foo\nbar\n\r\nbaz";
    //     /// let mut lines = text.lines();
    //     ///
    //     /// assert_eq!(Some("foo"), lines.next());
    //     /// assert_eq!(Some("bar"), lines.next());
    //     /// assert_eq!(Some(""), lines.next());
    //     /// assert_eq!(Some("baz"), lines.next());
    //     ///
    //     /// assert_eq!(None, lines.next());
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn lines(&self) -> Lines<'_> {
    //         Lines(self.split_terminator('\n').map(LinesAnyMap))
    //     }

    /// Returns `true` if the given pattern matches a sub-slice of
    /// this string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.contains("nana"));
    /// assert!(!bananas.contains("apples"));
    /// ```
    // #[stable(feature = "rust1", since = "1.0.0")]
    fn contains<P: Pattern<'a, T>>(&'a self, pat: P) -> bool;

    //     /// Returns `true` if the given pattern matches a prefix of this
    //     /// string slice.
    //     ///
    //     /// Returns `false` if it does not.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let bananas = "bananas";
    //     ///
    //     /// assert!(bananas.starts_with("bana"));
    //     /// assert!(!bananas.starts_with("nana"));
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     pub fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
    //         pat.is_prefix_of(self)
    //     }

    //     /// Returns `true` if the given pattern matches a suffix of this
    //     /// string slice.
    //     ///
    //     /// Returns `false` if it does not.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let bananas = "bananas";
    //     ///
    //     /// assert!(bananas.ends_with("anas"));
    //     /// assert!(!bananas.ends_with("nana"));
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     pub fn ends_with<'a, P>(&'a self, pat: P) -> bool
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         pat.is_suffix_of(self)
    //     }

    /// Returns the byte index of the first character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find('L'), Some(0));
    /// assert_eq!(s.find('é'), Some(14));
    /// assert_eq!(s.find("Léopard"), Some(13));
    /// ```
    ///
    /// More complex patterns using point-free style and closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find(char::is_whitespace), Some(5));
    /// assert_eq!(s.find(char::is_lowercase), Some(1));
    /// assert_eq!(s.find(|c: char| c.is_whitespace() || c.is_lowercase()), Some(1));
    /// assert_eq!(s.find(|c: char| (c < 'o') && (c > 'a')), Some(4));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.find(x), None);
    /// ```
    // #[stable(feature = "rust1", since = "1.0.0")]
    fn find<P: Pattern<'a, T>>(&'a self, pat: P) -> Option<usize>;

    /// Returns the byte index of the last character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind('L'), Some(13));
    /// assert_eq!(s.rfind('é'), Some(14));
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind(char::is_whitespace), Some(12));
    /// assert_eq!(s.rfind(char::is_lowercase), Some(20));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.rfind(x), None);
    /// ```
    // #[stable(feature = "rust1", since = "1.0.0")]
    fn rfind<P>(&'a self, pat: P) -> Option<usize>
    where
        P: Pattern<'a, T>,
        P::Searcher: ReverseSearcher<'a, T>;

    // /// Splits a string slice by whitespace.
    // ///
    // /// The iterator returned will return string slices that are sub-slices of
    // /// the original string slice, separated by any amount of whitespace.
    // ///
    // /// 'Whitespace' is defined according to the terms of the Unicode Derived
    // /// Core Property `White_Space`. If you only want to split on ASCII whitespace
    // /// instead, use [`split_ascii_whitespace`].
    // ///
    // /// [`split_ascii_whitespace`]: #method.split_ascii_whitespace
    // ///
    // /// # Examples
    // ///
    // /// Basic usage:
    // ///
    // /// ```
    // /// let mut iter = "A few words".split_whitespace();
    // ///
    // /// assert_eq!(Some("A"), iter.next());
    // /// assert_eq!(Some("few"), iter.next());
    // /// assert_eq!(Some("words"), iter.next());
    // ///
    // /// assert_eq!(None, iter.next());
    // /// ```
    // ///
    // /// All kinds of whitespace are considered:
    // ///
    // /// ```
    // /// let mut iter = " Mary   had\ta\u{2009}little  \n\t lamb".split_whitespace();
    // /// assert_eq!(Some("Mary"), iter.next());
    // /// assert_eq!(Some("had"), iter.next());
    // /// assert_eq!(Some("a"), iter.next());
    // /// assert_eq!(Some("little"), iter.next());
    // /// assert_eq!(Some("lamb"), iter.next());
    // ///
    // /// assert_eq!(None, iter.next());
    // /// ```
    // #[stable(feature = "split_whitespace", since = "1.1.0")]
    // #[inline]
    // pub fn split_whitespace(&self) -> SplitWhitespace<'_> {
    //     SplitWhitespace { inner: self.split(IsWhitespace).filter(IsNotEmpty) }
    // }

    //     /// An iterator over substrings of this string slice, separated by
    //     /// characters matched by a pattern.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    //     /// allows a reverse search and forward/reverse search yields the same
    //     /// elements. This is true for, e.g., [`char`], but not for `&str`.
    //     ///
    //     /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    //     ///
    //     /// If the pattern allows a reverse search but its results might differ
    //     /// from a forward search, the [`rsplit`] method can be used.
    //     ///
    //     /// [`rsplit`]: #method.rsplit
    //     ///
    //     /// # Examples
    //     ///
    //     /// Simple patterns:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    //     /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    //     ///
    //     /// let v: Vec<&str> = "".split('X').collect();
    //     /// assert_eq!(v, [""]);
    //     ///
    //     /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    //     /// assert_eq!(v, ["lion", "", "tiger", "leopard"]);
    //     ///
    //     /// let v: Vec<&str> = "lion::tiger::leopard".split("::").collect();
    //     /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    //     ///
    //     /// let v: Vec<&str> = "abc1def2ghi".split(char::is_numeric).collect();
    //     /// assert_eq!(v, ["abc", "def", "ghi"]);
    //     ///
    //     /// let v: Vec<&str> = "lionXtigerXleopard".split(char::is_uppercase).collect();
    //     /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    //     /// ```
    //     ///
    //     /// A more complex pattern, using a closure:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "abc1defXghi".split(|c| c == '1' || c == 'X').collect();
    //     /// assert_eq!(v, ["abc", "def", "ghi"]);
    //     /// ```
    //     ///
    //     /// If a string contains multiple contiguous separators, you will end up
    //     /// with empty strings in the output:
    //     ///
    //     /// ```
    //     /// let x = "||||a||b|c".to_string();
    //     /// let d: Vec<_> = x.split('|').collect();
    //     ///
    //     /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    //     /// ```
    //     ///
    //     /// Contiguous separators are separated by the empty string.
    //     ///
    //     /// ```
    //     /// let x = "(///)".to_string();
    //     /// let d: Vec<_> = x.split('/').collect();
    //     ///
    //     /// assert_eq!(d, &["(", "", "", ")"]);
    //     /// ```
    //     ///
    //     /// Separators at the start or end of a string are neighbored
    //     /// by empty strings.
    //     ///
    //     /// ```
    //     /// let d: Vec<_> = "010".split("0").collect();
    //     /// assert_eq!(d, &["", "1", ""]);
    //     /// ```
    //     ///
    //     /// When the empty string is used as a separator, it separates
    //     /// every character in the string, along with the beginning
    //     /// and end of the string.
    //     ///
    //     /// ```
    //     /// let f: Vec<_> = "rust".split("").collect();
    //     /// assert_eq!(f, &["", "r", "u", "s", "t", ""]);
    //     /// ```
    //     ///
    //     /// Contiguous separators can lead to possibly surprising behavior
    //     /// when whitespace is used as the separator. This code is correct:
    //     ///
    //     /// ```
    //     /// let x = "    a  b c".to_string();
    //     /// let d: Vec<_> = x.split(' ').collect();
    //     ///
    //     /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    //     /// ```
    //     ///
    //     /// It does _not_ give you:
    //     ///
    //     /// ```,ignore
    //     /// assert_eq!(d, &["a", "b", "c"]);
    //     /// ```
    //     ///
    //     /// Use [`split_whitespace`] for this behavior.
    //     ///
    //     /// [`split_whitespace`]: #method.split_whitespace
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
    //         Split(SplitInternal {
    //             start: 0,
    //             end: self.len(),
    //             matcher: pat.into_searcher(self),
    //             allow_trailing_empty: true,
    //             finished: false,
    //         })
    //     }

    //     /// An iterator over substrings of the given string slice, separated by
    //     /// characters matched by a pattern and yielded in reverse order.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator requires that the pattern supports a reverse
    //     /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    //     /// search yields the same elements.
    //     ///
    //     /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    //     ///
    //     /// For iterating from the front, the [`split`] method can be used.
    //     ///
    //     /// [`split`]: #method.split
    //     ///
    //     /// # Examples
    //     ///
    //     /// Simple patterns:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "Mary had a little lamb".rsplit(' ').collect();
    //     /// assert_eq!(v, ["lamb", "little", "a", "had", "Mary"]);
    //     ///
    //     /// let v: Vec<&str> = "".rsplit('X').collect();
    //     /// assert_eq!(v, [""]);
    //     ///
    //     /// let v: Vec<&str> = "lionXXtigerXleopard".rsplit('X').collect();
    //     /// assert_eq!(v, ["leopard", "tiger", "", "lion"]);
    //     ///
    //     /// let v: Vec<&str> = "lion::tiger::leopard".rsplit("::").collect();
    //     /// assert_eq!(v, ["leopard", "tiger", "lion"]);
    //     /// ```
    //     ///
    //     /// A more complex pattern, using a closure:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "abc1defXghi".rsplit(|c| c == '1' || c == 'X').collect();
    //     /// assert_eq!(v, ["ghi", "def", "abc"]);
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn rsplit<'a, P>(&'a self, pat: P) -> RSplit<'a, P>
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         RSplit(self.split(pat).0)
    //     }

    //     /// An iterator over substrings of the given string slice, separated by
    //     /// characters matched by a pattern.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     ///
    //     /// Equivalent to [`split`], except that the trailing substring
    //     /// is skipped if empty.
    //     ///
    //     /// [`split`]: #method.split
    //     ///
    //     /// This method can be used for string data that is _terminated_,
    //     /// rather than _separated_ by a pattern.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    //     /// allows a reverse search and forward/reverse search yields the same
    //     /// elements. This is true for, e.g., [`char`], but not for `&str`.
    //     ///
    //     /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    //     ///
    //     /// If the pattern allows a reverse search but its results might differ
    //     /// from a forward search, the [`rsplit_terminator`] method can be used.
    //     ///
    //     /// [`rsplit_terminator`]: #method.rsplit_terminator
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    //     /// assert_eq!(v, ["A", "B"]);
    //     ///
    //     /// let v: Vec<&str> = "A..B..".split_terminator(".").collect();
    //     /// assert_eq!(v, ["A", "", "B", ""]);
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
    //         SplitTerminator(SplitInternal { allow_trailing_empty: false, ..self.split(pat).0 })
    //     }

    //     /// An iterator over substrings of `self`, separated by characters
    //     /// matched by a pattern and yielded in reverse order.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     /// Additional libraries might provide more complex patterns like
    //     /// regular expressions.
    //     ///
    //     /// Equivalent to [`split`], except that the trailing substring is
    //     /// skipped if empty.
    //     ///
    //     /// [`split`]: #method.split
    //     ///
    //     /// This method can be used for string data that is _terminated_,
    //     /// rather than _separated_ by a pattern.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator requires that the pattern supports a
    //     /// reverse search, and it will be double ended if a forward/reverse
    //     /// search yields the same elements.
    //     ///
    //     /// For iterating from the front, the [`split_terminator`] method can be
    //     /// used.
    //     ///
    //     /// [`split_terminator`]: #method.split_terminator
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "A.B.".rsplit_terminator('.').collect();
    //     /// assert_eq!(v, ["B", "A"]);
    //     ///
    //     /// let v: Vec<&str> = "A..B..".rsplit_terminator(".").collect();
    //     /// assert_eq!(v, ["", "B", "", "A"]);
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn rsplit_terminator<'a, P>(&'a self, pat: P) -> RSplitTerminator<'a, P>
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         RSplitTerminator(self.split_terminator(pat).0)
    //     }

    //     /// An iterator over substrings of the given string slice, separated by a
    //     /// pattern, restricted to returning at most `n` items.
    //     ///
    //     /// If `n` substrings are returned, the last substring (the `n`th substring)
    //     /// will contain the remainder of the string.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator will not be double ended, because it is
    //     /// not efficient to support.
    //     ///
    //     /// If the pattern allows a reverse search, the [`rsplitn`] method can be
    //     /// used.
    //     ///
    //     /// [`rsplitn`]: #method.rsplitn
    //     ///
    //     /// # Examples
    //     ///
    //     /// Simple patterns:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "Mary had a little lambda".splitn(3, ' ').collect();
    //     /// assert_eq!(v, ["Mary", "had", "a little lambda"]);
    //     ///
    //     /// let v: Vec<&str> = "lionXXtigerXleopard".splitn(3, "X").collect();
    //     /// assert_eq!(v, ["lion", "", "tigerXleopard"]);
    //     ///
    //     /// let v: Vec<&str> = "abcXdef".splitn(1, 'X').collect();
    //     /// assert_eq!(v, ["abcXdef"]);
    //     ///
    //     /// let v: Vec<&str> = "".splitn(1, 'X').collect();
    //     /// assert_eq!(v, [""]);
    //     /// ```
    //     ///
    //     /// A more complex pattern, using a closure:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "abc1defXghi".splitn(2, |c| c == '1' || c == 'X').collect();
    //     /// assert_eq!(v, ["abc", "defXghi"]);
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn splitn<'a, P: Pattern<'a>>(&'a self, n: usize, pat: P) -> SplitN<'a, P> {
    //         SplitN(SplitNInternal { iter: self.split(pat).0, count: n })
    //     }

    //     /// An iterator over substrings of this string slice, separated by a
    //     /// pattern, starting from the end of the string, restricted to returning
    //     /// at most `n` items.
    //     ///
    //     /// If `n` substrings are returned, the last substring (the `n`th substring)
    //     /// will contain the remainder of the string.
    //     ///
    //     /// The pattern can be any type that implements the Pattern trait. Notable
    //     /// examples are `&str`, [`char`], and closures that determines the split.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator will not be double ended, because it is not
    //     /// efficient to support.
    //     ///
    //     /// For splitting from the front, the [`splitn`] method can be used.
    //     ///
    //     /// [`splitn`]: #method.splitn
    //     ///
    //     /// # Examples
    //     ///
    //     /// Simple patterns:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(3, ' ').collect();
    //     /// assert_eq!(v, ["lamb", "little", "Mary had a"]);
    //     ///
    //     /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn(3, 'X').collect();
    //     /// assert_eq!(v, ["leopard", "tiger", "lionX"]);
    //     ///
    //     /// let v: Vec<&str> = "lion::tiger::leopard".rsplitn(2, "::").collect();
    //     /// assert_eq!(v, ["leopard", "lion::tiger"]);
    //     /// ```
    //     ///
    //     /// A more complex pattern, using a closure:
    //     ///
    //     /// ```
    //     /// let v: Vec<&str> = "abc1defXghi".rsplitn(2, |c| c == '1' || c == 'X').collect();
    //     /// assert_eq!(v, ["ghi", "abc1def"]);
    //     /// ```
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     #[inline]
    //     pub fn rsplitn<'a, P>(&'a self, n: usize, pat: P) -> RSplitN<'a, P>
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         RSplitN(self.splitn(n, pat).0)
    //     }

    /// An iterator over the disjoint matches of a pattern within the given string
    /// slice.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, e.g., [`char`], but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatches`] method can be used.
    ///
    /// [`rmatches`]: #method.rmatches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".matches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".matches(char::is_numeric).collect();
    /// assert_eq!(v, ["1", "2", "3"]);
    /// ```
    // #[stable(feature = "str_matches", since = "1.2.0")]
    fn matches<P: Pattern<'a, T>>(&'a self, pat: P) -> Matches<'a, T, P>;

    /// An iterator over the disjoint matches of a pattern within this string slice,
    /// yielded in reverse order.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`matches`] method can be used.
    ///
    /// [`matches`]: #method.matches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".rmatches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".rmatches(char::is_numeric).collect();
    /// assert_eq!(v, ["3", "2", "1"]);
    /// ```
    // #[stable(feature = "str_matches", since = "1.2.0")]
    fn rmatches<P>(&'a self, pat: P) -> RMatches<'a, T, P>
    where
        P: Pattern<'a, T>,
        P::Searcher: ReverseSearcher<'a, T>;

    //     /// An iterator over the disjoint matches of a pattern within this string
    //     /// slice as well as the index that the match starts at.
    //     ///
    //     /// For matches of `pat` within `self` that overlap, only the indices
    //     /// corresponding to the first match are returned.
    //     ///
    //     /// The pattern can be a `&str`, [`char`], or a closure that determines
    //     /// if a character matches.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    //     /// allows a reverse search and forward/reverse search yields the same
    //     /// elements. This is true for, e.g., [`char`], but not for `&str`.
    //     ///
    //     /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    //     ///
    //     /// If the pattern allows a reverse search but its results might differ
    //     /// from a forward search, the [`rmatch_indices`] method can be used.
    //     ///
    //     /// [`rmatch_indices`]: #method.rmatch_indices
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let v: Vec<_> = "abcXXXabcYYYabc".match_indices("abc").collect();
    //     /// assert_eq!(v, [(0, "abc"), (6, "abc"), (12, "abc")]);
    //     ///
    //     /// let v: Vec<_> = "1abcabc2".match_indices("abc").collect();
    //     /// assert_eq!(v, [(1, "abc"), (4, "abc")]);
    //     ///
    //     /// let v: Vec<_> = "ababa".match_indices("aba").collect();
    //     /// assert_eq!(v, [(0, "aba")]); // only the first `aba`
    //     /// ```
    //     #[stable(feature = "str_match_indices", since = "1.5.0")]
    //     #[inline]
    //     pub fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P> {
    //         MatchIndices(MatchIndicesInternal(pat.into_searcher(self)))
    //     }

    //     /// An iterator over the disjoint matches of a pattern within `self`,
    //     /// yielded in reverse order along with the index of the match.
    //     ///
    //     /// For matches of `pat` within `self` that overlap, only the indices
    //     /// corresponding to the last match are returned.
    //     ///
    //     /// The pattern can be a `&str`, [`char`], or a closure that determines if a
    //     /// character matches.
    //     ///
    //     /// # Iterator behavior
    //     ///
    //     /// The returned iterator requires that the pattern supports a reverse
    //     /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    //     /// search yields the same elements.
    //     ///
    //     /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    //     ///
    //     /// For iterating from the front, the [`match_indices`] method can be used.
    //     ///
    //     /// [`match_indices`]: #method.match_indices
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let v: Vec<_> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
    //     /// assert_eq!(v, [(12, "abc"), (6, "abc"), (0, "abc")]);
    //     ///
    //     /// let v: Vec<_> = "1abcabc2".rmatch_indices("abc").collect();
    //     /// assert_eq!(v, [(4, "abc"), (1, "abc")]);
    //     ///
    //     /// let v: Vec<_> = "ababa".rmatch_indices("aba").collect();
    //     /// assert_eq!(v, [(2, "aba")]); // only the last `aba`
    //     /// ```
    //     #[stable(feature = "str_match_indices", since = "1.5.0")]
    //     #[inline]
    //     pub fn rmatch_indices<'a, P>(&'a self, pat: P) -> RMatchIndices<'a, P>
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         RMatchIndices(self.match_indices(pat).0)
    //     }

    //     /// Returns a string slice with leading and trailing whitespace removed.
    //     ///
    //     /// 'Whitespace' is defined according to the terms of the Unicode Derived
    //     /// Core Property `White_Space`.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let s = " Hello\tworld\t";
    //     ///
    //     /// assert_eq!("Hello\tworld", s.trim());
    //     /// ```
    //     #[must_use = "this returns the trimmed string as a slice, \
    //                   without modifying the original"]
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     pub fn trim(&self) -> &str {
    //         self.trim_matches(|c: char| c.is_whitespace())
    //     }

    //     /// Returns a string slice with leading whitespace removed.
    //     ///
    //     /// 'Whitespace' is defined according to the terms of the Unicode Derived
    //     /// Core Property `White_Space`.
    //     ///
    //     /// # Text directionality
    //     ///
    //     /// A string is a sequence of bytes. `start` in this context means the first
    //     /// position of that byte string; for a left-to-right language like English or
    //     /// Russian, this will be left side, and for right-to-left languages like
    //     /// Arabic or Hebrew, this will be the right side.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let s = " Hello\tworld\t";
    //     /// assert_eq!("Hello\tworld\t", s.trim_start());
    //     /// ```
    //     ///
    //     /// Directionality:
    //     ///
    //     /// ```
    //     /// let s = "  English  ";
    //     /// assert!(Some('E') == s.trim_start().chars().next());
    //     ///
    //     /// let s = "  עברית  ";
    //     /// assert!(Some('ע') == s.trim_start().chars().next());
    //     /// ```
    //     #[must_use = "this returns the trimmed string as a new slice, \
    //                   without modifying the original"]
    //     #[stable(feature = "trim_direction", since = "1.30.0")]
    //     pub fn trim_start(&self) -> &str {
    //         self.trim_start_matches(|c: char| c.is_whitespace())
    //     }

    //     /// Returns a string slice with trailing whitespace removed.
    //     ///
    //     /// 'Whitespace' is defined according to the terms of the Unicode Derived
    //     /// Core Property `White_Space`.
    //     ///
    //     /// # Text directionality
    //     ///
    //     /// A string is a sequence of bytes. `end` in this context means the last
    //     /// position of that byte string; for a left-to-right language like English or
    //     /// Russian, this will be right side, and for right-to-left languages like
    //     /// Arabic or Hebrew, this will be the left side.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// let s = " Hello\tworld\t";
    //     /// assert_eq!(" Hello\tworld", s.trim_end());
    //     /// ```
    //     ///
    //     /// Directionality:
    //     ///
    //     /// ```
    //     /// let s = "  English  ";
    //     /// assert!(Some('h') == s.trim_end().chars().rev().next());
    //     ///
    //     /// let s = "  עברית  ";
    //     /// assert!(Some('ת') == s.trim_end().chars().rev().next());
    //     /// ```
    //     #[must_use = "this returns the trimmed string as a new slice, \
    //                   without modifying the original"]
    //     #[stable(feature = "trim_direction", since = "1.30.0")]
    //     pub fn trim_end(&self) -> &str {
    //         self.trim_end_matches(|c: char| c.is_whitespace())
    //     }

    /// Returns a string slice with all prefixes and suffixes that match a
    /// pattern repeatedly removed.
    ///
    /// The pattern can be a [`char`] or a closure that determines if a
    /// character matches.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    /// assert_eq!("123foo1bar123".trim_matches(char::is_numeric), "foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1foo1barXX".trim_matches(|c| c == '1' || c == 'X'), "foo1bar");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                      without modifying the original"]
    // #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_matches<P>(&'a self, pat: P) -> &'a [T]
    where
        P: Pattern<'a, T>,
        P::Searcher: DoubleEndedSearcher<'a, T>;

    //     /// Returns a string slice with all prefixes that match a pattern
    //     /// repeatedly removed.
    //     ///
    //     /// The pattern can be a `&str`, [`char`], or a closure that determines if
    //     /// a character matches.
    //     ///
    //     /// # Text directionality
    //     ///
    //     /// A string is a sequence of bytes. `start` in this context means the first
    //     /// position of that byte string; for a left-to-right language like English or
    //     /// Russian, this will be left side, and for right-to-left languages like
    //     /// Arabic or Hebrew, this will be the right side.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage:
    //     ///
    //     /// ```
    //     /// assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
    //     /// assert_eq!("123foo1bar123".trim_start_matches(char::is_numeric), "foo1bar123");
    //     ///
    //     /// let x: &[_] = &['1', '2'];
    //     /// assert_eq!("12foo1bar12".trim_start_matches(x), "foo1bar12");
    //     /// ```
    //     #[must_use = "this returns the trimmed string as a new slice, \
    //                   without modifying the original"]
    //     #[stable(feature = "trim_direction", since = "1.30.0")]
    //     pub fn trim_start_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
    //         let mut i = self.len();
    //         let mut matcher = pat.into_searcher(self);
    //         if let Some((a, _)) = matcher.next_reject() {
    //             i = a;
    //         }
    //         // SAFETY: `Searcher` is known to return valid indices.
    //         unsafe { self.get_unchecked(i..self.len()) }
    //     }

    //     /// Returns a string slice with the prefix removed.
    //     ///
    //     /// If the string starts with the pattern `prefix`, `Some` is returned with the substring where
    //     /// the prefix is removed. Unlike `trim_start_matches`, this method removes the prefix exactly
    //     /// once.
    //     ///
    //     /// If the string does not start with `prefix`, `None` is returned.
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// #![feature(str_strip)]
    //     ///
    //     /// assert_eq!("foobar".strip_prefix("foo"), Some("bar"));
    //     /// assert_eq!("foobar".strip_prefix("bar"), None);
    //     /// assert_eq!("foofoo".strip_prefix("foo"), Some("foo"));
    //     /// ```
    //     #[must_use = "this returns the remaining substring as a new slice, \
    //                   without modifying the original"]
    //     #[unstable(feature = "str_strip", reason = "newly added", issue = "67302")]
    //     pub fn strip_prefix<'a, P: Pattern<'a>>(&'a self, prefix: P) -> Option<&'a str> {
    //         let mut matcher = prefix.into_searcher(self);
    //         if let SearchStep::Match(start, len) = matcher.next() {
    //             debug_assert_eq!(
    //                 start, 0,
    //                 "The first search step from Searcher \
    //                 must include the first character"
    //             );
    //             // SAFETY: `Searcher` is known to return valid indices.
    //             unsafe { Some(self.get_unchecked(len..)) }
    //         } else {
    //             None
    //         }
    //     }

    //     /// Returns a string slice with the suffix removed.
    //     ///
    //     /// If the string ends with the pattern `suffix`, `Some` is returned with the substring where
    //     /// the suffix is removed. Unlike `trim_end_matches`, this method removes the suffix exactly
    //     /// once.
    //     ///
    //     /// If the string does not end with `suffix`, `None` is returned.
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// #![feature(str_strip)]
    //     /// assert_eq!("barfoo".strip_suffix("foo"), Some("bar"));
    //     /// assert_eq!("barfoo".strip_suffix("bar"), None);
    //     /// assert_eq!("foofoo".strip_suffix("foo"), Some("foo"));
    //     /// ```
    //     #[must_use = "this returns the remaining substring as a new slice, \
    //                   without modifying the original"]
    //     #[unstable(feature = "str_strip", reason = "newly added", issue = "67302")]
    //     pub fn strip_suffix<'a, P>(&'a self, suffix: P) -> Option<&'a str>
    //     where
    //         P: Pattern<'a>,
    //         <P as Pattern<'a>>::Searcher: ReverseSearcher<'a>,
    //     {
    //         let mut matcher = suffix.into_searcher(self);
    //         if let SearchStep::Match(start, end) = matcher.next_back() {
    //             debug_assert_eq!(
    //                 end,
    //                 self.len(),
    //                 "The first search step from ReverseSearcher \
    //                 must include the last character"
    //             );
    //             // SAFETY: `Searcher` is known to return valid indices.
    //             unsafe { Some(self.get_unchecked(..start)) }
    //         } else {
    //             None
    //         }
    //     }

    //     /// Returns a string slice with all suffixes that match a pattern
    //     /// repeatedly removed.
    //     ///
    //     /// The pattern can be a `&str`, [`char`], or a closure that
    //     /// determines if a character matches.
    //     ///
    //     /// # Text directionality
    //     ///
    //     /// A string is a sequence of bytes. `end` in this context means the last
    //     /// position of that byte string; for a left-to-right language like English or
    //     /// Russian, this will be right side, and for right-to-left languages like
    //     /// Arabic or Hebrew, this will be the left side.
    //     ///
    //     /// # Examples
    //     ///
    //     /// Simple patterns:
    //     ///
    //     /// ```
    //     /// assert_eq!("11foo1bar11".trim_end_matches('1'), "11foo1bar");
    //     /// assert_eq!("123foo1bar123".trim_end_matches(char::is_numeric), "123foo1bar");
    //     ///
    //     /// let x: &[_] = &['1', '2'];
    //     /// assert_eq!("12foo1bar12".trim_end_matches(x), "12foo1bar");
    //     /// ```
    //     ///
    //     /// A more complex pattern, using a closure:
    //     ///
    //     /// ```
    //     /// assert_eq!("1fooX".trim_end_matches(|c| c == '1' || c == 'X'), "1foo");
    //     /// ```
    //     #[must_use = "this returns the trimmed string as a new slice, \
    //                   without modifying the original"]
    //     #[stable(feature = "trim_direction", since = "1.30.0")]
    //     pub fn trim_end_matches<'a, P>(&'a self, pat: P) -> &'a str
    //     where
    //         P: Pattern<'a, Searcher: ReverseSearcher<'a>>,
    //     {
    //         let mut j = 0;
    //         let mut matcher = pat.into_searcher(self);
    //         if let Some((_, b)) = matcher.next_reject_back() {
    //             j = b;
    //         }
    //         // SAFETY: `Searcher` is known to return valid indices.
    //         unsafe { self.get_unchecked(0..j) }
    //     }

    //     /// Parses this string slice into another type.
    //     ///
    //     /// Because `parse` is so general, it can cause problems with type
    //     /// inference. As such, `parse` is one of the few times you'll see
    //     /// the syntax affectionately known as the 'turbofish': `::<>`. This
    //     /// helps the inference algorithm understand specifically which type
    //     /// you're trying to parse into.
    //     ///
    //     /// `parse` can parse any type that implements the [`FromStr`] trait.
    //     ///
    //     /// [`FromStr`]: str/trait.FromStr.html
    //     ///
    //     /// # Errors
    //     ///
    //     /// Will return [`Err`] if it's not possible to parse this string slice into
    //     /// the desired type.
    //     ///
    //     /// [`Err`]: str/trait.FromStr.html#associatedtype.Err
    //     ///
    //     /// # Examples
    //     ///
    //     /// Basic usage
    //     ///
    //     /// ```
    //     /// let four: u32 = "4".parse().unwrap();
    //     ///
    //     /// assert_eq!(4, four);
    //     /// ```
    //     ///
    //     /// Using the 'turbofish' instead of annotating `four`:
    //     ///
    //     /// ```
    //     /// let four = "4".parse::<u32>();
    //     ///
    //     /// assert_eq!(Ok(4), four);
    //     /// ```
    //     ///
    //     /// Failing to parse:
    //     ///
    //     /// ```
    //     /// let nope = "j".parse::<u32>();
    //     ///
    //     /// assert!(nope.is_err());
    //     /// ```
    //     #[inline]
    //     #[stable(feature = "rust1", since = "1.0.0")]
    //     pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
    //         FromStr::from_str(self)
    //     }

    //     /// Checks if all characters in this string are within the ASCII range.
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// let ascii = "hello!\n";
    //     /// let non_ascii = "Grüße, Jürgen ❤";
    //     ///
    //     /// assert!(ascii.is_ascii());
    //     /// assert!(!non_ascii.is_ascii());
    //     /// ```
    //     #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    //     #[inline]
    //     pub fn is_ascii(&self) -> bool {
    //         // We can treat each byte as character here: all multibyte characters
    //         // start with a byte that is not in the ascii range, so we will stop
    //         // there already.
    //         self.bytes().all(|b| b.is_ascii())
    //     }

    //     /// Checks that two strings are an ASCII case-insensitive match.
    //     ///
    //     /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    //     /// but without allocating and copying temporaries.
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// assert!("Ferris".eq_ignore_ascii_case("FERRIS"));
    //     /// assert!("Ferrös".eq_ignore_ascii_case("FERRöS"));
    //     /// assert!(!"Ferrös".eq_ignore_ascii_case("FERRÖS"));
    //     /// ```
    //     #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    //     #[inline]
    //     pub fn eq_ignore_ascii_case(&self, other: &str) -> bool {
    //         self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    //     }

    //     /// Converts this string to its ASCII upper case equivalent in-place.
    //     ///
    //     /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    //     /// but non-ASCII letters are unchanged.
    //     ///
    //     /// To return a new uppercased value without modifying the existing one, use
    //     /// [`to_ascii_uppercase`].
    //     ///
    //     /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// let mut s = String::from("Grüße, Jürgen ❤");
    //     ///
    //     /// s.make_ascii_uppercase();
    //     ///
    //     /// assert_eq!("GRüßE, JüRGEN ❤", s);
    //     /// ```
    //     #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    //     pub fn make_ascii_uppercase(&mut self) {
    //         // SAFETY: safe because we transmute two types with the same layout.
    //         let me = unsafe { self.as_bytes_mut() };
    //         me.make_ascii_uppercase()
    //     }

    //     /// Converts this string to its ASCII lower case equivalent in-place.
    //     ///
    //     /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    //     /// but non-ASCII letters are unchanged.
    //     ///
    //     /// To return a new lowercased value without modifying the existing one, use
    //     /// [`to_ascii_lowercase`].
    //     ///
    //     /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    //     ///
    //     /// # Examples
    //     ///
    //     /// ```
    //     /// let mut s = String::from("GRÜßE, JÜRGEN ❤");
    //     ///
    //     /// s.make_ascii_lowercase();
    //     ///
    //     /// assert_eq!("grÜße, jÜrgen ❤", s);
    //     /// ```
    //     #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    //     pub fn make_ascii_lowercase(&mut self) {
    //         // SAFETY: safe because we transmute two types with the same layout.
    //         let me = unsafe { self.as_bytes_mut() };
    //         me.make_ascii_lowercase()
    //     }

    //     /// Return an iterator that escapes each char in `self` with [`char::escape_debug`].
    //     ///
    //     /// Note: only extended grapheme codepoints that begin the string will be
    //     /// escaped.
    //     ///
    //     /// [`char::escape_debug`]: ../std/primitive.char.html#method.escape_debug
    //     ///
    //     /// # Examples
    //     ///
    //     /// As an iterator:
    //     ///
    //     /// ```
    //     /// for c in "❤\n!".escape_debug() {
    //     ///     print!("{}", c);
    //     /// }
    //     /// println!();
    //     /// ```
    //     ///
    //     /// Using `println!` directly:
    //     ///
    //     /// ```
    //     /// println!("{}", "❤\n!".escape_debug());
    //     /// ```
    //     ///
    //     ///
    //     /// Both are equivalent to:
    //     ///
    //     /// ```
    //     /// println!("❤\\n!");
    //     /// ```
    //     ///
    //     /// Using `to_string`:
    //     ///
    //     /// ```
    //     /// assert_eq!("❤\n!".escape_debug().to_string(), "❤\\n!");
    //     /// ```
    //     #[stable(feature = "str_escape", since = "1.34.0")]
    //     pub fn escape_debug(&self) -> EscapeDebug<'_> {
    //         let mut chars = self.chars();
    //         EscapeDebug {
    //             inner: chars
    //                 .next()
    //                 .map(|first| first.escape_debug_ext(true))
    //                 .into_iter()
    //                 .flatten()
    //                 .chain(chars.flat_map(CharEscapeDebugContinue)),
    //         }
    //     }

    //     /// Return an iterator that escapes each char in `self` with [`char::escape_default`].
    //     ///
    //     /// [`char::escape_default`]: ../std/primitive.char.html#method.escape_default
    //     ///
    //     /// # Examples
    //     ///
    //     /// As an iterator:
    //     ///
    //     /// ```
    //     /// for c in "❤\n!".escape_default() {
    //     ///     print!("{}", c);
    //     /// }
    //     /// println!();
    //     /// ```
    //     ///
    //     /// Using `println!` directly:
    //     ///
    //     /// ```
    //     /// println!("{}", "❤\n!".escape_default());
    //     /// ```
    //     ///
    //     ///
    //     /// Both are equivalent to:
    //     ///
    //     /// ```
    //     /// println!("\\u{{2764}}\\n!");
    //     /// ```
    //     ///
    //     /// Using `to_string`:
    //     ///
    //     /// ```
    //     /// assert_eq!("❤\n!".escape_default().to_string(), "\\u{2764}\\n!");
    //     /// ```
    //     #[stable(feature = "str_escape", since = "1.34.0")]
    //     pub fn escape_default(&self) -> EscapeDefault<'_> {
    //         EscapeDefault { inner: self.chars().flat_map(CharEscapeDefault) }
    //     }

    //     /// Return an iterator that escapes each char in `self` with [`char::escape_unicode`].
    //     ///
    //     /// [`char::escape_unicode`]: ../std/primitive.char.html#method.escape_unicode
    //     ///
    //     /// # Examples
    //     ///
    //     /// As an iterator:
    //     ///
    //     /// ```
    //     /// for c in "❤\n!".escape_unicode() {
    //     ///     print!("{}", c);
    //     /// }
    //     /// println!();
    //     /// ```
    //     ///
    //     /// Using `println!` directly:
    //     ///
    //     /// ```
    //     /// println!("{}", "❤\n!".escape_unicode());
    //     /// ```
    //     ///
    //     ///
    //     /// Both are equivalent to:
    //     ///
    //     /// ```
    //     /// println!("\\u{{2764}}\\u{{a}}\\u{{21}}");
    //     /// ```
    //     ///
    //     /// Using `to_string`:
    //     ///
    //     /// ```
    //     /// assert_eq!("❤\n!".escape_unicode().to_string(), "\\u{2764}\\u{a}\\u{21}");
    //     /// ```
    //     #[stable(feature = "str_escape", since = "1.34.0")]
    //     pub fn escape_unicode(&self) -> EscapeUnicode<'_> {
    //         EscapeUnicode { inner: self.chars().flat_map(CharEscapeUnicode) }
    //     }
}

impl<'a, T: 'a + Eq> AliasedSliceStr<'a, T> for [T] {}

impl<'a, T: 'a + Eq> SliceStr<'a, T> for [T] {
    #[inline]
    fn contains<P: Pattern<'a, T>>(&'a self, pat: P) -> bool {
        pat.is_contained_in(self)
    }

    #[inline]
    fn find<P: Pattern<'a, T>>(&'a self, pat: P) -> Option<usize> {
        pat.into_searcher(self).next_match().map(|(i, _)| i)
    }

    #[inline]
    fn rfind<P: Pattern<'a, T>>(&'a self, pat: P) -> Option<usize>
    where
        P::Searcher: ReverseSearcher<'a, T>,
    {
        pat.into_searcher(self).next_match_back().map(|(i, _)| i)
    }

    #[inline]
    fn matches<P: Pattern<'a, T>>(&'a self, pat: P) -> Matches<'a, T, P> {
        Matches(MatchesInternal(pat.into_searcher(self)))
    }

    #[inline]
    fn rmatches<P: Pattern<'a, T>>(&'a self, pat: P) -> RMatches<'a, T, P>
    where
        P::Searcher: ReverseSearcher<'a, T>,
    {
        RMatches(self.matches(pat).0)
    }

    fn trim_matches<P>(&'a self, pat: P) -> &'a [T]
    where
        P: Pattern<'a, T>,
        P::Searcher: DoubleEndedSearcher<'a, T>,
    {
        let mut i = 0;
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((a, b)) = matcher.next_reject() {
            i = a;
            j = b; // Remember earliest known match, correct it below if
                   // last match is different
        }
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { self.get_unchecked(i..j) }
    }
}

// impl_fn_for_zst! {
//     #[derive(Clone)]
//     struct CharEscapeDebugContinue impl Fn = |c: char| -> char::EscapeDebug {
//         c.escape_debug_ext(false)
//     };

//     #[derive(Clone)]
//     struct CharEscapeUnicode impl Fn = |c: char| -> char::EscapeUnicode {
//         c.escape_unicode()
//     };
//     #[derive(Clone)]
//     struct CharEscapeDefault impl Fn = |c: char| -> char::EscapeDefault {
//         c.escape_default()
//     };
// }

// #[stable(feature = "rust1", since = "1.0.0")]
// impl AsRef<[u8]> for str {
//     #[inline]
//     fn as_ref(&self) -> &[u8] {
//         self.as_bytes()
//     }
// }

// #[stable(feature = "rust1", since = "1.0.0")]
// impl Default for &str {
//     /// Creates an empty str
//     fn default() -> Self {
//         ""
//     }
// }

// #[stable(feature = "default_mut_str", since = "1.28.0")]
// impl Default for &mut str {
//     /// Creates an empty mutable str
//     fn default() -> Self {
//         // SAFETY: The empty string is valid UTF-8.
//         unsafe { from_utf8_unchecked_mut(&mut []) }
//     }
// }

// /// An iterator over the non-whitespace substrings of a string,
// /// separated by any amount of whitespace.
// ///
// /// This struct is created by the [`split_whitespace`] method on [`str`].
// /// See its documentation for more.
// ///
// /// [`split_whitespace`]: ../../std/primitive.str.html#method.split_whitespace
// /// [`str`]: ../../std/primitive.str.html
// #[stable(feature = "split_whitespace", since = "1.1.0")]
// #[derive(Clone, Debug)]
// pub struct SplitWhitespace<'a> {
//     inner: Filter<Split<'a, IsWhitespace>, IsNotEmpty>,
// }

// /// An iterator over the non-ASCII-whitespace substrings of a string,
// /// separated by any amount of ASCII whitespace.
// ///
// /// This struct is created by the [`split_ascii_whitespace`] method on [`str`].
// /// See its documentation for more.
// ///
// /// [`split_ascii_whitespace`]: ../../std/primitive.str.html#method.split_ascii_whitespace
// /// [`str`]: ../../std/primitive.str.html
// #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
// #[derive(Clone, Debug)]
// pub struct SplitAsciiWhitespace<'a> {
//     inner: Map<Filter<SliceSplit<'a, u8, IsAsciiWhitespace>, BytesIsNotEmpty>, UnsafeBytesToStr>,
// }

// impl_fn_for_zst! {
//     #[derive(Clone)]
//     struct IsWhitespace impl Fn = |c: char| -> bool {
//         c.is_whitespace()
//     };

//     #[derive(Clone)]
//     struct IsAsciiWhitespace impl Fn = |byte: &u8| -> bool {
//         byte.is_ascii_whitespace()
//     };

//     #[derive(Clone)]
//     struct IsNotEmpty impl<'a, 'b> Fn = |s: &'a &'b str| -> bool {
//         !s.is_empty()
//     };

//     #[derive(Clone)]
//     struct BytesIsNotEmpty impl<'a, 'b> Fn = |s: &'a &'b [u8]| -> bool {
//         !s.is_empty()
//     };

//     #[derive(Clone)]
//     struct UnsafeBytesToStr impl<'a> Fn = |bytes: &'a [u8]| -> &'a str {
//         // SAFETY: not safe
//         unsafe { from_utf8_unchecked(bytes) }
//     };
// }

// #[stable(feature = "split_whitespace", since = "1.1.0")]
// impl<'a> Iterator for SplitWhitespace<'a> {
//     type Item = &'a str;

//     #[inline]
//     fn next(&mut self) -> Option<&'a str> {
//         self.inner.next()
//     }

//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }

//     #[inline]
//     fn last(mut self) -> Option<&'a str> {
//         self.next_back()
//     }
// }

// #[stable(feature = "split_whitespace", since = "1.1.0")]
// impl<'a> DoubleEndedIterator for SplitWhitespace<'a> {
//     #[inline]
//     fn next_back(&mut self) -> Option<&'a str> {
//         self.inner.next_back()
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl FusedIterator for SplitWhitespace<'_> {}

// #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
// impl<'a> Iterator for SplitAsciiWhitespace<'a> {
//     type Item = &'a str;

//     #[inline]
//     fn next(&mut self) -> Option<&'a str> {
//         self.inner.next()
//     }

//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }

//     #[inline]
//     fn last(mut self) -> Option<&'a str> {
//         self.next_back()
//     }
// }

// #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
// impl<'a> DoubleEndedIterator for SplitAsciiWhitespace<'a> {
//     #[inline]
//     fn next_back(&mut self) -> Option<&'a str> {
//         self.inner.next_back()
//     }
// }

// #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
// impl FusedIterator for SplitAsciiWhitespace<'_> {}

// /// An iterator of [`u16`] over the string encoded as UTF-16.
// ///
// /// [`u16`]: ../../std/primitive.u16.html
// ///
// /// This struct is created by the [`encode_utf16`] method on [`str`].
// /// See its documentation for more.
// ///
// /// [`encode_utf16`]: ../../std/primitive.str.html#method.encode_utf16
// /// [`str`]: ../../std/primitive.str.html
// #[derive(Clone)]
// #[stable(feature = "encode_utf16", since = "1.8.0")]
// pub struct EncodeUtf16<'a> {
//     chars: Chars<'a>,
//     extra: u16,
// }

// #[stable(feature = "collection_debug", since = "1.17.0")]
// impl fmt::Debug for EncodeUtf16<'_> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.pad("EncodeUtf16 { .. }")
//     }
// }

// #[stable(feature = "encode_utf16", since = "1.8.0")]
// impl<'a> Iterator for EncodeUtf16<'a> {
//     type Item = u16;

//     #[inline]
//     fn next(&mut self) -> Option<u16> {
//         if self.extra != 0 {
//             let tmp = self.extra;
//             self.extra = 0;
//             return Some(tmp);
//         }

//         let mut buf = [0; 2];
//         self.chars.next().map(|ch| {
//             let n = ch.encode_utf16(&mut buf).len();
//             if n == 2 {
//                 self.extra = buf[1];
//             }
//             buf[0]
//         })
//     }

//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let (low, high) = self.chars.size_hint();
//         // every char gets either one u16 or two u16,
//         // so this iterator is between 1 or 2 times as
//         // long as the underlying iterator.
//         (low, high.and_then(|n| n.checked_mul(2)))
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl FusedIterator for EncodeUtf16<'_> {}

// /// The return type of [`str::escape_debug`].
// ///
// /// [`str::escape_debug`]: ../../std/primitive.str.html#method.escape_debug
// #[stable(feature = "str_escape", since = "1.34.0")]
// #[derive(Clone, Debug)]
// pub struct EscapeDebug<'a> {
//     inner: Chain<
//         Flatten<option::IntoIter<char::EscapeDebug>>,
//         FlatMap<Chars<'a>, char::EscapeDebug, CharEscapeDebugContinue>,
//     >,
// }

// /// The return type of [`str::escape_default`].
// ///
// /// [`str::escape_default`]: ../../std/primitive.str.html#method.escape_default
// #[stable(feature = "str_escape", since = "1.34.0")]
// #[derive(Clone, Debug)]
// pub struct EscapeDefault<'a> {
//     inner: FlatMap<Chars<'a>, char::EscapeDefault, CharEscapeDefault>,
// }

// /// The return type of [`str::escape_unicode`].
// ///
// /// [`str::escape_unicode`]: ../../std/primitive.str.html#method.escape_unicode
// #[stable(feature = "str_escape", since = "1.34.0")]
// #[derive(Clone, Debug)]
// pub struct EscapeUnicode<'a> {
//     inner: FlatMap<Chars<'a>, char::EscapeUnicode, CharEscapeUnicode>,
// }

// macro_rules! escape_types_impls {
//     ($( $Name: ident ),+) => {$(
//         #[stable(feature = "str_escape", since = "1.34.0")]
//         impl<'a> fmt::Display for $Name<'a> {
//             fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//                 self.clone().try_for_each(|c| f.write_char(c))
//             }
//         }

//         #[stable(feature = "str_escape", since = "1.34.0")]
//         impl<'a> Iterator for $Name<'a> {
//             type Item = char;

//             #[inline]
//             fn next(&mut self) -> Option<char> { self.inner.next() }

//             #[inline]
//             fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }

//             #[inline]
//             fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
//                 Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
//             {
//                 self.inner.try_fold(init, fold)
//             }

//             #[inline]
//             fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
//                 where Fold: FnMut(Acc, Self::Item) -> Acc,
//             {
//                 self.inner.fold(init, fold)
//             }
//         }

//         #[stable(feature = "str_escape", since = "1.34.0")]
//         impl<'a> FusedIterator for $Name<'a> {}
//     )+}
// }

// escape_types_impls!(EscapeDebug, EscapeDefault, EscapeUnicode);

#[cfg(test)]
mod tests {
    use super::{AliasedSliceStr, SliceStr};

    #[test]
    fn find() {
        let haystack = b"1234512345";
        assert!(haystack[..].find(&b"x"[..]) == None);
        assert!(haystack[..].find(&b"34"[..]) == Some(2));
        assert!(haystack[3..].find(&b"34"[..]) == Some(4));
    }

    #[test]
    fn rfind() {
        let haystack = b"1234512345";
        assert!(haystack[..].rfind(&b"x"[..]) == None);
        assert!(haystack[..].rfind(&b"34"[..]) == Some(7));
        assert!(haystack[3..].rfind(&b"34"[..]) == Some(4));
    }

    #[test]
    fn matches() {
        let haystack = b"1234512345";
        assert!(haystack[..].matches(&b"3"[..]).count() == 2);
        assert!(haystack[..].matches(&b"34"[..]).count() == 2);
        assert!(haystack[..].matches(&b"3451"[..]).count() == 1);

        let fst = haystack[..].matches(&b"3451"[..]).next().unwrap();
        assert!(fst == &b"3451"[..]);
    }

    #[test]
    fn contains() {
        let haystack = &b"1234512345"[..];
        assert!(!haystack[..].contains_slice(&b"xy"[..]));
        assert!(haystack[..].contains_slice(&b"45"[..]));
    }
}
