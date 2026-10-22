//! A fast bitset for algorithms that work on a bounded range of `u32` indices.
//!
//! [`OffsetBitset`] stores its bits in a [`FixedBitSet`], but only from the first used `usize`
//! block on: the bitset keeps an offset (counted in whole blocks) and stores bit `i` at
//! `i - offset * usize::BITS`. A set whose bits are clustered at high indices therefore doesn't
//! allocate the empty blocks below them.
//!
//! Compared to a `RoaringBitmap`, which is optimized for compact storage, every operation is a
//! simple loop over machine words. That makes unions and subset checks fast when the bits span a
//! limited range, at the cost of using one bit of memory per index in that range.

use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::BitOrAssign,
};

use fixedbitset::FixedBitSet;
use roaring::RoaringBitmap;

/// Number of bits per storage block.
const BLOCK_BITS: usize = usize::BITS as usize;

/// A set of `u32` indices backed by a [`FixedBitSet`] that only covers the block-aligned range
/// between the lowest and the highest stored bit.
///
/// The storage grows at the front or the back when a bit outside of the current range is
/// inserted. Equality and hashing only depend on which bits are set, not on the storage range.
#[derive(Clone, Default)]
pub struct OffsetBitset {
    /// The index of the first stored block, i.e. the first stored bit is
    /// `offset * usize::BITS`.
    offset: usize,
    /// The stored blocks. Its length is always a multiple of [`BLOCK_BITS`].
    bits: FixedBitSet,
}

impl OffsetBitset {
    /// Creates an empty set without allocating.
    pub const fn new() -> Self {
        Self {
            offset: 0,
            bits: FixedBitSet::new(),
        }
    }

    /// The stored blocks.
    #[inline]
    fn blocks(&self) -> &[usize] {
        self.bits.as_slice()
    }

    /// The index of the block after the last stored block.
    #[inline]
    fn end(&self) -> usize {
        self.offset + self.blocks().len()
    }

    /// The stored blocks with leading and trailing zero blocks removed, together with the block
    /// index of the first returned block. The empty set always returns `(0, &[])`.
    fn trimmed(&self) -> (usize, &[usize]) {
        let blocks = self.blocks();
        let Some(first) = blocks.iter().position(|&b| b != 0) else {
            return (0, &[]);
        };
        // There is a non-zero block, so `rposition` finds one.
        let last = blocks.iter().rposition(|&b| b != 0).unwrap_or(first);
        (self.offset + first, &blocks[first..=last])
    }

    /// Makes sure that the blocks `start..end` are stored, keeping all existing bits.
    fn ensure_blocks(&mut self, start: usize, end: usize) {
        debug_assert!(start < end);
        let len = self.blocks().len();
        if len == 0 {
            self.offset = start;
            self.bits = FixedBitSet::with_capacity((end - start) * BLOCK_BITS);
            return;
        }
        let old_end = self.offset + len;
        if start < self.offset {
            // Grow at the front (and maybe the back): copy the existing blocks into a new buffer.
            let new_end = end.max(old_end);
            let mut bits = FixedBitSet::with_capacity((new_end - start) * BLOCK_BITS);
            let shift = self.offset - start;
            bits.as_mut_slice()[shift..shift + len].copy_from_slice(self.blocks());
            self.offset = start;
            self.bits = bits;
        } else if end > old_end {
            // Grow at the back. `FixedBitSet::grow` reserves the memory with amortized growth.
            self.bits.grow((end - self.offset) * BLOCK_BITS);
        }
    }

    /// Adds `bit` to the set. Returns whether it wasn't in the set before.
    pub fn insert(&mut self, bit: u32) -> bool {
        let bit = bit as usize;
        let block = bit / BLOCK_BITS;
        if block < self.offset || block >= self.end() {
            self.ensure_blocks(block, block + 1);
        }
        let index = bit - self.offset * BLOCK_BITS;
        !self.bits.put(index)
    }

    /// Returns whether `bit` is in the set.
    pub fn contains(&self, bit: u32) -> bool {
        let bit = bit as usize;
        let block = bit / BLOCK_BITS;
        if block < self.offset || block >= self.end() {
            return false;
        }
        self.bits.contains(bit - self.offset * BLOCK_BITS)
    }

    /// Returns whether the set has no bits.
    pub fn is_empty(&self) -> bool {
        self.blocks().iter().all(|&b| b == 0)
    }

    /// Returns the number of bits in the set.
    pub fn len(&self) -> usize {
        self.blocks().iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Iterates over the bits in the set in ascending order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            blocks: self.blocks(),
            block_index: self.offset,
            current: 0,
        }
    }

    /// Adds all bits of `other` to the set.
    pub fn union_with(&mut self, other: &Self) {
        let (other_offset, other_blocks) = other.trimmed();
        if other_blocks.is_empty() {
            return;
        }
        let other_end = other_offset + other_blocks.len();
        if other_offset < self.offset || other_end > self.end() {
            self.ensure_blocks(other_offset, other_end);
        }
        let start = other_offset - self.offset;
        for (block, &other_block) in self.bits.as_mut_slice()[start..start + other_blocks.len()]
            .iter_mut()
            .zip(other_blocks)
        {
            *block |= other_block;
        }
    }

    /// Compares the set with `other`. Returns whether the set contains every bit of `other`
    /// (superset). If `STRICT` is true, it also returns whether the set has a bit that `other`
    /// doesn't have (only meaningful if it is a superset); otherwise the second value is `false`.
    fn superset_and_strict<const STRICT: bool>(&self, other: &Self) -> (bool, bool) {
        let blocks = self.blocks();
        let other_blocks = other.blocks();
        let any = |blocks: &[usize]| blocks.iter().any(|&b| b != 0);

        // The range of block indices stored in both sets.
        let start = self.offset.max(other.offset);
        let end = self.end().min(other.end());
        if start >= end {
            // No overlap: `other` must be empty.
            if any(other_blocks) {
                return (false, false);
            }
            return (true, STRICT && any(blocks));
        }

        // Bits of `other` outside of the overlap aren't in the set.
        let other_start = start - other.offset;
        let other_end = end - other.offset;
        if any(&other_blocks[..other_start]) || any(&other_blocks[other_end..]) {
            return (false, false);
        }

        let self_start = start - self.offset;
        let self_end = end - self.offset;
        let mut strict = false;
        for (&a, &b) in blocks[self_start..self_end]
            .iter()
            .zip(&other_blocks[other_start..other_end])
        {
            if b & !a != 0 {
                return (false, false);
            }
            if STRICT {
                strict |= a & !b != 0;
            }
        }
        if STRICT && !strict {
            strict = any(&blocks[..self_start]) || any(&blocks[self_end..]);
        }
        (true, strict)
    }

    /// Returns whether the set contains every bit of `other`.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.superset_and_strict::<false>(other).0
    }

    /// Returns whether every bit of the set is also in `other`.
    ///
    /// `!a.is_subset(b)` is the check for "`a` has a bit that `b` doesn't have", i.e. whether
    /// `b |= a` would change `b`.
    pub fn is_subset(&self, other: &Self) -> bool {
        other.is_superset(self)
    }

    /// Returns whether the set contains every bit of `other` and at least one bit that `other`
    /// doesn't contain.
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        let (superset, strict) = self.superset_and_strict::<true>(other);
        superset && strict
    }

    /// Converts the set into a [`RoaringBitmap`] with the same bits.
    ///
    /// This builds every roaring container (a range of 2^16 bits) directly from the stored blocks
    /// instead of inserting each bit, and produces the same container types as inserting the bits
    /// one by one would, so the result is equal (and serializes identically) to an inserted one.
    pub fn to_roaring(&self) -> RoaringBitmap {
        /// The number of blocks per roaring container.
        const CONTAINER_BLOCKS: usize = (1 << 16) / BLOCK_BITS;
        /// Roaring stores a container with up to this many bits as a sorted array.
        const ROARING_ARRAY_LIMIT: u32 = 4096;

        let (first_block, blocks) = self.trimmed();
        let end = first_block + blocks.len();
        let mut result = RoaringBitmap::new();
        let mut bytes = Vec::new();
        let mut index = first_block;
        while index < end {
            let chunk_end = ((index / CONTAINER_BLOCKS + 1) * CONTAINER_BLOCKS).min(end);
            let chunk = &blocks[index - first_block..chunk_end - first_block];
            let count = chunk.iter().map(|b| b.count_ones()).sum::<u32>();
            let part = if count == 0 {
                None
            } else if count == ROARING_ARRAY_LIMIT {
                // `from_lsb0_bytes` stores exactly this many bits as a bitmap, but inserting them
                // stores them as an array, so insert them instead.
                let bits = Iter {
                    blocks: chunk,
                    block_index: index,
                    current: 0,
                };
                Some(
                    RoaringBitmap::from_sorted_iter(bits)
                        .expect("the bits of an OffsetBitset are iterated in ascending order"),
                )
            } else {
                bytes.clear();
                bytes.extend(chunk.iter().flat_map(|block| block.to_le_bytes()));
                // A chunk is at most one container, so the range fits into `u32`.
                Some(RoaringBitmap::from_lsb0_bytes(
                    (index * BLOCK_BITS) as u32,
                    &bytes,
                ))
            };
            if let Some(part) = part {
                if result.is_empty() {
                    result = part;
                } else {
                    // The parts are in different containers, so this only adds containers.
                    result |= part;
                }
            }
            index = chunk_end;
        }
        result
    }
}

impl BitOrAssign<&OffsetBitset> for OffsetBitset {
    fn bitor_assign(&mut self, rhs: &OffsetBitset) {
        self.union_with(rhs)
    }
}

impl Extend<u32> for OffsetBitset {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, iter: T) {
        let bits = iter.into_iter().collect::<Vec<_>>();
        let (Some(&min), Some(&max)) = (bits.iter().min(), bits.iter().max()) else {
            return;
        };
        // Allocate the whole range once instead of growing for every bit.
        let mut start = min as usize / BLOCK_BITS;
        let mut end = max as usize / BLOCK_BITS + 1;
        if !self.blocks().is_empty() {
            start = start.min(self.offset);
            end = end.max(self.end());
        }
        self.ensure_blocks(start, end);
        let first_bit = self.offset * BLOCK_BITS;
        for bit in bits {
            self.bits.insert(bit as usize - first_bit);
        }
    }
}

impl FromIterator<u32> for OffsetBitset {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl PartialEq for OffsetBitset {
    fn eq(&self, other: &Self) -> bool {
        self.trimmed() == other.trimmed()
    }
}

impl Eq for OffsetBitset {}

impl Hash for OffsetBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.trimmed().hash(state);
    }
}

impl fmt::Debug for OffsetBitset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a OffsetBitset {
    type Item = u32;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the bits of an [`OffsetBitset`] in ascending order, see
/// [`OffsetBitset::iter`].
pub struct Iter<'a> {
    /// The blocks that weren't loaded yet.
    blocks: &'a [usize],
    /// The block index of `current`.
    block_index: usize,
    /// The remaining bits of the current block.
    current: usize,
}

impl Iterator for Iter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        while self.current == 0 {
            let (&first, rest) = self.blocks.split_first()?;
            self.current = first;
            self.blocks = rest;
            self.block_index += 1;
        }
        let bit = self.current.trailing_zeros() as usize;
        // Clear the lowest set bit.
        self.current &= self.current - 1;
        Some(((self.block_index - 1) * BLOCK_BITS + bit) as u32)
    }
}

#[cfg(test)]
mod tests;
