//! Measures the in-memory storage and serialized size of `OffsetBitset` and `RoaringBitmap` for
//! different universe sizes, densities and bit distributions.
//!
//! Storage is measured with a counting global allocator: the bytes requested from the allocator
//! by the set (its heap allocations), plus `size_of` the set itself. Allocator bookkeeping and
//! size-class rounding are not included.
//!
//! Serialized sizes are measured by actually encoding (and decoding again) with bincode's
//! standard config, the config Turbopack persists with:
//! - `RoaringBitmap` through its serde implementation (the portable Roaring format as a
//!   length-prefixed byte string), which is how `RoaringBitmapWrapper` is encoded.
//! - `OffsetBitset` has no serialization. As a hypothetical encoding, its `(offset, words)` as
//!   `(u64, Vec<u64>)`, once with bincode's standard varint integers and once with fixed-width
//!   integers.
//!
//! Run with `cargo run --release -p turbo-bitset --example sizes`. Prints CSV.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("this measurement assumes 64-bit `usize` blocks");

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use roaring::RoaringBitmap;
use turbo_bitset::OffsetBitset;

struct Counting;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap bytes allocated by `build` that are still live when it returns, plus the inline size.
fn storage<T>(build: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATED.load(Ordering::Relaxed);
    let value = build();
    let after = ALLOCATED.load(Ordering::Relaxed);
    (value, after - before + size_of::<T>())
}

/// Drops `value` and checks that this frees exactly the heap bytes `storage` measured for it.
fn check_freed<T>(value: T, measured: usize) {
    let before = ALLOCATED.load(Ordering::Relaxed);
    drop(value);
    let freed = before - ALLOCATED.load(Ordering::Relaxed);
    assert_eq!(freed + size_of::<T>(), measured, "unbalanced allocation");
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Bits of a set: `uniform` picks every bit independently with probability `density`;
/// `clustered` places runs of 64 consecutive bits at random positions until the density is
/// reached.
fn bits(rng: &mut Lcg, universe: u32, per_million: u32, clustered: bool) -> Vec<u32> {
    let mut bits = if clustered {
        let target = (universe as u64 * per_million as u64 / 1_000_000).max(1) as usize;
        let mut set = std::collections::BTreeSet::new();
        while set.len() < target {
            let start = rng.next() % universe;
            for bit in start..(start + 64).min(universe) {
                if set.len() == target {
                    break;
                }
                set.insert(bit);
            }
        }
        set.into_iter().collect::<Vec<_>>()
    } else {
        (0..universe)
            .filter(|_| rng.next() % 1_000_000 < per_million)
            .collect::<Vec<_>>()
    };
    if bits.is_empty() {
        bits.push(rng.next() % universe);
    }
    bits
}

fn shuffled(rng: &mut Lcg, bits: &[u32]) -> Vec<u32> {
    let mut bits = bits.to_vec();
    for i in (1..bits.len()).rev() {
        bits.swap(i, rng.next() as usize % (i + 1));
    }
    bits
}

fn main() {
    let varint = bincode::config::standard();
    let fixint = bincode::config::standard().with_fixed_int_encoding();
    println!(
        "distribution,universe,density_ppm,bits,actual_density_ppm,offset_sorted,offset_random,\
         roaring_sorted,roaring_random,offset_words,offset_ser_fixed,offset_ser_varint,\
         roaring_portable,roaring_ser,roaring_ser_optimized"
    );
    for clustered in [false, true] {
        for universe in [1_024u32, 16_384, 65_536, 1_048_576] {
            for per_million in [100u32, 1_000, 10_000, 100_000, 500_000, 1_000_000] {
                let mut rng = Lcg(42);
                let bits = bits(&mut rng, universe, per_million, clustered);
                let random_order = shuffled(&mut rng, &bits);

                let (offset_sorted, offset_sorted_bytes) =
                    storage(|| bits.iter().copied().collect::<OffsetBitset>());
                let (offset_random, offset_random_bytes) = storage(|| {
                    let mut set = OffsetBitset::new();
                    for &bit in &random_order {
                        set.insert(bit);
                    }
                    set
                });
                let (roaring_sorted, roaring_sorted_bytes) =
                    storage(|| RoaringBitmap::from_sorted_iter(bits.iter().copied()).unwrap());
                let (roaring_random, roaring_random_bytes) = storage(|| {
                    let mut set = RoaringBitmap::new();
                    for &bit in &random_order {
                        set.insert(bit);
                    }
                    set
                });
                assert_eq!(offset_sorted, offset_random);
                assert_eq!(roaring_sorted, roaring_random);
                assert_eq!(offset_sorted.to_roaring(), roaring_sorted);

                // The block-aligned word range of the set: from the lowest to the highest word.
                let first_word = (bits[0] / 64) as u64;
                let last_word = (*bits.last().unwrap() / 64) as u64;
                let words = (last_word - first_word + 1) as usize;
                let mut word_values = vec![0u64; words];
                for &bit in &bits {
                    word_values[(bit as u64 / 64 - first_word) as usize] |= 1 << (bit % 64);
                }
                let encoded: (u64, Vec<u64>) = (first_word, word_values);
                let offset_varint = bincode::encode_to_vec(&encoded, varint).unwrap();
                let offset_fixint = bincode::encode_to_vec(&encoded, fixint).unwrap();
                for (bytes, config) in [(&offset_varint, false), (&offset_fixint, true)] {
                    let decoded: (u64, Vec<u64>) = if config {
                        bincode::decode_from_slice(bytes, fixint).unwrap().0
                    } else {
                        bincode::decode_from_slice(bytes, varint).unwrap().0
                    };
                    assert_eq!(decoded, encoded);
                    let decoded_bits = decoded
                        .1
                        .iter()
                        .enumerate()
                        .flat_map(|(index, &word)| {
                            (0..64)
                                .filter(move |bit| word & (1 << bit) != 0)
                                .map(move |bit| ((decoded.0 + index as u64) * 64 + bit) as u32)
                        })
                        .collect::<OffsetBitset>();
                    assert_eq!(decoded_bits, offset_sorted);
                }
                assert_eq!(offset_fixint.len(), 16 + 8 * words);

                // The portable Roaring format, and the same through bincode's serde support.
                let mut portable = Vec::new();
                roaring_sorted.serialize_into(&mut portable).unwrap();
                assert_eq!(portable.len(), roaring_sorted.serialized_size());
                let roaring_ser = bincode::serde::encode_to_vec(&roaring_sorted, varint).unwrap();
                let decoded: RoaringBitmap =
                    bincode::serde::decode_from_slice(&roaring_ser, varint)
                        .unwrap()
                        .0;
                assert_eq!(decoded, roaring_sorted);
                let mut optimized = roaring_sorted.clone();
                optimized.optimize();
                let optimized_ser = bincode::serde::encode_to_vec(&optimized, varint).unwrap();
                let decoded: RoaringBitmap =
                    bincode::serde::decode_from_slice(&optimized_ser, varint)
                        .unwrap()
                        .0;
                assert_eq!(decoded, roaring_sorted);

                println!(
                    "{},{universe},{per_million},{},{},{offset_sorted_bytes},\
                     {offset_random_bytes},{roaring_sorted_bytes},{roaring_random_bytes},{words},\
                     {},{},{},{},{}",
                    if clustered { "clustered" } else { "uniform" },
                    bits.len(),
                    bits.len() as u64 * 1_000_000 / universe as u64,
                    offset_fixint.len(),
                    offset_varint.len(),
                    portable.len(),
                    roaring_ser.len(),
                    optimized_ser.len(),
                );
                check_freed(offset_sorted, offset_sorted_bytes);
                check_freed(offset_random, offset_random_bytes);
                check_freed(roaring_sorted, roaring_sorted_bytes);
                check_freed(roaring_random, roaring_random_bytes);
            }
        }
    }
}
