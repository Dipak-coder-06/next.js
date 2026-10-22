//! Compares [`OffsetBitset`] with [`FixedBitSet`] and [`RoaringBitmap`] on the operations that the
//! module graph algorithms use, over a range of bit densities.
//!
//! Every benchmark id is `<impl>/<pattern>-<universe>-<density>`:
//! - `universe` is the size of the bit range the set is drawn from (e.g. the number of chunk
//!   groups),
//! - `density` is the fraction of that range that is set, in per mille,
//! - `pattern` is `low` for a range starting at 0, or `high` for the same sets shifted to start at
//!   bit 1,000,000, where `OffsetBitset` skips the empty blocks below the range.
//!
//! `FixedBitSet` is used as is: it stores every bit from 0 up to the highest bit, so for the `high`
//! pattern it also stores the empty range below bit 1,000,000.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use fixedbitset::FixedBitSet;
use roaring::RoaringBitmap;
use turbo_bitset::OffsetBitset;

const UNIVERSES: [u32; 2] = [1_024, 16_384];
/// Densities in per mille.
const DENSITIES: [u32; 5] = [1, 10, 100, 500, 1_000];
const PATTERNS: [(&str, u32); 2] = [("low", 0), ("high", 1_000_000)];

/// A small deterministic linear congruential generator, so the inputs are the same on every run.
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

/// One benchmark input: two sets of the same density, each with its bits in ascending order.
struct Input {
    name: String,
    /// The bits of `a`, ascending.
    bits: Vec<u32>,
    /// A different set of the same density.
    other_bits: Vec<u32>,
    /// Every other bit of `a` (or nothing if `a` has a single bit), so `a` is a proper superset
    /// of it.
    subset_bits: Vec<u32>,
    /// `subset_bits` plus the bit after the last bit of `a`, so `a` is not a superset of it. The
    /// extra bit is the highest one, so a check that walks the bits in order finds it last.
    not_subset_bits: Vec<u32>,
    /// Indices to look up: half of them are bits of the set, the other half are random indices
    /// in the universe (which may or may not be in the set).
    probes: Vec<u32>,
}

fn random_bits(rng: &mut Lcg, base: u32, universe: u32, density: u32) -> Vec<u32> {
    let mut bits = (0..universe)
        .filter(|_| rng.next() % 1_000 < density)
        .map(|i| base + i)
        .collect::<Vec<_>>();
    if bits.is_empty() {
        bits.push(base + rng.next() % universe);
    }
    bits
}

fn inputs() -> Vec<Input> {
    let mut inputs = Vec::new();
    for (pattern, base) in PATTERNS {
        // The same seed for every pattern, so the `high` sets are exactly the `low` sets shifted.
        let mut rng = Lcg(42);
        for universe in UNIVERSES {
            for density in DENSITIES {
                let bits = random_bits(&mut rng, base, universe, density);
                let other_bits = random_bits(&mut rng, base, universe, density);
                let subset_bits = bits.iter().copied().skip(1).step_by(2).collect::<Vec<_>>();
                let mut not_subset_bits = subset_bits.clone();
                not_subset_bits.push(bits.last().unwrap() + 1);
                let probes = (0..1_024)
                    .map(|i| {
                        if i % 2 == 0 {
                            bits[rng.next() as usize % bits.len()]
                        } else {
                            base + rng.next() % universe
                        }
                    })
                    .collect();
                inputs.push(Input {
                    name: format!("{pattern}-{universe}-{density}"),
                    bits,
                    other_bits,
                    subset_bits,
                    not_subset_bits,
                    probes,
                });
            }
        }
    }
    inputs
}

fn offset(bits: &[u32]) -> OffsetBitset {
    bits.iter().copied().collect()
}

fn fixed(bits: &[u32]) -> FixedBitSet {
    let len = bits.iter().max().map_or(0, |&max| max as usize + 1);
    let mut set = FixedBitSet::with_capacity(len);
    for &bit in bits {
        set.insert(bit as usize);
    }
    set
}

fn roaring(bits: &[u32]) -> RoaringBitmap {
    bits.iter().copied().collect()
}

/// The setup and the measured operation of one implementation in [`bench_all`].
struct Impl<Setup, Run> {
    /// Builds the per-input state from the input, outside of the measurement.
    setup: Setup,
    /// The measured operation.
    run: Run,
}

/// Creates an [`Impl`]. (A function instead of a struct literal so that the closure argument
/// types are inferred.)
fn imp<S>(
    setup: impl Fn(&Input) -> S,
    run: impl Fn(&S),
) -> Impl<impl Fn(&Input) -> S, impl Fn(&S)> {
    Impl { setup, run }
}

/// Benchmarks the three implementations on every input.
fn bench_all<O, F, R>(
    c: &mut Criterion,
    name: &str,
    inputs: &[Input],
    offset: Impl<impl Fn(&Input) -> O, impl Fn(&O)>,
    fixed: Impl<impl Fn(&Input) -> F, impl Fn(&F)>,
    roaring: Impl<impl Fn(&Input) -> R, impl Fn(&R)>,
) {
    let mut group = c.benchmark_group(format!("bitset/{name}"));
    for input in inputs {
        let state = (offset.setup)(input);
        group.bench_function(BenchmarkId::new("OffsetBitset", &input.name), |b| {
            b.iter(|| (offset.run)(black_box(&state)))
        });
        let state = (fixed.setup)(input);
        group.bench_function(BenchmarkId::new("FixedBitSet", &input.name), |b| {
            b.iter(|| (fixed.run)(black_box(&state)))
        });
        let state = (roaring.setup)(input);
        group.bench_function(BenchmarkId::new("RoaringBitmap", &input.name), |b| {
            b.iter(|| (roaring.run)(black_box(&state)))
        });
    }
    group.finish();
}

/// Checks outside of any measurement that the implementations agree on the inputs.
fn validate(inputs: &[Input]) {
    let as_usize = |bits: &[u32]| bits.iter().map(|&b| b as usize).collect::<Vec<_>>();
    for input in inputs {
        let a = fixed(&input.bits);
        assert_eq!(a.ones().collect::<Vec<_>>(), as_usize(&input.bits));
        assert_eq!(
            input
                .bits
                .iter()
                .map(|&b| b as usize)
                .collect::<FixedBitSet>(),
            a
        );
        let mut inserted = FixedBitSet::new();
        for &bit in &shuffled(&input.bits) {
            inserted.grow_and_insert(bit as usize);
        }
        assert_eq!(inserted.ones().collect::<Vec<_>>(), as_usize(&input.bits));
        assert_eq!(a.count_ones(..), offset(&input.bits).count_ones());

        let mut union = a.clone();
        union.union_with(&fixed(&input.other_bits));
        let mut offset_union = offset(&input.bits);
        offset_union |= &offset(&input.other_bits);
        assert_eq!(
            union.ones().collect::<Vec<_>>(),
            offset_union.iter().map(|b| b as usize).collect::<Vec<_>>()
        );

        for (other, superset, proper) in [
            (&input.subset_bits, true, true),
            (&input.not_subset_bits, false, false),
            (&input.bits, true, false),
        ] {
            let b = fixed(other);
            assert_eq!(a.is_superset(&b), superset);
            assert_eq!(
                a.is_superset(&b) && a.count_ones(..) != b.count_ones(..),
                proper
            );
            assert_eq!(offset(&input.bits).is_superset(&offset(other)), superset);
            assert_eq!(
                offset(&input.bits).is_proper_superset(&offset(other)),
                proper
            );
            let (ra, rb) = (roaring(&input.bits), roaring(other));
            assert_eq!(ra.is_superset(&rb), superset);
            assert_eq!(ra.is_superset(&rb) && ra.len() != rb.len(), proper);
        }
    }
}

pub fn bench(c: &mut Criterion) {
    let inputs = inputs();
    validate(&inputs);

    // Building a set by inserting its bits one by one (in a fixed random order).
    bench_all(
        c,
        "insert",
        &inputs,
        imp(
            |i| shuffled(&i.bits),
            |bits| {
                let mut set = OffsetBitset::new();
                for &bit in bits {
                    black_box(set.insert(bit));
                }
                black_box(set);
            },
        ),
        imp(
            |i| shuffled(&i.bits),
            |bits| {
                let mut set = FixedBitSet::new();
                for &bit in bits {
                    let bit = bit as usize;
                    let added = !set.contains(bit);
                    set.grow_and_insert(bit);
                    black_box(added);
                }
                black_box(set);
            },
        ),
        imp(
            |i| shuffled(&i.bits),
            |bits| {
                let mut set = RoaringBitmap::new();
                for &bit in bits {
                    black_box(set.insert(bit));
                }
                black_box(set);
            },
        ),
    );

    // Building a set from an ascending iterator.
    bench_all(
        c,
        "from_iter",
        &inputs,
        imp(
            |i| i.bits.clone(),
            |bits| {
                black_box(bits.iter().copied().collect::<OffsetBitset>());
            },
        ),
        imp(
            |i| i.bits.clone(),
            |bits| {
                black_box(bits.iter().map(|&b| b as usize).collect::<FixedBitSet>());
            },
        ),
        imp(
            |i| i.bits.clone(),
            |bits| {
                black_box(bits.iter().copied().collect::<RoaringBitmap>());
            },
        ),
    );

    bench_all(
        c,
        "contains",
        &inputs,
        imp(
            |i| (offset(&i.bits), i.probes.clone()),
            |(set, probes)| {
                for &bit in probes {
                    black_box(set.contains(bit));
                }
            },
        ),
        imp(
            |i| (fixed(&i.bits), i.probes.clone()),
            |(set, probes)| {
                for &bit in probes {
                    black_box(set.contains(bit as usize));
                }
            },
        ),
        imp(
            |i| (roaring(&i.bits), i.probes.clone()),
            |(set, probes)| {
                for &bit in probes {
                    black_box(set.contains(bit));
                }
            },
        ),
    );

    bench_all(
        c,
        "iter",
        &inputs,
        imp(
            |i| offset(&i.bits),
            |set| {
                black_box(set.iter().fold(0u32, u32::wrapping_add));
            },
        ),
        imp(
            |i| fixed(&i.bits),
            |set| {
                black_box(
                    set.ones()
                        .fold(0u32, |sum, bit| sum.wrapping_add(bit as u32)),
                );
            },
        ),
        imp(
            |i| roaring(&i.bits),
            |set| {
                black_box(set.iter().fold(0u32, u32::wrapping_add));
            },
        ),
    );

    bench_all(
        c,
        "count_ones",
        &inputs,
        imp(
            |i| offset(&i.bits),
            |set| {
                black_box(set.count_ones());
            },
        ),
        imp(
            |i| fixed(&i.bits),
            |set| {
                black_box(set.count_ones(..));
            },
        ),
        imp(
            |i| roaring(&i.bits),
            |set| {
                black_box(set.len());
            },
        ),
    );

    bench_all(
        c,
        "clone",
        &inputs,
        imp(
            |i| offset(&i.bits),
            |set| {
                black_box(set.clone());
            },
        ),
        imp(
            |i| fixed(&i.bits),
            |set| {
                black_box(set.clone());
            },
        ),
        imp(
            |i| roaring(&i.bits),
            |set| {
                black_box(set.clone());
            },
        ),
    );

    // `a ⊇ b` where it holds, and where it fails because of the highest bit of `b`. The outcomes
    // are checked in `validate`.
    for (name, subset) in [
        (
            "superset_true",
            (|i: &Input| &i.subset_bits) as fn(&Input) -> &Vec<u32>,
        ),
        ("superset_false", |i: &Input| &i.not_subset_bits),
    ] {
        bench_all(
            c,
            name,
            &inputs,
            imp(
                |i| (offset(&i.bits), offset(subset(i))),
                |(a, b)| {
                    black_box(a.is_superset(b));
                },
            ),
            imp(
                |i| (fixed(&i.bits), fixed(subset(i))),
                |(a, b)| {
                    black_box(a.is_superset(b));
                },
            ),
            imp(
                |i| (roaring(&i.bits), roaring(subset(i))),
                |(a, b)| {
                    black_box(a.is_superset(b));
                },
            ),
        );
    }

    // `a ⊋ b`, for a proper superset, and for equal sets (where it fails). The outcomes are
    // checked in `validate`.
    for (name, other) in [
        (
            "proper_superset_true",
            (|i: &Input| &i.subset_bits) as fn(&Input) -> &Vec<u32>,
        ),
        ("proper_superset_false", |i: &Input| &i.bits),
    ] {
        bench_all(
            c,
            name,
            &inputs,
            imp(
                |i| (offset(&i.bits), offset(other(i))),
                |(a, b)| {
                    black_box(a.is_proper_superset(b));
                },
            ),
            imp(
                |i| (fixed(&i.bits), fixed(other(i))),
                |(a, b)| {
                    black_box(a.is_superset(b) && a.count_ones(..) != b.count_ones(..));
                },
            ),
            imp(
                |i| (roaring(&i.bits), roaring(other(i))),
                |(a, b)| {
                    black_box(a.is_superset(b) && a.len() != b.len());
                },
            ),
        );
    }

    // `a |= b` with a different set of the same density. The copy of `a` is made outside of the
    // measurement.
    let mut group = c.benchmark_group("bitset/union");
    for input in &inputs {
        let (a, b) = (offset(&input.bits), offset(&input.other_bits));
        group.bench_function(BenchmarkId::new("OffsetBitset", &input.name), |bencher| {
            bencher.iter_batched_ref(
                || a.clone(),
                |a| {
                    *a |= black_box(&b);
                    black_box(&*a);
                },
                BatchSize::SmallInput,
            )
        });
        let (a, b) = (fixed(&input.bits), fixed(&input.other_bits));
        group.bench_function(BenchmarkId::new("FixedBitSet", &input.name), |bencher| {
            bencher.iter_batched_ref(
                || a.clone(),
                |a| {
                    a.union_with(black_box(&b));
                    black_box(&*a);
                },
                BatchSize::SmallInput,
            )
        });
        let (a, b) = (roaring(&input.bits), roaring(&input.other_bits));
        group.bench_function(BenchmarkId::new("RoaringBitmap", &input.name), |bencher| {
            bencher.iter_batched_ref(
                || a.clone(),
                |a| {
                    *a |= black_box(&b);
                    black_box(&*a);
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    // Converting into a `RoaringBitmap`, the cost at the output boundary of the algorithms.
    let mut group = c.benchmark_group("bitset/to_roaring");
    for input in &inputs {
        let set = offset(&input.bits);
        group.bench_function(BenchmarkId::new("OffsetBitset", &input.name), |b| {
            b.iter(|| black_box(black_box(&set).to_roaring()))
        });
    }
    group.finish();
}

/// The bits in a fixed pseudo-random order.
fn shuffled(bits: &[u32]) -> Vec<u32> {
    let mut rng = Lcg(7);
    let mut bits = bits.to_vec();
    for i in (1..bits.len()).rev() {
        bits.swap(i, rng.next() as usize % (i + 1));
    }
    bits
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(200))
        .measurement_time(std::time::Duration::from_millis(500))
}

criterion_group! {
    name = bitset_benches;
    config = config();
    targets = bench
}
criterion_main!(bitset_benches);
