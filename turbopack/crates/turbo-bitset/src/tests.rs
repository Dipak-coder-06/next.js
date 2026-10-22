use std::{collections::BTreeSet, hash::BuildHasher};

use roaring::RoaringBitmap;
use rustc_hash::FxBuildHasher;

use crate::OffsetBitset;

/// Checks every read operation of `set` against the reference `expected`.
fn check(set: &OffsetBitset, expected: &BTreeSet<u32>) {
    assert_eq!(
        set.iter().collect::<Vec<_>>(),
        expected.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(set.count_ones(), expected.len());
    assert_eq!(set.is_zero(), expected.is_empty());
    for &bit in expected {
        assert!(set.contains(bit), "missing {bit}");
    }
    // Neighbors of every bit, plus some fixed probes.
    let probes = expected
        .iter()
        .flat_map(|&b| [b.wrapping_sub(1), b.wrapping_add(1), b ^ 64])
        .chain([0, 1, 63, 64, 65, 127, 128, 1000, u32::MAX - 1, u32::MAX]);
    for bit in probes {
        assert_eq!(
            set.contains(bit),
            expected.contains(&bit),
            "contains({bit})"
        );
    }
    let roaring = set.to_roaring();
    assert_eq!(
        roaring.iter().collect::<Vec<_>>(),
        expected.iter().copied().collect::<Vec<_>>()
    );
    // Same internal representation as inserting the bits.
    assert_eq!(roaring, expected.iter().copied().collect::<RoaringBitmap>());
    // Rebuilding from the bits gives an equal set.
    let rebuilt = expected.iter().copied().collect::<OffsetBitset>();
    assert_eq!(&rebuilt, set);
    assert_eq!(hash(&rebuilt), hash(set));
}

fn hash(set: &OffsetBitset) -> u64 {
    FxBuildHasher.hash_one(set)
}

/// The pointer and capacity of the storage, to check whether it was reallocated.
fn storage(set: &OffsetBitset) -> (*const usize, usize) {
    (set.blocks.as_ptr(), set.blocks.capacity())
}

fn from_bits(bits: &[u32]) -> (OffsetBitset, BTreeSet<u32>) {
    (
        bits.iter().copied().collect(),
        bits.iter().copied().collect(),
    )
}

#[test]
fn empty() {
    let set = OffsetBitset::new();
    check(&set, &BTreeSet::new());
    assert_eq!(set, OffsetBitset::default());
    assert!(set.is_superset(&OffsetBitset::new()));
    assert!(!set.is_proper_superset(&OffsetBitset::new()));
}

#[test]
fn insert_returns_whether_added() {
    let mut set = OffsetBitset::new();
    assert!(set.insert(5));
    assert!(!set.insert(5));
    assert!(set.insert(1000));
    assert!(!set.insert(1000));
    // Below the current offset (front growth).
    assert!(set.insert(0));
    assert!(!set.insert(0));
    check(&set, &[0, 5, 1000].into());
}

#[test]
fn block_boundaries() {
    let block = usize::BITS;
    for bits in [
        &[0][..],
        &[block - 1],
        &[block],
        &[block - 1, block],
        &[2 * block - 1, 2 * block],
        &[0, block - 1, block, 3 * block - 1],
        &[u32::MAX],
        &[u32::MAX - block, u32::MAX],
    ] {
        let (set, expected) = from_bits(bits);
        check(&set, &expected);
    }
}

#[test]
fn front_and_back_growth() {
    let mut set = OffsetBitset::new();
    let mut expected = BTreeSet::new();
    // High first, then progressively lower and higher bits.
    for bit in [10_000, 9_000, 20_000, 64, 63, 0, 100_000, 99_999] {
        assert_eq!(set.insert(bit), expected.insert(bit));
        check(&set, &expected);
    }
}

#[test]
fn front_growth_reuses_capacity() {
    let block = usize::BITS;
    let mut set = OffsetBitset::new();
    let mut expected = BTreeSet::new();
    for bit in [100 * block + 3, 101 * block] {
        set.insert(bit);
        expected.insert(bit);
    }
    // Leave room for 8 more blocks.
    set.blocks.reserve(8);
    let (ptr, capacity) = storage(&set);
    assert!(capacity >= 10);

    // Growing at the front within the capacity doesn't reallocate.
    for bit in [99 * block + 7, 96 * block, 99 * block - 1, 95 * block + 1] {
        assert_eq!(set.insert(bit), expected.insert(bit));
        check(&set, &expected);
        assert_eq!(storage(&set), (ptr, capacity));
    }
    // Growing at the front and the back at once within the capacity doesn't reallocate either.
    let other = [94 * block, 102 * block + 5]
        .into_iter()
        .collect::<OffsetBitset>();
    set |= &other;
    expected.extend(other.iter());
    check(&set, &expected);
    assert_eq!(storage(&set), (ptr, capacity));

    // Growing beyond the reserved capacity keeps all bits.
    let bit = 10 * block + 1;
    assert!(set.insert(bit));
    expected.insert(bit);
    check(&set, &expected);
}

#[test]
fn front_and_back_growth_in_one_step() {
    let block = usize::BITS;
    for reserve in [0, 2, 100] {
        let mut set = [10 * block, 11 * block - 1]
            .into_iter()
            .collect::<OffsetBitset>();
        set.blocks.reserve(reserve);
        let mut expected = set.iter().collect::<BTreeSet<_>>();
        // A union that extends the range below and above.
        let other = [3 * block + 2, 10 * block, 20 * block + 9];
        set |= &other.into_iter().collect::<OffsetBitset>();
        expected.extend(other);
        check(&set, &expected);
        // Extending below and above.
        let more = [block, 2 * block - 1, 30 * block];
        set.extend(more);
        expected.extend(more);
        check(&set, &expected);
    }
}

#[test]
fn union() {
    let cases: &[(&[u32], &[u32])] = &[
        (&[], &[]),
        (&[], &[1, 2, 3]),
        (&[1, 2, 3], &[]),
        // Disjoint, other below.
        (&[5_000, 5_001], &[1, 2]),
        // Disjoint, other above.
        (&[1, 2], &[5_000, 5_001]),
        // Overlapping ranges.
        (&[100, 200, 300], &[200, 250, 400]),
        // Other inside self's range.
        (&[0, 10_000], &[5_000]),
        // Self inside other's range.
        (&[5_000], &[0, 10_000]),
        // Equal.
        (&[7, 700, 7_000], &[7, 700, 7_000]),
    ];
    for &(a, b) in cases {
        let (mut set, mut expected) = from_bits(a);
        let (other, other_expected) = from_bits(b);
        set |= &other;
        expected.extend(other_expected.iter().copied());
        check(&set, &expected);
        // `other` is unchanged.
        check(&other, &other_expected);

        let (mut set2, _) = from_bits(a);
        set2.union_with(&other);
        assert_eq!(set2, set);
    }
}

#[test]
fn superset() {
    let cases: &[(&[u32], &[u32])] = &[
        (&[], &[]),
        (&[1], &[]),
        (&[], &[1]),
        (&[1, 2, 3], &[1, 2, 3]),
        (&[1, 2, 3], &[2]),
        (&[2], &[1, 2, 3]),
        (&[0, 10_000], &[10_000]),
        (&[10_000], &[0, 10_000]),
        (&[5_000, 5_001], &[1, 2]),
        (&[1, 2], &[5_000, 5_001]),
        (&[0, 64, 128, 192], &[64, 128]),
        (&[64, 128], &[0, 64, 128, 192]),
        (&[64, 128], &[64, 129]),
    ];
    for &(a, b) in cases {
        let (set, expected) = from_bits(a);
        let (other, other_expected) = from_bits(b);
        let is_superset = expected.is_superset(&other_expected);
        assert_eq!(set.is_superset(&other), is_superset, "{a:?} ⊇ {b:?}");
        assert_eq!(
            set.is_proper_superset(&other),
            is_superset && expected != other_expected,
            "{a:?} ⊋ {b:?}"
        );
        assert_eq!(
            set.is_subset(&other),
            expected.is_subset(&other_expected),
            "{a:?} ⊆ {b:?}"
        );
    }
}

#[test]
fn equality_and_hash_ignore_layout() {
    // Same bits, built so that the storage has different offsets and capacities.
    let mut a = OffsetBitset::new();
    a.insert(1_000);
    a.insert(1_100);

    let mut b = OffsetBitset::new();
    b.insert(0);
    b.insert(1_000);
    b.insert(1_100);
    b.insert(50_000);
    assert_ne!(a, b);

    // `c` was built in a different order and unioned with an empty set.
    let mut c = [1_100, 1_000].into_iter().collect::<OffsetBitset>();
    c |= &OffsetBitset::new();
    let mut wide = OffsetBitset::new();
    wide.insert(1_000);
    wide.insert(1_100);
    wide |= &a;
    assert_eq!(a, c);
    assert_eq!(a, wide);
    assert_eq!(hash(&a), hash(&c));
    assert_eq!(hash(&a), hash(&wide));

    // Sets that only differ in where a high bit lives.
    let x = [64].into_iter().collect::<OffsetBitset>();
    let y = [0].into_iter().collect::<OffsetBitset>();
    assert_ne!(x, y);
    assert_ne!(OffsetBitset::new(), y);
}

#[test]
fn equal_sets_with_different_storage_ranges() {
    // Build a set by hand whose storage has extra zero blocks before and after its bits.
    let a = [3_000, 3_001].into_iter().collect::<OffsetBitset>();
    let block = usize::BITS as usize;
    let offset = 3_000 / block - 2;
    let mut blocks = vec![0usize; 6];
    for bit in [3_000, 3_001] {
        let bit = bit - offset * block;
        blocks[bit / block] |= 1 << (bit % block);
    }
    let wide = OffsetBitset { offset, blocks };
    check(&wide, &[3_000, 3_001].into());
    assert_eq!(a, wide);
    assert_eq!(hash(&a), hash(&wide));
    assert!(a.is_superset(&wide));
    assert!(wide.is_superset(&a));
    assert!(!a.is_proper_superset(&wide));

    // An empty set with allocated storage equals the default empty set.
    let empty_wide = OffsetBitset {
        offset: 10,
        blocks: vec![0; 3],
    };
    check(&empty_wide, &BTreeSet::new());
    assert_eq!(empty_wide, OffsetBitset::new());
    assert_eq!(hash(&empty_wide), hash(&OffsetBitset::new()));
    assert!(empty_wide.is_superset(&OffsetBitset::new()));
    assert!(a.is_proper_superset(&empty_wide));

    // Unions into and from wide storage.
    let mut u = wide.clone();
    u |= &[5].into_iter().collect();
    check(&u, &[5, 3_000, 3_001].into());
    let mut v = [5].into_iter().collect::<OffsetBitset>();
    v |= &wide;
    assert_eq!(u, v);
}

#[test]
fn extend() {
    let mut set = OffsetBitset::new();
    set.extend([900, 3, 70_000, 3]);
    check(&set, &[3, 900, 70_000].into());
}

#[test]
fn to_roaring() {
    // Bits spanning several containers, and the highest possible bits. (A set with both bit 0 and
    // `u32::MAX` would allocate 512 MiB, as the set is meant for bounded ranges.)
    for bits in [
        &[0, 1, 63, 64, 65_535, 65_536, 1 << 20][..],
        &[u32::MAX - 70_000, u32::MAX - 1, u32::MAX],
    ] {
        let (set, expected) = from_bits(bits);
        let roaring = set.to_roaring();
        assert_eq!(roaring, expected.iter().copied().collect::<RoaringBitmap>());
    }
}

#[test]
fn to_roaring_container_types() {
    // A dense container (stored as a bitmap by roaring), a sparse one (stored as an array), a
    // container at exactly the array limit, and a range that starts in the middle of a container.
    let bits = (0..5_000)
        .chain(70_000..70_010)
        .chain((2 << 16)..(2 << 16) + 4_096)
        .chain((3 << 16) + 1_000..(3 << 16) + 9_000)
        .collect::<Vec<u32>>();
    let (set, expected) = from_bits(&bits);
    check(&set, &expected);
    let roaring = set.to_roaring();
    let inserted = bits.iter().copied().collect::<RoaringBitmap>();
    assert_eq!(roaring, inserted);
    // The serialization (which `RoaringBitmapWrapper` uses for hashing) is identical too.
    let mut a = Vec::new();
    let mut b = Vec::new();
    roaring.serialize_into(&mut a).unwrap();
    inserted.serialize_into(&mut b).unwrap();
    assert_eq!(a, b);
}

/// A small deterministic linear congruential generator, so the test needs no random number crate.
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

#[test]
fn differential() {
    let mut rng = Lcg(42);
    for round in 0..200 {
        // Vary the universe, so that the sets are sometimes dense and sometimes sparse.
        let universe = [64, 200, 1_000, 20_000, 1 << 20][round % 5];
        let base = if round % 3 == 0 {
            rng.next() % 100_000
        } else {
            0
        };
        let gen_set = |rng: &mut Lcg| {
            let count = rng.next() % 50;
            let bits = (0..count)
                .map(|_| base + rng.next() % universe)
                .collect::<Vec<_>>();
            from_bits(&bits)
        };
        let (mut a, mut a_expected) = gen_set(&mut rng);
        let (b, b_expected) = gen_set(&mut rng);
        check(&a, &a_expected);
        check(&b, &b_expected);

        let is_superset = a_expected.is_superset(&b_expected);
        assert_eq!(a.is_superset(&b), is_superset);
        assert_eq!(
            a.is_proper_superset(&b),
            is_superset && a_expected != b_expected
        );
        assert_eq!(a.is_subset(&b), a_expected.is_subset(&b_expected));
        assert_eq!(a == b, a_expected == b_expected);

        a |= &b;
        a_expected.extend(b_expected.iter().copied());
        check(&a, &a_expected);
        assert!(a.is_superset(&b));
    }
}

#[test]
fn differential_dense() {
    // Dense sets over a few blocks, where most blocks are full and the sets often overlap or
    // contain each other.
    let mut rng = Lcg(7);
    for round in 0..300 {
        let universe = [64, 128, 300, 1_000][round % 4];
        let base = [0, 64, 1_000][round % 3];
        let gen_set = |rng: &mut Lcg, density: u32| {
            let bits = (0..universe)
                .filter(|_| rng.next() % 100 < density)
                .map(|i| base + i)
                .collect::<Vec<_>>();
            from_bits(&bits)
        };
        let density = [50, 90, 100][round % 3];
        let (mut a, mut a_expected) = gen_set(&mut rng, density);
        let (b, b_expected) = if round % 2 == 0 {
            // A subset of `a`.
            let bits = a_expected
                .iter()
                .copied()
                .filter(|_| rng.next().is_multiple_of(2))
                .collect::<Vec<_>>();
            from_bits(&bits)
        } else {
            gen_set(&mut rng, density)
        };
        check(&a, &a_expected);
        check(&b, &b_expected);

        let is_superset = a_expected.is_superset(&b_expected);
        assert_eq!(a.is_superset(&b), is_superset);
        assert_eq!(
            a.is_proper_superset(&b),
            is_superset && a_expected != b_expected
        );
        assert_eq!(a.is_subset(&b), a_expected.is_subset(&b_expected));
        assert_eq!(b.is_subset(&a), b_expected.is_subset(&a_expected));
        assert_eq!(a == b, a_expected == b_expected);

        a |= &b;
        a_expected.extend(b_expected.iter().copied());
        check(&a, &a_expected);
    }
}
