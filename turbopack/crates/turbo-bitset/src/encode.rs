//! The bincode encoding of [`OffsetBitset`].
//!
//! The encoding starts with an `i64` length header and has one of two forms, whichever is smaller:
//!
//! - `0`: the empty set.
//! - A positive length `len` (**bitmap form**): a `u32` byte offset followed by `len` raw bytes.
//!   Bit `j` of byte `i` is the bit `8 * (byte_offset + i) + j`. Leading and trailing zero bytes
//!   are trimmed, so the payload covers the minimal byte range.
//! - A negative length `-n` (**list form**): `n` bits in ascending order, the first one as a `u32`
//!   followed by `n - 1` `u32` differences to the previous bit (each at least 1).
//!
//! Integers use the encoder's integer encoding, so they are varints with the standard config. The
//! form is always chosen by comparing the sizes with varint integers, so the choice doesn't depend
//! on the config.

use bincode::{
    Decode, Encode,
    config::{Config, Endianness, IntEncoding},
    de::{Decoder, read::Reader},
    enc::{Encoder, write::Writer},
    error::{DecodeError, EncodeError},
    impl_borrow_decode,
};

use crate::{BLOCK_BITS, OffsetBitset};

/// Number of bytes per storage block.
const BLOCK_BYTES: usize = BLOCK_BITS / 8;

/// The number of bytes needed to store every `u32` bit.
const MAX_BYTES: u64 = 1 << 29;

/// The size of `value` as a bincode varint.
fn varint_len(value: u64) -> usize {
    match value {
        0..=250 => 1,
        251..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    }
}

/// The size of `value` as a bincode (zigzag) signed varint.
fn signed_varint_len(value: i64) -> usize {
    varint_len(((value << 1) ^ (value >> 63)) as u64)
}

/// The trimmed byte range of a set's bits: the byte offset of the first non-zero byte, the
/// trimmed blocks, and the number of zero bytes to skip at the start of the first and at the end
/// of the last block.
struct ByteRange<'a> {
    byte_offset: u64,
    blocks: &'a [usize],
    skip_front: usize,
    skip_back: usize,
}

impl ByteRange<'_> {
    fn len(&self) -> u64 {
        (self.blocks.len() * BLOCK_BYTES - self.skip_front - self.skip_back) as u64
    }
}

/// Collects the bytes of the bitmap form into a stack buffer, so that the writer is called once per
/// buffer instead of once per block. The writes have a constant size (and only the used bytes are
/// kept), so that they compile to plain stores instead of `memcpy` calls.
struct BufferedWriter<'a, W: Writer> {
    writer: &'a mut W,
    buffer: [u8; BUFFER_BYTES],
    len: usize,
}

const BUFFER_BYTES: usize = 256;

impl<'a, W: Writer> BufferedWriter<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            buffer: [0; BUFFER_BYTES],
            len: 0,
        }
    }

    /// Writes the first `len` bytes of `bytes`.
    #[inline]
    fn write_prefix<const N: usize>(
        &mut self,
        bytes: [u8; N],
        len: usize,
    ) -> Result<(), EncodeError> {
        debug_assert!(len <= N);
        if self.len + N > BUFFER_BYTES {
            self.flush()?;
        }
        self.buffer[self.len..self.len + N].copy_from_slice(&bytes);
        self.len += len;
        Ok(())
    }

    /// Writes the bytes `start..end` of `block`.
    #[inline]
    fn write_block(&mut self, block: usize, start: usize, end: usize) -> Result<(), EncodeError> {
        // Shift the skipped bytes out, so the bytes to write start at the beginning.
        let bytes = (block >> (start * 8)).to_le_bytes();
        self.write_prefix(bytes, end - start)
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        self.writer.write(&self.buffer[..self.len])?;
        self.len = 0;
        Ok(())
    }
}

/// How the encoder's config writes integers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IntFormat {
    /// Little-endian varints (the standard config).
    Varint,
    /// Little-endian fixed-width integers (the hash config).
    Fixint,
    /// Anything else, which is encoded through the encoder one integer at a time.
    Other,
}

fn int_format<C: Config>(config: &C) -> IntFormat {
    let little = matches!(config.endianness(), Endianness::Little);
    match config.int_encoding() {
        IntEncoding::Variable if little => IntFormat::Varint,
        IntEncoding::Fixed if little => IntFormat::Fixint,
        _ => IntFormat::Other,
    }
}

/// The number of bits in `blocks`, if it is below `limit`.
fn count_below(blocks: &[usize], limit: usize) -> Option<usize> {
    let mut count = 0;
    // Check the limit per chunk, so that the counting itself stays a simple loop.
    for chunk in blocks.chunks(16) {
        count += chunk.iter().map(|b| b.count_ones() as usize).sum::<usize>();
        if count >= limit {
            return None;
        }
    }
    Some(count)
}

/// Appends `value` as a little-endian bincode varint.
#[inline]
fn push_varint_u32(bytes: &mut Vec<u8>, value: u32) {
    match value {
        0..=250 => bytes.push(value as u8),
        251..=0xffff => {
            let [a, b] = (value as u16).to_le_bytes();
            bytes.extend_from_slice(&[251, a, b]);
        }
        _ => {
            let [a, b, c, d] = value.to_le_bytes();
            bytes.extend_from_slice(&[252, a, b, c, d]);
        }
    }
}

impl OffsetBitset {
    /// The trimmed byte range of the set, or `None` if it is empty.
    fn byte_range(&self) -> Option<ByteRange<'_>> {
        let (first_block, blocks) = self.trimmed();
        let (&first, &last) = (blocks.first()?, blocks.last()?);
        let skip_front = first.trailing_zeros() as usize / 8;
        let skip_back = last.leading_zeros() as usize / 8;
        Some(ByteRange {
            byte_offset: (first_block * BLOCK_BYTES + skip_front) as u64,
            blocks,
            skip_front,
            skip_back,
        })
    }

    /// The size of the list form (with varint integers), if it is smaller than `limit`.
    fn list_len_below(&self, count: usize, limit: usize) -> Option<usize> {
        let mut len = signed_varint_len(-(count as i64));
        let mut previous: Option<u32> = None;
        for bit in self.iter() {
            len += varint_len(match previous {
                None => bit,
                Some(previous) => bit - previous,
            } as u64);
            if len >= limit {
                return None;
            }
            previous = Some(bit);
        }
        Some(len)
    }

    /// The items of the list form (after the header) in `format`, if the list form (with varint
    /// integers) is smaller than `limit`. This computes the size and the bytes in one pass over
    /// the bits.
    fn list_bytes_below(&self, format: IntFormat, count: usize, limit: usize) -> Option<Vec<u8>> {
        let mut len = signed_varint_len(-(count as i64));
        let mut bytes = Vec::with_capacity(match format {
            IntFormat::Fixint => count * size_of::<u32>(),
            _ => limit - len,
        });
        let mut previous: Option<u32> = None;
        for bit in self.iter() {
            let value = match previous {
                None => bit,
                Some(previous) => bit - previous,
            };
            len += varint_len(value as u64);
            if len >= limit {
                return None;
            }
            match format {
                IntFormat::Fixint => bytes.extend_from_slice(&value.to_le_bytes()),
                _ => push_varint_u32(&mut bytes, value),
            }
            previous = Some(bit);
        }
        Some(bytes)
    }
}

impl Encode for OffsetBitset {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let Some(range) = self.byte_range() else {
            return 0i64.encode(encoder);
        };
        let bitmap_len = range.len();
        let bitmap_size = signed_varint_len(bitmap_len as i64)
            + varint_len(range.byte_offset)
            + bitmap_len as usize;
        // Every bit takes at least one byte in the list form, so only try it if that lower bound
        // doesn't already rule it out. Counting stops early once it does, which is quick for
        // dense sets.
        if let Some(count) = count_below(range.blocks, bitmap_size)
            && signed_varint_len(-(count as i64)) + count < bitmap_size
        {
            match int_format(encoder.config()) {
                IntFormat::Other => {
                    if self.list_len_below(count, bitmap_size).is_some() {
                        (-(count as i64)).encode(encoder)?;
                        let mut previous: Option<u32> = None;
                        for bit in self.iter() {
                            match previous {
                                None => bit.encode(encoder)?,
                                Some(previous) => (bit - previous).encode(encoder)?,
                            }
                            previous = Some(bit);
                        }
                        return Ok(());
                    }
                }
                format => {
                    if let Some(bytes) = self.list_bytes_below(format, count, bitmap_size) {
                        (-(count as i64)).encode(encoder)?;
                        return encoder.writer().write(&bytes);
                    }
                }
            }
        }

        (bitmap_len as i64).encode(encoder)?;
        // The byte offset is below `MAX_BYTES`, so it fits into `u32`.
        (range.byte_offset as u32).encode(encoder)?;
        let last = range.blocks.len() - 1;
        let mut writer = BufferedWriter::new(encoder.writer());
        for (index, &block) in range.blocks.iter().enumerate() {
            let start = if index == 0 { range.skip_front } else { 0 };
            let end = if index == last {
                BLOCK_BYTES - range.skip_back
            } else {
                BLOCK_BYTES
            };
            writer.write_block(block, start, end)?;
        }
        writer.flush()
    }
}

impl<Context> Decode<Context> for OffsetBitset {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = i64::decode(decoder)?;
        match len {
            0 => Ok(Self::new()),
            1.. => decode_bitmap(decoder, len as u64),
            _ => decode_list(decoder, len.unsigned_abs()),
        }
    }
}

impl_borrow_decode!(OffsetBitset);

fn decode_bitmap<D: Decoder>(decoder: &mut D, len: u64) -> Result<OffsetBitset, DecodeError> {
    let byte_offset = u32::decode(decoder)? as u64;
    if len > MAX_BYTES || byte_offset + len > MAX_BYTES {
        return Err(DecodeError::Other(
            "OffsetBitset bitmap extends beyond the u32 range",
        ));
    }
    let end_byte = (byte_offset + len) as usize;
    let byte_offset = byte_offset as usize;
    let first_block = byte_offset / BLOCK_BYTES;
    let end_block = end_byte.div_ceil(BLOCK_BYTES);
    let block_count = end_block - first_block;
    decoder.claim_bytes_read(block_count * BLOCK_BYTES)?;
    let mut blocks = Vec::with_capacity(block_count);
    for index in 0..block_count {
        let block_start = (first_block + index) * BLOCK_BYTES;
        let start = byte_offset.saturating_sub(block_start);
        let end = (end_byte - block_start).min(BLOCK_BYTES);
        let mut bytes = [0; BLOCK_BYTES];
        decoder.reader().read(&mut bytes[start..end])?;
        blocks.push(usize::from_le_bytes(bytes));
    }
    Ok(OffsetBitset {
        offset: first_block,
        blocks,
    })
}

fn decode_list<D: Decoder>(decoder: &mut D, count: u64) -> Result<OffsetBitset, DecodeError> {
    if count > 1 << 32 {
        return Err(DecodeError::Other(
            "OffsetBitset list has more items than the u32 range",
        ));
    }
    // Set the bits directly in the storage. The bits are ascending, so it only grows at the back,
    // and the bytes of every new block are claimed from the decoder's limit before allocating
    // them (each item takes at least one byte of input, but a single item can add many blocks).
    let first = u32::decode(decoder)?;
    let offset = first as usize / BLOCK_BITS;
    let mut blocks: Vec<usize> = Vec::new();
    let mut set_bit = |decoder: &mut D, bit: u32| -> Result<(), DecodeError> {
        let index = bit as usize / BLOCK_BITS - offset;
        if index >= blocks.len() {
            decoder.claim_bytes_read((index + 1 - blocks.len()) * BLOCK_BYTES)?;
            blocks.resize(index + 1, 0);
        }
        blocks[index] |= 1 << (bit as usize % BLOCK_BITS);
        Ok(())
    };
    set_bit(decoder, first)?;
    let mut previous = first;
    for _ in 1..count {
        let difference = u32::decode(decoder)?;
        if difference == 0 {
            return Err(DecodeError::Other(
                "OffsetBitset list is not strictly ascending",
            ));
        }
        previous = previous.checked_add(difference).ok_or(DecodeError::Other(
            "OffsetBitset list exceeds the u32 range",
        ))?;
        set_bit(decoder, previous)?;
    }
    // The storage grew with amortized reallocations, so drop the spare capacity.
    blocks.shrink_to_fit();
    Ok(OffsetBitset { offset, blocks })
}
