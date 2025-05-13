# lzutf8.py – compact, Python clone of the JS LZUTF8 codec
# (ty o3)
from __future__ import annotations
from typing import List, Tuple, Union
import base64

ByteLike = Union[bytes, bytearray, memoryview]

# ---------------------------------------------------------------------------
# UTF-8  / Base-64 helpers
# ---------------------------------------------------------------------------
def encode_utf8(s: str) -> bytes:
    return s.encode('utf-8')

def decode_utf8(b: ByteLike) -> str:
    return bytes(b).decode('utf-8')

def encode_base64(b: ByteLike) -> str:
    return base64.b64encode(bytes(b)).decode('ascii')

def decode_base64(s: str) -> bytes:
    return base64.b64decode(s.encode('ascii'))

# ---------------------------------------------------------------------------
# Small hash table – enough for Python speed
# ---------------------------------------------------------------------------
class _HashTable:
    _MAX_BUCKET = 64           # keep newest 64 entries like the JS code

    def __init__(self, size: int = 65537) -> None:       # same prime as JS
        self._buckets: List[List[int]] = [[] for _ in range(size)]

    def _bucket(self, a: int, b: int, c: int, d: int) -> List[int]:
        idx = (a * 7_880_599 + b * 39_601 + c * 199 + d) % len(self._buckets)
        return self._buckets[idx]

    # ------------------------------------------------------------------
    def add(self, pos: int, window: memoryview) -> None:
        bucket = self._bucket(window[0], window[1], window[2], window[3])
        bucket.append(pos)
        if len(bucket) > self._MAX_BUCKET:
            del bucket[: len(bucket) // 2]        # drop oldest half

    def candidates(self, window: memoryview) -> List[int]:
        return self._bucket(window[0], window[1], window[2], window[3])

# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------
# Replace the whole Compressor class in lzutf8.py with this version
class Compressor:
    """
    Much faster pure-Python encoder:
    – For every position we ask CPython’s highly-optimised bytes.rfind()
      to locate the newest occurrence of the 4-byte prefix in the
      32 KiB history window (same 32 767 distance limit as the JS code).
    – If found, we extend the match up to 31 bytes and emit a back-pointer.
    – Otherwise we output the literal byte.
    """

    _MIN_LEN  = 4
    _MAX_LEN  = 31
    _MAX_DIST = 32_767         # sliding-window size

    # ------------------------------------------------------------------
    def compress(self, src: Union[str, ByteLike]) -> bytes:
        if isinstance(src, str):
            buf = encode_utf8(src)
        elif isinstance(src, (bytes, bytearray, memoryview)):
            buf = bytes(src)
        else:
            raise TypeError('compress: input must be str or bytes-like')

        return bytes(self._encode_block(buf))

    # ------------------------------------------------------------------
    def _encode_block(self, buf: bytes) -> bytearray:
        out   = bytearray()
        size  = len(buf)
        pos   = 0

        while pos < size:
            # ----------------------------------------------------------
            #  Try to find a match of at least 4 bytes
            # ----------------------------------------------------------
            match_len  = 0
            match_dist = 0

            remaining = size - pos
            if remaining >= self._MIN_LEN:
                win_lo   = max(0, pos - self._MAX_DIST)
                window   = buf[win_lo:pos]          # bytes copy but cheap
                pattern  = buf[pos:pos + self._MIN_LEN]

                idx = window.rfind(pattern)         # C implementation – fast
                if idx != -1:
                    dist = pos - (win_lo + idx)
                    # grow the match
                    max_len = min(self._MAX_LEN, remaining)
                    length  = self._MIN_LEN
                    while (length < max_len and
                           buf[pos + length] == buf[pos + length - dist]):
                        length += 1
                    match_len, match_dist = length, dist

            if match_len:                           # emit pointer token
                if match_dist < 128:                # 2-byte token
                    out.append(0xC0 | match_len)
                    out.append(match_dist)
                else:                               # 3-byte token
                    out.append(0xE0 | match_len)
                    out.extend([(match_dist >> 8) & 0xFF, match_dist & 0xFF])

                pos += match_len
                continue

            # ----------------------------------------------------------
            #  No match – emit raw byte
            # ----------------------------------------------------------
            out.append(buf[pos])
            pos += 1

        return out

# ---------------------------------------------------------------------------
# Decompressor
# ---------------------------------------------------------------------------
class Decompressor:
    def decompress(self, src: Union[str, ByteLike]) -> bytes:
        if isinstance(src, str):                 # assume Base-64 text
            data = decode_base64(src)
        elif isinstance(src, (bytes, bytearray, memoryview)):
            data = bytes(src)
        else:
            raise TypeError('decompress: input must be str or bytes-like')

        return bytes(self._decode_block(memoryview(data)))

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_block(inp: memoryview) -> bytearray:
        out = bytearray()
        i   = 0
        n   = len(inp)

        while i < n:
            b0 = inp[i]
            if b0 >> 6 != 0b11:              # literal byte
                out.append(b0)
                i += 1
                continue

            length = b0 & 0b11111
            if (b0 >> 5) == 0b110:           # 2-byte pointer
                if i + 1 >= n:
                    raise ValueError('Truncated stream')
                dist = inp[i + 1]
                i += 2
            else:                            # 3-byte pointer
                if i + 2 >= n:
                    raise ValueError('Truncated stream')
                dist = (inp[i + 1] << 8) | inp[i + 2]
                i += 3

            if dist == 0 or dist > len(out):
                raise ValueError('Invalid back-reference')

            start = len(out) - dist
            for _ in range(length):          # cope with overlap
                out.append(out[start])
                start += 1

        return out

# ---------------------------------------------------------------------------
# Convenience one-shot wrappers
# ---------------------------------------------------------------------------
_compressor   = Compressor()
_decompressor = Decompressor()

def compress(data: Union[str, ByteLike],
             output_encoding: str = 'ByteArray') -> Union[bytes, str]:
    raw = _compressor.compress(data)
    if output_encoding == 'ByteArray':
        return raw
    if output_encoding == 'Base64':
        return encode_base64(raw)
    if output_encoding == 'String':          # mostly for debug
        return decode_utf8(raw)
    raise ValueError('compress: unsupported output_encoding')

def decompress(data: Union[str, ByteLike],
               input_encoding : str = 'ByteArray',
               output_encoding: str = 'String') -> Union[str, bytes]:

    if input_encoding == 'Base64' and isinstance(data, str):
        raw_in = decode_base64(data)
    else:
        raw_in = bytes(data) if not isinstance(data, (bytes, bytearray)) else data

    raw_out = _decompressor.decompress(raw_in)
    if output_encoding == 'String':
        return decode_utf8(raw_out)
    if output_encoding in ('ByteArray', 'Buffer'):
        return raw_out
    raise ValueError('decompress: unsupported output_encoding')

# ---------------------------------------------------------------------------
# Tiny self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    txt = '🌍  Hello, LZUTF8 (Python port)!  🌍' * 16
    c   = compress(txt, 'Base64')
    d   = decompress(c, 'Base64', 'String')
    assert d == txt
    print('✓ self-test passed – compressed =', len(c), 'base-64 chars')