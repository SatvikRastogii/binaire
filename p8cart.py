"""Pulls the Lua source code out of a PICO-8 .p8.png cartridge image.

A .p8.png is a normal picture of the cartridge, but the whole 32KB cart is hidden in the
lowest bits of the pixels. The code section is compressed, usually with the "pxa" format
(PICO-8 0.2.0+), sometimes with the older ":c:" format.
"""
import io

from PIL import Image

ROM_SIZE = 0x8000
CODE_START = 0x4300  # gfx, map, sfx and music come first, the code sits after them

# the old format encodes common characters as single bytes 0x01-0x3b using this table
LEGACY_LUT = "\n 0123456789abcdefghijklmnopqrstuvwxyz!#%(){}[]<>+=/*:;.,~_"

# PICO-8's own charset (P8SCII) mapped to unicode, taken from the manual. some of these are
# emoji with a variation selector (two codepoints), which is why they're lists and not strings
_GLYPHS_10 = list("▮■□⁙⁘‖◀▶「」¥•、。゛゜")
_GLYPHS_80 = (
    ["█", "▒", "🐱", "⬇️", "░", "✽", "●", "♥", "☉", "웃", "⌂", "⬅️", "😐", "♪", "🅾️", "◆",
     "…", "➡️", "★", "⧗", "⬆️", "ˇ", "∧", "❎", "▤", "▥"]
    + list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんっゃゅょ")
    + list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンッャュョ")
    + ["◜", "◝"]
)
assert len(_GLYPHS_10) == 16 and len(_GLYPHS_80) == 128
P8SCII = [chr(i) for i in range(0x10)] + _GLYPHS_10 + [chr(i) for i in range(0x20, 0x7F)] + ["○"] + _GLYPHS_80


def rom_from_png(png: bytes) -> bytes:
    """One byte per pixel: the low 2 bits of A, R, G and B, in that order."""
    px = Image.open(io.BytesIO(png)).convert("RGBA").tobytes()
    return bytes(
        ((px[i + 3] & 3) << 6) | ((px[i] & 3) << 4) | ((px[i + 1] & 3) << 2) | (px[i + 2] & 3)
        for i in range(0, min(len(px), ROM_SIZE * 4), 4)
    )


def _decompress_pxa(d: bytes) -> bytes:
    n = (d[4] << 8) | d[5]  # decompressed length
    pos = 64  # 8 byte header, then a bitstream read lowest bit first

    def bit() -> int:
        nonlocal pos
        b = (d[pos >> 3] >> (pos & 7)) & 1
        pos += 1
        return b

    def bits(k: int) -> int:
        v = 0
        for j in range(k):
            v |= bit() << j
        return v

    mtf = list(range(256))
    out = bytearray()
    while len(out) < n:
        if bit():
            # literal: a unary prefix says how many bits the index has, then the index
            # into a move-to-front list of bytes (recently used bytes get short codes)
            u = 0
            while bit():
                u += 1
            c = mtf.pop(bits(4 + u) + (((1 << u) - 1) << 4))
            mtf.insert(0, c)
            out.append(c)
        else:
            # back-reference: prefix 11 -> 5 bit offset, 10 -> 10 bit, 0 -> 15 bit.
            # checked this against real carts, the order matters and is easy to get wrong
            width = (5 if bit() else 10) if bit() else 15
            off = bits(width) + 1
            if width == 10 and off == 1:
                # special marker: a run of raw bytes until a zero byte
                while c := bits(8):
                    out.append(c)
            else:
                # length is 3 + groups of 3 bits, a group of 7 means "keep reading"
                count = 3
                while True:
                    part = bits(3)
                    count += part
                    if part != 7:
                        break
                if off > len(out):
                    raise ValueError(f"pxa back-reference {off} beyond output length {len(out)}")
                for _ in range(count):
                    out.append(out[-off])
    return bytes(out[:n])


def _decompress_legacy(d: bytes) -> bytes:
    # 0x00 = next byte is a literal, 0x01-0x3b = table lookup, anything higher = back-reference
    n = (d[4] << 8) | d[5]
    out = bytearray()
    i = 8
    while len(out) < n:
        c = d[i]
        i += 1
        if c == 0:
            out.append(d[i])
            i += 1
        elif c <= 0x3B:
            out.append(ord(LEGACY_LUT[c - 1]))
        else:
            off = (c - 0x3C) * 16 + (d[i] & 0xF)
            count = (d[i] >> 4) + 2
            i += 1
            if off > len(out):
                raise ValueError(f"legacy back-reference {off} beyond output length {len(out)}")
            for _ in range(count):
                out.append(out[-off])
    return bytes(out[:n])


def decode_code(rom: bytes) -> str:
    code = rom[CODE_START:ROM_SIZE]
    try:
        if code[:4] == b"\0pxa":
            raw = _decompress_pxa(code)
        elif code[:4] == b":c:\0":
            raw = _decompress_legacy(code)
        else:  # no compression header, the code is plain bytes up to the first zero
            raw = code.split(b"\0", 1)[0]
    except IndexError as e:
        raise ValueError("compressed code stream is truncated") from e
    return "".join(P8SCII[b] for b in raw)


def cart_code(png: bytes) -> str:
    return decode_code(rom_from_png(png))
