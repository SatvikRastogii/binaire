"""Extract Lua source code from a PICO-8 .p8.png cartridge image."""
import io

from PIL import Image

ROM_SIZE = 0x8000
CODE_START = 0x4300

LEGACY_LUT = "\n 0123456789abcdefghijklmnopqrstuvwxyz!#%(){}[]<>+=/*:;.,~_"

# P8SCII glyphs → Unicode (PICO-8 manual). Several are multi-codepoint emoji, hence lists.
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
    """Each pixel hides one byte in the low 2 bits of A, R, G, B."""
    px = Image.open(io.BytesIO(png)).convert("RGBA").tobytes()
    return bytes(
        ((px[i + 3] & 3) << 6) | ((px[i] & 3) << 4) | ((px[i + 1] & 3) << 2) | (px[i + 2] & 3)
        for i in range(0, min(len(px), ROM_SIZE * 4), 4)
    )


def _decompress_pxa(d: bytes) -> bytes:
    n = (d[4] << 8) | d[5]
    pos = 64  # bitstream starts after the 8-byte header

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
            u = 0
            while bit():
                u += 1
            c = mtf.pop(bits(4 + u) + (((1 << u) - 1) << 4))
            mtf.insert(0, c)
            out.append(c)
        else:
            width = (5 if bit() else 10) if bit() else 15
            off = bits(width) + 1
            if width == 10 and off == 1:
                while c := bits(8):
                    out.append(c)
            else:
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
        else:
            raw = code.split(b"\0", 1)[0]
    except IndexError as e:
        raise ValueError("compressed code stream is truncated") from e
    return "".join(P8SCII[b] for b in raw)


def cart_code(png: bytes) -> str:
    return decode_code(rom_from_png(png))
