import re
from pathlib import Path

import pytest

import p8cart

FIXTURES = Path(__file__).parent / "fixtures"


def fingerprint(code: str) -> tuple[int, int, int]:
    """FNV-1a over non-whitespace code points, line count, non-whitespace length.

    Lexaloffle's viewer renders tabs as spaces, so whitespace is excluded from the hash.
    """
    code = code.replace("\t", "    ")
    h = 0x811C9DC5
    for ch in re.sub(r"\s+", "", code):
        h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return h, len(code.split("\n")), len(re.sub(r"\s+", "", code))


# Captured from https://www.lexaloffle.com/bbs/snippet.php?cart_id=<id>&src=1 (official decoder), 2026-09-10.
OFFICIAL = {
    "weyubomopo-0": (2971063621, 89, 1047),  # pxa format, small
    "petal_quest-12": (814389582, 2142, 41266),  # pxa format, uses button glyphs
    "celeste-15133": (1672290677, 1429, 21230),  # legacy :c: format (cart 15133)
}


@pytest.mark.parametrize("name", OFFICIAL)
def test_matches_official_viewer(name):
    code = p8cart.cart_code((FIXTURES / f"{name}.p8.png").read_bytes())
    assert fingerprint(code) == OFFICIAL[name]


def test_known_content():
    zone = p8cart.cart_code((FIXTURES / "weyubomopo-0.p8.png").read_bytes())
    assert zone.startswith('--"the zone"\n--squishyam 9/9/2026')
    assert 'char="●"' in zone
    petal = p8cart.cart_code((FIXTURES / "petal_quest-12.p8.png").read_bytes())
    assert "⬅️" in petal and "❎" in petal
    celeste = p8cart.cart_code((FIXTURES / "celeste-15133.p8.png").read_bytes())
    assert celeste.startswith("-- ~celeste~\n-- matt thorson + noel berry")


def test_truncated_stream_raises():
    rom = bytearray(p8cart.rom_from_png((FIXTURES / "weyubomopo-0.p8.png").read_bytes()))
    rom[p8cart.CODE_START + 4 : p8cart.CODE_START + 6] = (0xFFFF).to_bytes(2, "big")
    with pytest.raises(ValueError):
        p8cart.decode_code(bytes(rom[: p8cart.CODE_START + 40]))


def test_raw_uncompressed_code():
    rom = bytes(p8cart.CODE_START) + b"print('hi')\n\x8b\0junk"
    assert p8cart.decode_code(rom) == "print('hi')\n⬅️"
