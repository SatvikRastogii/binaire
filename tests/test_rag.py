import csv
import json

import rag
import scrape

HEADER = "-- game: t by a"


def test_chunk_keeps_small_functions_intact():
    funcs = [f"function f{i}()\n" + "  x=x+1\n" * 40 + "end" for i in range(10)]  # ~330 chars each
    code = "x=0\n" + "\n".join(funcs)
    chunks = rag.chunk_code(code, HEADER, limit=1500)
    assert len(chunks) > 1
    for fn in funcs:
        assert sum(fn in c for c in chunks) == 1
    for c in chunks:
        assert c.startswith(HEADER + "\n")
        assert len(c) <= 1500 + len(HEADER) + 1


def test_chunk_splits_oversized_function_on_lines():
    big = "function big()\n" + "".join(f"  v{i}={i}\n" for i in range(800)) + "end"
    chunks = rag.chunk_code(big, HEADER, limit=1500)
    assert len(chunks) > 1
    body = "\n".join(c[len(HEADER) + 1:] for c in chunks)
    assert body == big
    assert all(len(c) <= 1500 + len(HEADER) + 1 for c in chunks)


def test_chunk_splits_single_huge_line():
    line = "data=\"" + "a" * 4000 + "\""
    chunks = rag.chunk_code(line, HEADER, limit=1500)
    assert "".join(c[len(HEADER) + 1:] for c in chunks) == line
    assert all(len(c) <= 1500 + len(HEADER) + 1 for c in chunks)


def test_load_docs(tmp_path):
    path = tmp_path / "games.csv"
    base = {col: "" for col in scrape.COLUMNS}
    rows = [
        {**base, "rank": 1, "tid": 11, "cart_id": "a-0", "game_name": "Alpha", "author": "ann", "license": "No License",
         "like_count": 3, "description": "", "code": "function _init()\nend",
         "top_comments": json.dumps([{"author": "bob", "stars": 2, "date": "d", "text": "nice"}]), "thread_url": "u1"},
        {**base, "rank": 2, "tid": 22, "cart_id": "b-0", "game_name": "Beta", "author": "ben", "license": "CC4-BY-NC-SA",
         "like_count": 0, "description": "shoot", "code": "print('hi')", "top_comments": "[]", "thread_url": "u2"},
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=scrape.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    docs = rag.load_docs(path)
    assert [d["id"] for d in docs] == ["11-overview", "11-code-0", "22-overview", "22-code-0"]
    assert "bob (2 stars): nice" in docs[0]["text"]
    assert docs[1]["text"] == "-- game: Alpha by ann\nfunction _init()\nend"
    for d in docs:
        assert None not in d["meta"].values()
        assert d["meta"]["kind"] in ("overview", "code")
        assert isinstance(d["meta"]["tid"], int)


def test_build_context_respects_budget_and_dedupes_sources():
    meta = {"tid": 1, "game_name": "G", "author": "A", "license": "L", "kind": "code", "thread_url": "u"}
    hits = [{"text": "x" * 400, "meta": meta}, {"text": "y" * 400, "meta": meta}, {"text": "z" * 400, "meta": {**meta, "tid": 2}}]
    context, sources = rag.build_context(hits, budget=900)
    assert "x" * 400 in context and "y" * 400 in context and "z" * 400 not in context
    assert [s["tid"] for s in sources] == [1]


def test_extract_code_and_write_p8(tmp_path):
    answer = "Here:\n```lua\nfunction _draw()\n cls()\nend\n```\n- note"
    code = rag.extract_code(answer)
    assert code == "function _draw()\n cls()\nend"
    assert rag.extract_code("no fence") == "no fence"
    out = rag.write_p8(code, tmp_path / "g.p8")
    assert out.read_bytes() == b"pico-8 cartridge // http://www.pico-8.com\nversion 42\n__lua__\nfunction _draw()\n cls()\nend\n"
