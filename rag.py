"""Task 3: a RAG database over the scraped carts. Builds the vector db, searches it, and generates PICO-8 code."""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "games.csv"
DB_PATH = ROOT / "chroma"
COLLECTION = "pico8_games"
CHUNK_LIMIT = 1500
# ~3.5K tokens of context. with 3K tokens for the answer that stays under groq's free tier cap of 8K tokens/min
CONTEXT_BUDGET = 12_000
DEFAULT_MODEL = "openai/gpt-oss-120b"

# the biggest carts go past csv's default 128K field limit. sys.maxsize would overflow on windows
csv.field_size_limit(2**31 - 1)

# unindented "function foo(" or "foo = function(", i.e. where a new top-level function starts
FUNC_START = re.compile(r"^(?:local\s+)?function\b|^[\w.:\[\]\"']+\s*=\s*function\b")

SYSTEM_PROMPT = """You are an expert PICO-8 game developer. You write code for the PICO-8 fantasy console (its Lua dialect).
PICO-8 facts:
- 128x128 screen, 16 colours: 0 black, 1 dark blue, 2 dark purple, 3 dark green, 4 brown, 5 dark grey, 6 light grey, 7 white, 8 red, 9 orange, 10 yellow, 11 green, 12 blue, 13 indigo, 14 pink, 15 peach.
- Game loop: _init() runs once, _update() runs at 30fps (or _update60() at 60fps), _draw() draws each frame.
- Input: btn(i) held / btnp(i) pressed, i = 0 left, 1 right, 2 up, 3 down, 4 O button, 5 X button.
- Graphics: cls, pset, pget, line, rect, rectfill, circ, circfill, oval, spr, sspr, map, print(str,x,y,col), pal, camera, clip.
- Sound: sfx(n), music(n).
- Maths: rnd, flr, ceil, abs, min, max, mid, sqrt, atan2, sin, cos (angles in turns 0..1; sin is inverted: sin(0.25) == -1).
- Tables: add, del, deli, count, all(), foreach, pairs, ipairs. Strings: sub, tostr, tonum, chr, ord, split, #s. No string./table./math. libraries.
- Shorthand operators +=, -=, *=, /=, != are valid. Numbers are 16.16 fixed point (-32768 to 32767.99). Carts are limited to 8192 tokens.
Rules:
- Reply with ONE complete, runnable program in a single ```lua code block, then at most 3 short bullet notes.
- Use only ASCII characters in code and comments (the PICO-8 button glyphs ⬅️➡️⬆️⬇️🅾️❎ are the only exception).
- Draw with shapes and print() unless the user says they have sprite/map/sound data; never reference assets that don't exist.
- Use the reference snippets for idioms and API usage; do not copy them wholesale."""


# --- documents ---

def chunk_code(code: str, header: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Splits code into chunks of at most `limit` chars, each starting with `header`.

    Cuts happen at top-level functions so a function usually stays in one piece. Anything
    longer than the limit gets split by lines, then small neighbouring pieces are packed together.
    """
    blocks, cur = [], []
    for line in code.split("\n"):
        if cur and FUNC_START.match(line):
            blocks.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))

    pieces = []
    for block in blocks:
        if len(block) <= limit:
            pieces.append(block)
            continue
        part = ""
        for line in block.split("\n"):
            while len(line) > limit:  # some carts keep sprite or level data in one giant string
                if part:
                    pieces.append(part)
                    part = ""
                pieces.append(line[:limit])
                line = line[limit:]
            if part and len(part) + 1 + len(line) > limit:
                pieces.append(part)
                part = line
            else:
                part = f"{part}\n{line}" if part else line
        if part:
            pieces.append(part)

    chunks, cur_chunk = [], ""
    for piece in pieces:
        if cur_chunk and len(cur_chunk) + 1 + len(piece) > limit:
            chunks.append(cur_chunk)
            cur_chunk = piece
        else:
            cur_chunk = f"{cur_chunk}\n{piece}" if cur_chunk else piece
    if cur_chunk:
        chunks.append(cur_chunk)
    return [f"{header}\n{c}" for c in chunks if c.strip()]


def load_docs(csv_path: Path = CSV_PATH) -> list[dict]:
    # every game gives one "overview" doc (what it is, what people said about it) plus its code chunks.
    # chroma only accepts str/int/float/bool metadata, so empty values stay as "" and never None
    docs = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            meta = {
                "tid": int(r["tid"]),
                "cart_id": r["cart_id"],
                "game_name": r["game_name"],
                "author": r["author"],
                "license": r["license"],
                "like_count": int(r["like_count"]),
                "thread_url": r["thread_url"],
            }
            comments = json.loads(r["top_comments"])
            overview = "\n".join([
                f"Game: {r['game_name']}",
                f"Author: {r['author']}",
                f"License: {r['license']}",
                f"Likes: {r['like_count']}",
                f"Description: {r['description'][:1500] or '(none)'}",
                "Top comments:" + "".join(f"\n- {c['author']} ({c['stars']} stars): {c['text'][:300]}" for c in comments),
            ])
            docs.append({"id": f"{r['tid']}-overview", "text": overview, "meta": {**meta, "kind": "overview"}})
            header = f"-- game: {r['game_name']} by {r['author']}"
            for i, chunk in enumerate(chunk_code(r["code"], header)):
                docs.append({"id": f"{r['tid']}-code-{i}", "text": chunk, "meta": {**meta, "kind": "code"}})
    return docs


# --- vector database ---

def _client():
    # imported here so the chunking code and its tests don't need chroma loaded
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))


def build(csv_path: Path = CSV_PATH) -> dict:
    """Rebuilds the collection from scratch, so running it twice never leaves duplicates."""
    docs = load_docs(csv_path)
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION)
    for i in range(0, len(docs), 256):
        batch = docs[i:i + 256]
        col.add(ids=[d["id"] for d in batch], documents=[d["text"] for d in batch], metadatas=[d["meta"] for d in batch])
    counts = {}
    for d in docs:
        counts[d["meta"]["kind"]] = counts.get(d["meta"]["kind"], 0) + 1
    return {"total": col.count(), **counts}


def ensure_built() -> bool:
    """Builds the database only if it isn't there yet, e.g. the first start after a cloud deploy."""
    if COLLECTION in [c.name for c in _client().list_collections()]:
        return False
    build()
    return True


def search(query: str, k: int = 8) -> list[dict]:
    client = _client()
    if COLLECTION not in [c.name for c in client.list_collections()]:
        raise RuntimeError("The RAG database is empty. Run `python rag.py build` first.")
    res = client.get_collection(COLLECTION).query(query_texts=[query], n_results=k)
    return [
        {"id": i, "text": doc, "meta": meta, "distance": dist}
        for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# --- generation ---

def build_context(hits: list[dict], budget: int = CONTEXT_BUDGET) -> tuple[str, list[dict]]:
    # takes hits in ranked order until the budget runs out. each snippet is labelled with its
    # game, author and license so the sources can be credited in the answer
    blocks, sources, used = [], [], 0
    for n, h in enumerate(hits, 1):
        m = h["meta"]
        block = f"[{n}] {m['game_name']} by {m['author']} ({m['kind']}, license: {m['license']})\n{h['text']}"
        if used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)
        if m["tid"] not in {s["tid"] for s in sources}:
            sources.append({k: m[k] for k in ("tid", "game_name", "author", "license", "thread_url")})
    return "\n\n".join(blocks), sources


def extract_code(answer: str) -> str:
    # first fenced code block in the reply, or the whole reply if the model skipped the fences
    m = re.search(r"```[^\n]*\n(.*?)```", answer, re.S)
    return (m.group(1) if m else answer).strip()


def ask(query: str, k: int = 6) -> tuple[str, str, list[dict]]:
    """Finds relevant snippets and asks the model for a PICO-8 program. Returns (answer, code, sources)."""
    from huggingface_hub import InferenceClient

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key.")
    context, sources = build_context(search(query, k))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Reference snippets from real PICO-8 carts:\n\n{context}\n\nRequest: {query}"},
    ]
    # with a groq key (not an hf_ token) InferenceClient talks to groq's api directly
    resp = InferenceClient(provider="groq", api_key=key).chat_completion(
        messages=messages,
        model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL,
        max_tokens=3000,
        temperature=0.3,
        extra_body={"reasoning_effort": "low"},
    )
    answer = resp.choices[0].message.content or ""
    return answer, extract_code(answer), sources


def p8_text(code: str) -> str:
    # minimal .p8 file: just the header and the code section, PICO-8 fills in the rest as empty
    return f"pico-8 cartridge // http://www.pico-8.com\nversion 42\n__lua__\n{code.rstrip()}\n"


def write_p8(code: str, path) -> Path:
    path = Path(path)
    path.write_text(p8_text(code), encoding="utf-8", newline="\n")
    return path


# --- cli ---

def main(argv=None) -> int:
    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="PICO-8 RAG database")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="(re)build the vector database from data/games.csv")
    s = sub.add_parser("search", help="query the database (no LLM)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    a = sub.add_parser("ask", help="generate PICO-8 code with retrieved context")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=6)
    a.add_argument("--out", help="write the generated code to this .p8 file")
    args = p.parse_args(argv)

    try:
        if args.cmd == "build":
            print(f"Built '{COLLECTION}' in {DB_PATH}: {build()}")
        elif args.cmd == "search":
            for n, h in enumerate(search(args.query, args.k), 1):
                m = h["meta"]
                print(f"\n[{n}] {m['game_name']} by {m['author']}  ({m['kind']}, distance {h['distance']:.3f})  {m['thread_url']}")
                print("    " + h["text"][:400].replace("\n", "\n    "))
        else:
            try:
                answer, code, sources = ask(args.query, args.k)
            except Exception as e:  # e.g. rate limited. still show what the search found
                print(f"Generation failed: {type(e).__name__}: {e}\nClosest matches in the database:")
                for h in search(args.query, args.k):
                    print(f"  - {h['meta']['game_name']} by {h['meta']['author']} ({h['meta']['kind']}) {h['meta']['thread_url']}")
                return 1
            print(answer)
            print("\nSources:")
            for src in sources:
                print(f"  - {src['game_name']} by {src['author']} ({src['license']}) {src['thread_url']}")
            if args.out:
                print(f"\nWrote {write_p8(code, args.out)}")
    except Exception as e:  # print something readable instead of a traceback
        print(f"Error: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
