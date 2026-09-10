# Binaire ML Assessment — Design

Date: 2026-09-10 · Brief: `ML_practical_assessment.pdf` (Binaire Private Limited)

## 1. Goal

Deliver a complete, working Python project covering the three tasks in the brief:

1. **HF API** — using only the `huggingface_hub` Python library (no CLI): follow the
   organization <https://huggingface.co/freznelai> and download two models:
   - `freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm`
   - `freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm`
2. **Scraping** — a CSV dataset of the first 100 carts listed at
   <https://www.lexaloffle.com/bbs/?cat=7&carts_tab=1&#sub=2&mode=carts> with: game name,
   author, artwork, game code, license, like count, description, top-5 comments.
3. **RAG** — turn the dataset into a queryable RAG database that generates PICO-8 code,
   usable from a CLI and a web UI.

Out of scope: fine-tuning or training on scraped content (lexaloffle.com `robots.txt`
declares `ai-train=no`; retrieval use is not restricted), validating generated Lua by
executing it, deployment.

## 2. Verified facts this design relies on

All checked against the live services on 2026-09-10.

**Hugging Face**
- `freznelai` is an organization (`/api/organizations/freznelai/overview`), not a user.
- Both models are public, ungated, Apache-2.0. Files: `.gitattributes`, `README.md`, one
  `.frzm` file each (Hub-reported repo storage: detector ≈0.2 MB, landmarker ≈3.7 MB).
- `huggingface_hub` (installed 1.10.1, latest 1.30.0) has **no follow method**, and the
  public OpenAPI spec has no follow endpoint. The website's endpoint
  `POST https://huggingface.co/api/organizations/freznelai/follow` exists: it returns 401
  unauthenticated, while unknown sibling paths return 404.
- `HfApi.get_organization_overview(organization, token=...)` returns `is_following`,
  which gives an authoritative check after the follow call.
- `InferenceClient(provider="groq", api_key=<non-hf_ key>)` sends requests directly to
  `https://api.groq.com/openai/v1/chat/completions`; HF credits are not used.
  `chat_completion` accepts `extra_body`.
- `openai/gpt-oss-120b` has a live `groq` provider mapping on the Hub.

**Groq free tier (gpt-oss-120b):** 30 RPM, 1K RPD, **8K TPM**, 200K TPD.

**Lexaloffle BBS**
- The page's cart grid is loaded by JavaScript from
  `https://www.lexaloffle.com/bbs/lister.php?use_hurl=1&cat=7&sub=2&mode=carts&page={n}`.
  The URL in the brief resolves to **"New Carts" (newest first)**. 30 carts per page; each
  cart is a `div` with id `pdat_<n>` containing `<a href="?tid=<thread id>">` and the title.
- Thread page: `https://www.lexaloffle.com/bbs/?tid={tid}`; long threads paginate via
  `?page={n}&tid={tid}` links.
- Each post is a `div` with id `p<pid>`, and each post has a star (like) button with class
  `rate_<pid>_like` whose text is the like count. The first post is the cart release
  (e.g. Petal Quest: author `noelcody`, `License: CC4-BY-NC-SA`, 106 likes). Every later
  post is a comment with its own author, date, star count and text.
- The first post references its cart as `cartsrc_<cart_id>`. The cart image is
  `https://www.lexaloffle.com/bbs/cposts/{cart_id[:2]}/{cart_id}.p8.png`; old numeric ids use
  `cposts/{id // 10000}/{id}.p8.png` (the site's `get_cart_url`). The scraper takes the URL from
  the thread page instead of computing it.
- **Game code is not in the HTML.** It is stored inside the cart PNG and decoded
  client-side by a WASM module (`cart_tools.js`); `snippet.php` does the same, no `.p8`
  text file is served (404), and no PyPI package decodes carts.
- `robots.txt`: `User-agent: *` → `Allow: /`; `Content-Signal: search=yes,ai-train=no,use=reference`.

**Cart PNG format** (a throwaway spike decoded `petal_quest-12` → 55,962 chars and
`weyubomopo-0` → 1,353 chars of clean Lua):
- 160×205 RGBA; one byte per pixel = `(A&3)<<6 | (R&3)<<4 | (G&3)<<2 | (B&3)`, row-major.
  First 0x8000 bytes = cart ROM; code region = 0x4300–0x7FFF.
- Code header `\0pxa` (current format): bytes 4–5 = decompressed length (big-endian); the
  bitstream starts at byte 8, read LSB-first.
  - Bit 1 → **literal**: count 1-bits until a 0 → `u`; `idx = bits(4+u) + (((1<<u)-1)<<4)`;
    take `mtf[idx]`, move it to the front of the move-to-front list (initially 0..255), and emit it.
  - Bit 0 → **copy**: offset width = `(5 if bit() else 10) if bit() else 15` (verified);
    `off = bits(width) + 1`. If `width == 10 and off == 1`, it's a raw block: emit 8-bit bytes until 0.
    Otherwise `len = 3 + Σ 3-bit groups`, stopping at the first group ≠ 7; copy byte-by-byte from `off` back.
- Code header `:c:\0` (legacy format): bytes 4–5 = length; the stream starts at byte 8.
  `0x00` → the next byte is a literal; `0x01–0x3b` → `LUT[b-1]` with
  `LUT = "\n 0123456789abcdefghijklmnopqrstuvwxyz!#%(){}[]<>+=/*:;.,~_"`;
  `≥0x3c` → `off = (b-0x3c)*16 + (next&0xf)`, `len = (next>>4)+2`.
- Any other header: raw bytes up to the first `\0`.
- Bytes 0x80–0xFF are PICO-8 glyphs (P8SCII), mapped to Unicode per the PICO-8 manual's
  128-entry table: 26 symbols `█ ▒ 🐱 ⬇️ ░ ✽ ● ♥ ☉ 웃 ⌂ ⬅️ 😐 ♪ 🅾️ ◆ … ➡️ ★ ⧗ ⬆️ ˇ ∧ ❎ ▤ ▥`,
  then 50 hiragana, 50 katakana, `◜ ◝`. Some entries are multi-codepoint emoji, so the
  table is a list, not a string.

**Environment:** Windows 11, Python 3.11.9. ChromaDB 1.5.7 with its built-in local
`all-MiniLM-L6-v2` ONNX embedder was smoke-tested here (collection names must be 3–512
chars of `[a-zA-Z0-9._-]`). Gradio 6.26.0 and `huggingface_hub` 1.30.0 resolve cleanly.

## 3. Decisions

| Topic | Decision | Why |
|---|---|---|
| Language | Python for all three tasks | Task 1 mandates the Python HF API; one toolchain |
| Scraping | `requests` + BeautifulSoup/lxml against `lister.php` and thread pages; own cart decoder | ~200 requests, no browser; headless Chrome + WASM would be ~10× slower and fragile |
| Listing order | "New Carts" as the brief's URL shows; the committed CSV is a timestamped snapshot | The list is newest-first and changes daily |
| Top-5 comments | Top 5 by star count, ties → earliest post; all thread pages read | User choice |
| Artwork | The cart `.p8.png` (the cartridge image with its label art): URL + local copy | It is the game's artwork and also the code source, so one download serves both |
| Vector DB | ChromaDB persistent store in `chroma/`, default local MiniLM embeddings | No key, no PyTorch, verified working locally |
| LLM | `openai/gpt-oss-120b` via `InferenceClient(provider="groq")` using `GROQ_API_KEY` | User choice; one library for HF and LLM |
| Token budget | ≤ 12,000 chars (~3.5K tokens) retrieved context, `max_tokens=3000`, `reasoning_effort="low"` | Stays under Groq's 8K TPM per request |
| Interfaces | CLI (`rag.py build/search/ask`) + Gradio UI (`app.py`) over the same functions | User choice |
| Secrets | `.env` (git-ignored) with `HF_TOKEN`, `GROQ_API_KEY`; `.env.example` committed | Never commit keys |

## 4. Layout

```
hf_task.py          Task 1
p8cart.py           cart PNG → Lua source
scrape.py           Task 2 → data/games.csv, data/artwork/
rag.py              Task 3: build | search | ask  (+ search()/ask() used by app.py)
app.py              Gradio UI
tests/
  fixtures/         saved lister/thread HTML, cart PNGs, golden .lua files
  test_p8cart.py  test_scrape.py  test_rag.py
data/games.csv      committed dataset
data/artwork/       committed cart images (<cart_id>.p8.png)
requirements.txt    pinned versions
.env.example  .gitignore  README.md
docs/design.md  docs/plan.md
```
Git-ignored: `.env`, `.venv/`, `models/`, `chroma/`, `.cache/`, `__pycache__/`, `*.p8` outputs.
`ML_practical_assessment.pdf` stays untracked unless the user asks otherwise.

## 5. Components

### 5.1 `hf_task.py` (Task 1)
1. `load_dotenv()`; read `HF_TOKEN`, and exit with a clear message pointing to `.env.example` if it's missing.
2. `HfApi(token).whoami()` → print the username (validates the token).
3. Follow: `get_session().post(f"{constants.ENDPOINT}/api/organizations/freznelai/follow",
   headers=build_hf_headers(token=token))`, then `hf_raise_for_status(resp)`. On an HTTP error,
   print the status and server message and exit non-zero.
4. Verify: `get_organization_overview("freznelai", token=token).is_following is True`,
   else exit non-zero.
5. For each model: `snapshot_download(repo_id, local_dir=f"models/{name}", token=token)`, then
   compare each local file's size with `model_info(repo_id, files_metadata=True).siblings[].size`
   and print a table. A mismatch is a failure.

Idempotent: re-running re-posts the follow (harmless) and re-uses the downloaded files.

### 5.2 `p8cart.py`
- `rom_from_png(png_bytes) -> bytes` (Pillow `tobytes()` on RGBA, 0x8000 bytes).
- `decode_code(rom) -> str`: dispatch on header (`\0pxa` / `:c:\0` / raw), then map
  bytes → text (ASCII as-is; 0x10–0x1F, 0x7F and 0x80–0xFF via the P8SCII table; 0x00–0x0F kept as `chr`).
- `cart_code(png_bytes) -> str` = both combined. An unknown or corrupt stream raises
  `ValueError` naming the problem.

### 5.3 `scrape.py` (Task 2)
- **HTTP:** one `requests.Session`, User-Agent `binaire-ml-assessment/1.0 (educational scraper)`,
  urllib3 `Retry(total=5, backoff_factor=1, status_forcelist=[429,500,502,503,504])`,
  30 s timeout. The raw response cache in `.cache/http/<sha1(url)>` makes re-runs hit the site zero times;
  delete the folder to refresh.
- **Listing:** fetch pages 1, 2, … and collect `(tid, title)` in order, deduplicating by tid
  until there are 100 (new posts shifting pages mid-scrape can't cause duplicates or gaps).
- **Per game** (ThreadPoolExecutor, 4 workers):
  - Thread page 1, plus pages 2..N if pagination links exist.
  - **First post:** name (thread `<title>`, HTML-unescaped), author (first non-empty
    `uid=` link text), `cart_id` (`cartsrc_` id), license (text after `License:`, else `""`),
    like count (`rate_<pid>_like` text, int), description (the post's message text,
    whitespace-normalized, excluding the cart player, the `Code ▽ | Embed ▽ | License` bar and the embed
    snippet; the exact element is pinned from fixtures during implementation).
  - **Comments:** all other posts across pages → `{author, stars, date, text}`, sorted by
    stars desc then post order, top 5 kept.
  - **Cart PNG:** download, save to `data/artwork/<cart_id>.p8.png`, decode via `p8cart`.
- **Output:** `data/games.csv`, UTF-8 with BOM, `csv` module quoting, rows in listing order.
  Columns: `rank, tid, cart_id, game_name, author, artwork_url, artwork_path, license,
  like_count, description, code, top_comments, thread_url, scraped_at`.
  `top_comments` is a JSON array; `scraped_at` is ISO-8601 UTC.
- **Validation (before writing):** exactly 100 rows, unique `tid`, and non-empty `game_name`,
  `author`, `cart_id`, `code`, with each artwork file present. Otherwise print the failing
  games and exit non-zero without writing the CSV. The summary prints counts of legitimately
  empty `license` / `description` / `top_comments`.
- **Known limitation (documented in README):** Excel truncates cells over 32,767 chars, and
  some carts exceed that. The CSV itself is complete; pandas and LibreOffice read it fully.

### 5.4 `rag.py` (Task 3)
- **Documents** (built from `data/games.csv`):
  - One `overview` doc per game: name, author, license, likes, description, top comments.
    Id `"{tid}-overview"`.
  - `code` docs: split the source at top-level function starts (unindented lines matching
    `function name(` or `name = function(`), pack consecutive blocks into chunks of ≤1,500 chars,
    and split any single block over 1,500 chars on line boundaries. Each doc is prefixed
    `-- game: {name} by {author}`. Ids `"{tid}-code-{i}"`.
  - Metadata on every doc (Chroma allows only str/int/float/bool, so empty values are
    `""`, never `None`): `tid, cart_id, game_name, author, license, like_count, kind, thread_url`.
- `build()` → `chromadb.PersistentClient("chroma")`: delete collection `pico8_games` if it
  exists, create it, and add in batches. Prints doc counts per kind.
- `search(query, k=8) -> list[hit]`: a pure vector query, with no LLM and no key.
- `ask(query, k=6) -> (answer, code, sources)`:
  - Retrieve k hits and keep them in rank order until the 12,000-char context budget is used.
  - **System prompt:** a PICO-8 primer (128×128 screen, 16-colour palette, `_init/_update/_draw`
    loop, `btn/btnp` 0–5, `spr/map/sfx/music/print/rectfill/circfill`, PICO-8 Lua differences
    such as `+=`, `!=` and no standard `string`/`table` libraries, 8,192-token cart limit).
    It instructs the model to return one complete runnable `lua` code block.
  - **User message:** the numbered snippets, each with its game/author attribution, then the request.
  - **Call:** `InferenceClient(provider="groq", api_key=GROQ_API_KEY).chat_completion(model=LLM_MODEL,
    messages, max_tokens=3000, temperature=0.3, extra_body={"reasoning_effort": "low"})`.
    `LLM_MODEL` comes from the env and defaults to `openai/gpt-oss-120b`.
  - Extract the first fenced code block (whole reply if none). `sources` = the unique games
    used, with URL and license.
- **CLI:**
  - `python rag.py build`
  - `python rag.py search "…" [-k 8]`
  - `python rag.py ask "…" [-k 6] [--out game.p8]`
- `write_p8(code, path)`: writes `pico-8 cartridge // http://www.pico-8.com\nversion 42\n__lua__\n{code}\n`
  (UTF-8), which loads in PICO-8 0.2.6+ or can be pasted into the free web Education Edition.

### 5.5 `app.py`
Gradio `Blocks` with a prompt box and two buttons:
- **Generate** → `ask`: a code panel (Lua highlighting if the installed Gradio supports it),
  sources in markdown, and a downloadable `.p8` file.
- **Search only** → `search`: ranked hits in markdown.

Serves on `http://127.0.0.1:7860`. Errors are shown in the UI instead of crashing it.

## 6. Error handling

| Situation | Behaviour |
|---|---|
| Missing `HF_TOKEN` / `GROQ_API_KEY` | Clear message naming the variable and `.env.example`; `search` and `build` need neither |
| Follow endpoint rejects token auth | Print exact status and body, exit non-zero (never report success unverified) |
| Network errors / 429 / 5xx while scraping | Retries with backoff; after that, the game is reported as failed, and the run fails validation |
| Cart decode failure | `ValueError` with the cart id; counted as a failed game |
| Groq 429 (TPM/RPM) or other API error | Message includes the status and retry-after if present; retrieved snippets are still shown |
| `chroma/` missing when searching | Message: run `python rag.py build` first |

## 7. Testing and verification

- `tests/test_p8cart.py`: fixture PNGs for a `\0pxa` cart, a legacy `:c:` cart (Celeste, 15133)
  and a glyph-heavy cart (Petal Quest). Decoded text must match fingerprints (FNV-1a of
  non-whitespace code points, line count, non-whitespace length) captured from Lexaloffle's own
  in-browser decoder (`snippet.php?cart_id=…&src=1`). The viewer renders tabs as 4 spaces, so
  whitespace is excluded; fingerprints avoid golden text files that git line-ending conversion
  could alter.
- `tests/test_scrape.py`: saved `lister.php` HTML → 30 `(tid, title)` pairs in order. Saved
  Petal Quest thread → author `noelcody`, license `CC4-BY-NC-SA`, likes 106, `cart_id`
  `petal_quest-12`, non-empty description, 5 comments sorted by stars. A saved thread with
  0 comments → an empty list.
- `tests/test_rag.py`: the chunker keeps functions intact, never exceeds 1,500 chars, and prefixes
  attribution; metadata has no `None`.
- **End to end (manual, recorded in the plan):**
  - `hf_task.py` prints `is_following=True` and a matching file table.
  - `scrape.py` writes 100 validated rows.
  - `rag.py build`, then `search "platformer with double jump"` returns relevant games.
  - `ask "make a snake game"` returns a Lua block and writes `snake.p8`.
  - `app.py` is exercised in a browser.
  - `pytest` is all green.

## 8. Delivery

- Commits after each milestone, authored by the user only, with no co-author trailers, and never pushed.
- The README covers setup (`python -m venv .venv`, `pip install -r requirements.txt`, and the
  `.env` keys, including how to create an HF token and a Groq key), one command per task,
  dataset schema, design notes, and limitations.
- Submission (done by the user): email `hr@binaire.app` with subject
  `Machine-Learning - Assessment – [Full Name] - [Enrollment / Roll Number] - [Institute Name]`.

## 9. Risks

1. **Follow via token** — the endpoint may only accept browser sessions. That's only testable
   with the user's token. Mitigation: exact error reporting. Any token-free alternative would
   break "HF API only", so none is planned.
2. **Groq free-tier limits** — 8K TPM allows roughly one `ask` per minute. That's acceptable
   for a demo, and the error message says when to retry.
3. **Site markup drift** — parsing is pinned by fixture tests, and live validation catches drift.
4. **Generated code correctness** — not executed or validated. The README says so and explains
   how to run the output in PICO-8.
