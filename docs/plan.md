# Binaire ML Assessment Implementation Plan

> Execute task-by-task; steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Python project that (1) follows `freznelai` and downloads two models via the HF Python API, (2) scrapes the 100 newest PICO-8 carts into `data/games.csv`, and (3) serves a queryable RAG database (CLI + web UI) that generates PICO-8 code.

**Architecture:** Flat modules, one per job: `hf_task.py`, `p8cart.py` (cart PNG decoder), `scrape.py`, `rag.py` (ChromaDB + Groq via `huggingface_hub.InferenceClient`), `app.py` (Gradio). Parsing and decoding are pinned by fixture tests; live runs are validated end to end.

**Tech Stack:** Python 3.11, huggingface_hub, requests, beautifulsoup4 + lxml, Pillow, chromadb (local MiniLM ONNX embeddings), gradio, python-dotenv, pytest.

**Spec:** `docs/design.md`

## Global Constraints

- Task 1 uses only the `huggingface_hub` Python library — no `hf`/`huggingface-cli` commands.
- Secrets live only in `.env` (git-ignored); `.env.example` is committed with empty values.
- Commits are authored by the user only: no `Co-Authored-By`/session trailers, and **never push**.
- Stage files by name; never `git add -A`. Never stage `.env`, `models/`, `chroma/`, `.cache/`, or the PDF.
- Scraper User-Agent: `binaire-ml-assessment/1.0 (educational scraper)`; ≤ 4 concurrent requests.
- LLM: `openai/gpt-oss-120b` via `InferenceClient(provider="groq", api_key=GROQ_API_KEY)`, `max_tokens=3000`, context ≤ 12,000 chars, `extra_body={"reasoning_effort": "low"}`.
- ChromaDB collection name: `pico8_games`; store path: `chroma/`.
- CSV: `data/games.csv`, UTF-8 with BOM, columns exactly: `rank, tid, cart_id, game_name, author, artwork_url, artwork_path, license, like_count, description, code, top_comments, thread_url, scraped_at`.

---

### Task 1: Environment and scaffold

**Files:** Create `requirements.txt`; `.gitignore`, `.env.example` (already written)

- [ ] **Step 1:** Create a venv and install:
  `python -m venv .venv` then `.venv\Scripts\python -m pip install huggingface_hub==1.30.0 requests beautifulsoup4 lxml pillow chromadb gradio python-dotenv pytest`
- [ ] **Step 2:** Pin the top-level versions actually installed (`pip show` each) into `requirements.txt`.
- [ ] **Step 3:** Verify: `.venv\Scripts\python -c "import huggingface_hub, chromadb, gradio, bs4, PIL, dotenv; print(huggingface_hub.__version__)"` → `1.30.0`.
- [ ] **Step 4:** Commit `.gitignore .env.example requirements.txt docs/plan.md` — "Add project scaffold and implementation plan". Check with `git status` that `.env` is not listed.

### Task 2: Task 1 — HF follow + model download (`hf_task.py`)

**Interfaces:** Produces the CLI `python hf_task.py` (exit 0 only if follow is verified and all files match).

- [ ] **Step 1:** Implement per design §5.1: `load_dotenv()`, a `HF_TOKEN` check, `whoami`, a follow POST via `get_session()` + `build_hf_headers(token=...)` + `hf_raise_for_status`, an `is_following` check via `get_organization_overview`, and `snapshot_download(repo_id, local_dir=models/<name>)` for both repos, plus a size check against `model_info(files_metadata=True)`.
- [ ] **Step 2:** Offline sanity check: with `HF_TOKEN` empty, `python hf_task.py` must exit non-zero with a message naming `HF_TOKEN` and `.env.example`.
- [ ] **Step 3:** With a real token: run and confirm `is_following=True` and a table in which every file shows `OK`. (If only the follow fails, report the exact server status and body to the user.)
- [ ] **Step 4:** Commit `hf_task.py` — "Add Task 1: follow freznelai and download models via HF API".

### Task 3: Cart decoder (`p8cart.py`)

**Files:** Create `p8cart.py`, `tests/test_p8cart.py`, `tests/fixtures/*.p8.png`, `tests/fixtures/*.lua`

**Interfaces:** Produces `rom_from_png(png: bytes) -> bytes`, `decode_code(rom: bytes) -> str`, `cart_code(png: bytes) -> str`.

- [ ] **Step 1:** Download fixture carts: `petal_quest-12` (pxa, large), `weyubomopo-0` (pxa, small), one legacy `:c:` cart (a pre-2020 cart, e.g. from Celeste `tid=2145`), and one cart whose code uses glyphs such as `⬅️`.
- [ ] **Step 2:** Capture fingerprints for each from Lexaloffle's own viewer, `https://www.lexaloffle.com/bbs/snippet.php?cart_id=<id>&src=1` (rendered in Chrome; `#output` text with `<br>` → newline): FNV-1a of non-whitespace code points, line count, non-whitespace length.
- [ ] **Step 3:** Write tests: decoded fingerprints equal the official ones; known content snippets (including glyphs); `decode_code` raises `ValueError` on a truncated stream; raw (uncompressed) code decodes up to the first `\0`.
- [ ] **Step 4:** Run `pytest tests/test_p8cart.py -v` → FAIL (module missing).
- [ ] **Step 5:** Implement per design §2 "Cart PNG format" (pxa with the verified offset-width rule, legacy LUT, raw fallback, P8SCII table as a 128-entry list).
- [ ] **Step 6:** Run the tests → PASS.
- [ ] **Step 7:** Commit `p8cart.py tests/test_p8cart.py tests/fixtures/` — "Add PICO-8 cart PNG code decoder with golden tests".

### Task 4: Scraper (`scrape.py`)

**Files:** Create `scrape.py`, `tests/test_scrape.py`, `tests/fixtures/lister_p1.html`, `tests/fixtures/thread_petal_quest.html`, `tests/fixtures/thread_no_comments.html`

**Interfaces:**
- Consumes `p8cart.cart_code`.
- Produces `parse_listing(html) -> list[tuple[int, str]]`, `parse_thread(html) -> dict` (keys: `game_name, author, cart_id, license, like_count, description, comments, page_count`), `top_comments(comments, n=5) -> list[dict]`, and the CLI `python scrape.py`.

- [ ] **Step 1:** Save the fixtures (lister page 1, the Petal Quest thread `tid=146496`, and a 0-comment thread).
- [ ] **Step 2:** Inspect the first-post DOM in the fixture to pin the description element and the comment body/date elements.
- [ ] **Step 3:** Write the tests:
  - The listing gives 30 unique `(tid, title)` pairs, first pair as in the fixture.
  - Petal Quest gives author `noelcody`, license `CC4-BY-NC-SA`, like_count 106, cart_id `petal_quest-12`, and a description starting with "Petal Quest is a tiny adventure".
  - Comments are non-empty, the top 5 are sorted by stars desc, and each has `author/stars/date/text`.
  - The no-comment thread gives `comments == []`.
- [ ] **Step 4:** Run the tests → FAIL. Implement the parsers and the pipeline per design §5.3 (session + Retry, `.cache/http`, listing dedup to 100, thread pagination, cart download/decode, validation, CSV write).
- [ ] **Step 5:** Run the tests → PASS.
- [ ] **Step 6:** Live run: `python scrape.py` → the summary shows 100 rows written and exit 0. Spot-check 3 rows against the website by hand.
- [ ] **Step 7:** Commit `scrape.py tests/test_scrape.py tests/fixtures/*.html` — "Add Task 2 scraper for PICO-8 BBS carts". Then commit `data/games.csv data/artwork/` — "Add scraped dataset of 100 PICO-8 carts".

### Task 5: RAG database (`rag.py`)

**Files:** Create `rag.py`, `tests/test_rag.py`

**Interfaces:**
- Consumes `data/games.csv`.
- Produces `load_docs(csv_path) -> list[dict(id, text, meta)]`, `chunk_code(code: str, header: str, limit=1500) -> list[str]`, `build()`, `search(query, k=8) -> list[dict(id, text, meta, distance)]`, `ask(query, k=6) -> tuple[str, str, list[dict]]`, `write_p8(code, path)`, and the CLI `build | search | ask [--out]`.

- [ ] **Step 1:** Write the tests:
  - `chunk_code` never splits inside a top-level function smaller than the limit.
  - Every chunk is ≤ limit plus the header and starts with the header.
  - A 5,000-char function is split on line boundaries.
  - `load_docs` on a 2-row CSV gives overview + code docs whose metadata contains no `None`.
  - `write_p8` output starts with `pico-8 cartridge` and contains `__lua__`.
- [ ] **Step 2:** Run → FAIL. Implement per design §5.4.
- [ ] **Step 3:** Run → PASS.
- [ ] **Step 4:** Live: `python rag.py build` (prints counts), `python rag.py search "platformer with double jump"` (relevant hits), `python rag.py ask "make a snake game" --out snake.p8` (Lua block, sources, and the file written).
- [ ] **Step 5:** Commit `rag.py tests/test_rag.py` — "Add Task 3 RAG database with Groq-backed PICO-8 code generation".

### Task 6: Web UI (`app.py`)

- [ ] **Step 1:** Implement per design §5.5 (Generate → code, sources, .p8 file; Search only → markdown; errors shown in the UI).
- [ ] **Step 2:** Run `python app.py` and exercise both buttons in Chrome at `http://127.0.0.1:7860`, including an error case (empty prompt).
- [ ] **Step 3:** Commit `app.py` — "Add Gradio web UI for the RAG database".

### Task 7: README and final verification

- [ ] **Step 1:** Write `README.md`: setup, `.env` keys (how to get the HF token and the Groq key), one command per task, CSV schema, design notes (endpoint discovery, cart decoding, dedup/caching, token budget), limitations (Excel 32,767-char cells, follow-token risk, generated code not executed), and the submission email format.
- [ ] **Step 2:** Full check: `pytest -q` all green; `git status` clean except the untracked PDF; `git log` shows no Claude trailers.
- [ ] **Step 3:** Commit `README.md` — "Add README with setup, usage and design notes".
