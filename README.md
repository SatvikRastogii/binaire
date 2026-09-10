# Binaire — Machine Learning Assessment

Three tasks, all in Python:

| Task | What it does | Entry point |
|---|---|---|
| 1. HF API | Follows [huggingface.co/freznelai](https://huggingface.co/freznelai) and downloads both FreznelAI models using only the `huggingface_hub` Python library (no CLI) | `hf_task.py` |
| 2. Scraping | Builds `data/games.csv` from the 100 newest carts on the [PICO-8 BBS cartridge listing](https://www.lexaloffle.com/bbs/?cat=7&carts_tab=1&#sub=2&mode=carts) | `scrape.py` |
| 3. RAG | Turns the dataset into a queryable vector database that generates PICO-8 code (CLI + web UI) | `rag.py`, `app.py` |

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # includes the HF API: huggingface_hub
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Fill in `.env` (it is git-ignored):

| Variable | Needed for | How to get it |
|---|---|---|
| `HF_TOKEN` | Task 1 (the follow acts as your account) | huggingface.co → Settings → Access Tokens → **Write** token |
| `GROQ_API_KEY` | `rag.py ask` and the web UI | console.groq.com → API Keys |
| `LLM_MODEL` | optional | any Groq-served model; default `openai/gpt-oss-120b` |

## Task 1 — Hugging Face API

```bash
python hf_task.py
```

1. Validates the token (`HfApi.whoami`).
2. Follows the `freznelai` organization. `huggingface_hub` has no follow method, so the script sends the Hub's own
   follow request (`POST /api/organizations/freznelai/follow`, the endpoint behind the website's *Follow* button)
   through the library's HTTP session (`get_session`, `build_hf_headers`, `hf_raise_for_status`). It then confirms
   the result with `HfApi.get_organization_overview(...).is_following`.
3. Downloads both models with `snapshot_download` into `models/<repo name>/`, and checks every file's size against
   `HfApi.model_info(files_metadata=True)`:
   - `freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm`
   - `freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm`

Exits non-zero if any step can't be verified.

## Task 2 — Scraping the PICO-8 BBS

```bash
python scrape.py        # ~15 s; re-runs are served from .cache/ (delete it to re-scrape)
```

Output: `data/games.csv` (100 rows) and `data/artwork/<cart_id>.p8.png`.

| Column | Content |
|---|---|
| `rank` | position in the listing (1 = newest) |
| `tid` | BBS thread id |
| `cart_id` | Lexaloffle cart id (e.g. `petal_quest-12`) |
| `game_name` | name of the game |
| `author` | author's BBS username |
| `artwork_url` / `artwork_path` | the cartridge image (the game's label art) online / local copy |
| `license` | e.g. `CC4-BY-NC-SA`, or `No License` as labelled on the site |
| `like_count` | stars on the release post |
| `description` | text of the release post (empty when the author wrote none) |
| `code` | full Lua source decoded from the cart |
| `top_comments` | JSON list of up to 5 comments `{author, stars, date, text}`, most-starred first (ties → earliest) |
| `thread_url` | link to the thread |
| `scraped_at` | UTC timestamp of the run (the listing is newest-first, so it is a snapshot) |

**How it works**

- **Listing.** The brief's URL builds its grid with JavaScript from `bbs/lister.php?…&sub=2&mode=carts&page=N`
  (30 carts per page, "New Carts" order). The scraper reads that endpoint directly and de-duplicates thread ids,
  so carts posted mid-run can't shift pages into duplicates or gaps.
- **Threads.** Every post is a `div#p<id>` with its own star button. The first post is the release: name, author,
  license, likes, description. The rest are comments, and paginated threads are followed across all pages.
- **Game code.** The code is not in the HTML: PICO-8 stores the whole cartridge inside the `.p8.png` image, and
  the site decodes it in the browser with WebAssembly. `p8cart.py` does this in pure Python:
  1. It reads the hidden byte stored in the low 2 bits of each pixel's ARGB channels.
  2. It decompresses the code section, supporting both the current `pxa` format and the legacy `:c:` format.
  3. It maps PICO-8's glyph bytes (e.g. ⬅️ ❎) to Unicode.

  Its output matches Lexaloffle's own viewer exactly on new-format, legacy-format and glyph-heavy carts (see
  `tests/test_p8cart.py`).
- **Efficiency.** Pages are fetched by 4 parallel workers through a pooled session with automatic retry/backoff on
  429/5xx. The whole run is about 210 requests in about 15 s. A disk cache makes re-runs free and polite.
- **Safety net.** The CSV is written only if all 100 rows have a name, author, cart, license, code and artwork file.
  Otherwise the failing games are listed and the script exits non-zero.

## Task 3 — RAG database

```bash
python rag.py build                                  # ~1 min: embeds 100 overviews + ~1.6K code chunks into chroma/
python rag.py search "platformer with double jump"   # query the database (no API key needed)
python rag.py ask "make a snake game" --out snake.p8 # retrieve + generate a runnable PICO-8 cart
python app.py                                        # web UI at http://127.0.0.1:7860
```

- **Database:** ChromaDB, stored on disk in `chroma/`. It uses Chroma's built-in local `all-MiniLM-L6-v2` ONNX
  embeddings, so no key is needed; the model (~80 MB) is downloaded once on the first build.
- **Documents:** per game, one *overview* (name, author, license, likes, description, top comments), plus the code
  split at top-level function boundaries into chunks of up to 1,500 characters. Each chunk is tagged with its game,
  author, license and URL.
- **Generation:** the top matches go into the prompt, alongside a PICO-8 primer (screen, palette, game loop, input,
  API, dialect limits). The prompt is sent to `openai/gpt-oss-120b` on Groq, called through
  `huggingface_hub.InferenceClient(provider="groq")`. The retrieved context is capped at about 3.5K tokens so each
  request fits Groq's free tier (8K tokens/minute).
- **Output:** one complete Lua program, the source games it drew on (with their licenses), and optionally a `.p8`
  file you can load in PICO-8 0.2.6+ or paste into the free web version (Education Edition).
- **Web UI:** *Generate* shows the code, notes, sources and a downloadable `.p8` file. *Search only* shows the
  ranked database matches.

## Tests

```bash
python -m pytest
```

19 tests, run against saved fixtures:
- **Decoder:** fingerprints captured from Lexaloffle's official in-browser decoder.
- **Parsers:** listing, a thread with comments, a thread without comments, and a thread with no license.
- **RAG:** the chunker, document building, the context budget, and the `.p8` writer.

## Project layout

```
hf_task.py    Task 1
p8cart.py     .p8.png → Lua decoder
scrape.py     Task 2
rag.py        Task 3 (build / search / ask)
app.py        Task 3 web UI
data/         games.csv + artwork (committed)
tests/        pytest suite + fixtures
docs/         design and implementation plan
```

Generated or local only (git-ignored): `.env`, `.venv/`, `models/`, `chroma/`, `.cache/`, `*.p8`.

## Notes and limitations

- **Snapshot.** The listing is newest-first, so re-scraping later gives a different 100 games. The committed CSV
  is the snapshot taken at `scraped_at`.
- **Excel.** Excel truncates cells longer than 32,767 characters, and 24 carts have more code than that. The CSV
  itself is complete: pandas, LibreOffice and Python's `csv` module read it fully.
- **Control bytes.** Cart code keeps PICO-8's print control bytes (0x01–0x0F) exactly as stored in the cart.
- **Generated code.** It is not executed or validated automatically. Run it in PICO-8 to check it.
- **Scraping etiquette.** Scraping follows lexaloffle.com's `robots.txt`, which allows general crawling and
  disallows AI *training* (`ai-train=no`). This project only retrieves content at query time and trains nothing.
- **Groq free tier.** The free tier allows roughly one `ask` per minute. A rate-limit error names the limit; wait
  and retry.
