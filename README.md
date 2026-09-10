# Binaire ML Assessment

My submission for the Binaire machine learning assessment. The brief has three tasks, and each one is a script:

- `hf_task.py` follows the [freznelai](https://huggingface.co/freznelai) org on Hugging Face and downloads their two face models, using only the Python `huggingface_hub` library.
- `scrape.py` scrapes the 100 newest carts from the [PICO-8 BBS](https://www.lexaloffle.com/bbs/?cat=7&carts_tab=1&#sub=2&mode=carts) into `data/games.csv`.
- `rag.py` turns that CSV into a vector database you can query, and uses it to generate PICO-8 code.

There's also a small web UI on top of the RAG part: `streamlit_app.py` (the one deployed on Streamlit Cloud), or `app.py` if you'd rather use Gradio.

Everything is Python 3.11.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill it in:

- `HF_TOKEN`: a Hugging Face token with write access. You only need it for task 1, because following an org has to be done as a logged-in user.
- `GROQ_API_KEY`: used for code generation. A free key from https://console.groq.com/keys is enough.
- `LLM_MODEL`: optional. Defaults to `openai/gpt-oss-120b`.

## Task 1: Hugging Face

```
python hf_task.py
```

It logs in with your token, follows freznelai, and checks that the follow actually went through. Then it downloads both models into `models/` and compares every file's size with what the Hub reports, so a truncated download gets caught.

`huggingface_hub` has no function for following someone. I checked the latest release and the Hub's public API docs, and neither has one. The Follow button on the website sends `POST /api/organizations/<org>/follow`, so the script sends that same request through the library's own session and auth headers. It then reads `is_following` back from `get_organization_overview()` to confirm the follow worked.

## Task 2: Scraping the BBS

```
python scrape.py
```

It takes about 15 seconds and produces `data/games.csv` plus the cart images in `data/artwork/`. Pages are cached in `.cache/`, so running it again is instant and doesn't hit the site. Delete that folder if you want fresh data.

What's in the CSV:

| column | what it is |
|---|---|
| `rank` | position in the listing, 1 is the newest |
| `tid` | the BBS thread id |
| `cart_id` | Lexaloffle's id for the cart, e.g. `petal_quest-12` |
| `game_name`, `author` | the game's name and the author's username |
| `artwork_url`, `artwork_path` | the cartridge image (which is the game's label art), online and saved locally |
| `license` | usually `CC4-BY-NC-SA`, or `No License` when the page says so |
| `like_count` | stars on the release post |
| `description` | the text of the release post; some authors don't write one |
| `code` | the full Lua source of the game |
| `top_comments` | up to 5 comments as JSON (`author`, `stars`, `date`, `text`), most starred first |
| `thread_url`, `scraped_at` | link to the thread, and when the scrape ran |

This part turned out to be less straightforward than it looked. Here's what I ran into:

- **The link in the brief doesn't contain the games.** That page builds its grid with JavaScript. The actual data comes from `bbs/lister.php` (30 carts per page, newest first), so the scraper calls that directly.
- **The game code isn't on the page either.** PICO-8 stores the whole cartridge inside the PNG image, two bits per colour channel, and the site decodes it in your browser with WebAssembly. No Python package does this, so I wrote `p8cart.py`. It pulls the bytes out of the image and decompresses the code, and it handles both the current compression format and the older pre-2020 one. I compared its output with Lexaloffle's own code viewer on three carts (a new one, an old one, and one full of button glyphs) and they match exactly. Those checks are in `tests/test_p8cart.py`.
- **Old carts are stored somewhere else.** Carts with plain numeric ids, like Celeste (`15133`), live in a different folder from newer ones. So the scraper takes the image link from the thread page instead of building the URL itself.
- **The cart player looks like a post.** Posts are `div#p<id>`, but the embedded player is `div#p<cart_id>`, which for numeric carts looks exactly like a post. A post only counts if it has its own like button.
- **"No License" is a real answer.** 35 of the 100 carts show "No License" on the page, so that's what goes in the column instead of leaving it blank.
- **The list moves while you scrape it.** It's newest first, so a new post during a run can push a cart onto the next page. Thread ids are deduplicated, so nothing gets counted twice or skipped.

The scraper won't write the CSV unless all 100 rows have a name, author, license, code and artwork. If anything fails, it prints which games failed and exits with an error.

## Task 3: The RAG database

```
python rag.py build                                    # takes about a minute
python rag.py search "platformer with double jump"     # searches the database only, no API key needed
python rag.py ask "make a snake game" --out snake.p8   # generates a cart
```

How it works:

1. **Documents.** Each game becomes one overview document (description, license, likes, top comments). Its code is split into chunks of up to 1,500 characters, cut at function boundaries so functions stay whole where they fit. That gives about 1,700 documents in total.
2. **Search.** They go into ChromaDB using its built-in MiniLM embeddings. Those run locally, so building and searching don't need a key.
3. **Generation.** For `ask`, the closest matches go into the prompt along with a short PICO-8 cheat sheet: the screen size, the palette, the button numbers, the functions PICO-8 has, and the parts of normal Lua it doesn't have. The prompt goes to gpt-oss-120b on Groq, via `huggingface_hub.InferenceClient`.
4. **Output.** You get a complete program, the games it drew from (with their licenses), and optionally a `.p8` file that PICO-8 can open directly.

I capped the retrieved context at about 3.5K tokens because Groq's free tier allows 8K tokens per minute. That works out to roughly one request a minute, which is fine for trying it out.

## Web UI

```
streamlit run streamlit_app.py     # http://localhost:8501
python app.py                      # same thing in Gradio, http://127.0.0.1:7860
```

Describe the game you want and hit Generate. You get the code, the sources, and a button to download the `.p8`. "Search only" shows what the database found without calling the model.

To deploy on Streamlit Community Cloud:
- Point it at `streamlit_app.py`.
- Pick Python 3.11.
- Add `GROQ_API_KEY = "..."` under Secrets.

The database is built the first time the app starts, which takes a minute or two. `requirements.txt` also pulls in `pysqlite3-binary` on Linux, because Chroma needs a newer SQLite than Streamlit's servers have shipped with in the past.

## Tests

```
python -m pytest
```

There are 19 tests. They run against saved pages and cart images in `tests/fixtures/`, so they don't need network access.

## Layout

```
hf_task.py          task 1
p8cart.py           cart image -> Lua decoder
scrape.py           task 2
rag.py              task 3 (build / search / ask)
streamlit_app.py    web UI (Streamlit)
app.py              web UI (Gradio)
data/               the dataset and cart images
tests/              tests and fixtures
docs/               design notes
```

## Limitations

- **The dataset is a snapshot.** The listing changes every day, so running the scraper again later gives different games. `scraped_at` tells you when this one was taken.
- **Excel cuts off long code.** Excel caps a cell at 32,767 characters, and 24 of the carts have more code than that. The CSV itself is complete, and pandas or LibreOffice read it fine.
- **Generated code isn't tested automatically.** Treat it as a starting point and run it in PICO-8.
- **The carts are used for retrieval only, not training.** Lexaloffle's robots.txt allows crawling but opts out of AI training, and nothing here trains a model.
