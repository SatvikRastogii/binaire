"""Task 2: scrapes the 100 newest PICO-8 carts from the Lexaloffle BBS into data/games.csv."""
import copy
import csv
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from p8cart import cart_code

BASE = "https://www.lexaloffle.com/bbs/"
# the link in the brief (bbs/?cat=7&carts_tab=1&#sub=2&mode=carts) fills its grid with JS,
# and this is where that JS gets the data from. 30 carts per page, newest first
LISTING_URL = BASE + "lister.php?use_hurl=1&cat=7&sub=2&mode=carts&page={page}"
THREAD_URL = BASE + "?tid={tid}"
THREAD_PAGE_URL = BASE + "?page={page}&tid={tid}"
USER_AGENT = "binaire-ml-assessment/1.0 (educational scraper)"
N_GAMES = 100
WORKERS = 4
CACHE_DIR = Path(".cache/http")
ART_DIR = Path("data/artwork")
CSV_PATH = Path("data/games.csv")
COLUMNS = [
    "rank", "tid", "cart_id", "game_name", "author", "artwork_url", "artwork_path", "license",
    "like_count", "description", "code", "top_comments", "thread_url", "scraped_at",
]


# --- http ---

def make_session() -> requests.Session:
    # retries with backoff on rate limits and server errors instead of failing the whole run
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS))
    return s


def fetch(session: requests.Session, url: str) -> bytes:
    """GET with a simple disk cache, so running the script again doesn't hit the site. Delete .cache/ to re-scrape."""
    path = CACHE_DIR / hashlib.sha1(url.encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


# --- parsing ---

def parse_listing(html: str) -> list[tuple[int, str]]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for card in soup.select('div[id^="pdat_"]'):
        link = card.select_one('a[href*="tid="]')
        out.append((int(re.search(r"tid=(\d+)", link["href"]).group(1)), link.get_text(" ", strip=True)))
    return out


def _posts(soup: BeautifulSoup) -> list:
    # posts are div#p<pid>, but the embedded cart player is div#p<cart_id>, and for old numeric
    # carts (e.g. p15133) that looks exactly like a post. only real posts have their own like button
    return [
        d for d in soup.find_all("div", id=re.compile(r"^p\d+$"))
        if d.select_one(f".rate_{d['id'][1:]}_like")
    ]


def _clean_text(el) -> str:
    # works on a copy so the original tree stays intact for the other fields.
    # removes the cart player and embed widgets, keeps paragraph breaks as newlines
    el = copy.copy(el)
    player = el.select_one('[class^="playarea_"]')
    if player is not None:
        player.parent.decompose()
    for junk in el.select('script, style, textarea, iframe, [id^="cartsrc_"], [id^="cartembed_"]'):
        junk.decompose()
    for br in el.find_all("br"):
        br.replace_with("\n")
    for block in el.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote"]):
        block.append("\n")
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in el.get_text().splitlines())
    return "\n".join(line for line in lines if line)


def _body(post):
    # a post with a cart keeps its text next to the cart widget, a plain comment has it
    # in the div right after the author/date line
    src = post.select_one('[id^="cartsrc_"]')
    if src is not None:
        return src.parent
    header = post.select_one('a[href*="uid="] b').find_parent("div")
    return header.find_next_sibling("div")


def parse_post(post) -> dict:
    pid = int(post["id"][1:])
    author = post.select_one('a[href*="uid="] b')
    date = post.select_one('a.desktop_div[href*="pid="]')
    likes = re.sub(r"\D", "", post.select_one(f".rate_{pid}_like").get_text())
    body = _body(post)
    return {
        "pid": pid,
        "author": author.get_text(strip=True) if author else "",
        "date": date.get_text(strip=True).rstrip("*") if date else "",
        "stars": int(likes) if likes else 0,
        "text": _clean_text(body) if body is not None else "",
    }


def parse_thread(html: str) -> dict:
    """First page of a thread. The first post is the cart release, everything after it is a comment."""
    soup = BeautifulSoup(html, "lxml")
    posts = _posts(soup)
    if not posts:
        raise ValueError("no posts found in thread page")
    op = posts[0]
    first = parse_post(op)
    title = op.select_one('a[href*="?tid="]')
    cartsrc = op.select_one('[id^="cartsrc_"]')
    cart_id = cartsrc["id"][len("cartsrc_"):] if cartsrc else ""
    cart_link = op.select_one(f'a[href$="/{cart_id}.p8.png"]') if cart_id else None
    player = op.select_one('[class^="playarea_"]')
    # the bar under the player reads "... | Code | Embed | License: CC4-BY-NC-SA" or "... | No License"
    info = player.parent.get_text(" ", strip=True) if player is not None else ""
    lic = re.search(r"License:\s*(\S+)|\b(No License)\b", info)
    pages = [int(m) for a in soup.select('a[href*="page="][href*="tid="]') for m in re.findall(r"page=(\d+)", a["href"])]
    return {
        "pid": first["pid"],
        "game_name": title.get_text(strip=True) if title else (soup.title.get_text(strip=True) if soup.title else ""),
        "author": first["author"],
        "cart_id": cart_id,
        "cart_url": urljoin(BASE, cart_link["href"]) if cart_link else "",
        "license": (lic.group(1) or lic.group(2)) if lic else "",
        "like_count": first["stars"],
        "description": first["text"],
        "comments": [parse_post(p) for p in posts[1:]],
        "page_count": max(pages, default=1),
    }


def parse_comments(html: str) -> list[dict]:
    """Pages 2 and up of a long thread, where every post is a comment."""
    return [parse_post(p) for p in _posts(BeautifulSoup(html, "lxml"))]


def top_comments(comments: list[dict], n: int = 5) -> list[dict]:
    """Top n comments by stars. sorted() is stable, so ties stay in thread order (oldest first)."""
    seen, unique = set(), []
    for c in comments:
        if c["pid"] not in seen:
            seen.add(c["pid"])
            unique.append(c)
    ranked = sorted(unique, key=lambda c: -c["stars"])[:n]
    return [{k: c[k] for k in ("author", "stars", "date", "text")} for c in ranked]


# --- pipeline ---

def collect_listing(session: requests.Session, n: int) -> list[tuple[int, str]]:
    # the listing is newest first, so if someone posts while this runs, a cart can slide onto
    # the next page and show up twice. dedupe by thread id and keep going until there are n
    seen, games = set(), []
    for page in range(1, 20):
        new = [(tid, t) for tid, t in parse_listing(fetch(session, LISTING_URL.format(page=page)).decode("utf-8", "replace")) if tid not in seen]
        if not new:
            break
        for tid, t in new:
            seen.add(tid)
            games.append((tid, t))
        if len(games) >= n:
            return games[:n]
    raise RuntimeError(f"listing ended after {len(games)} carts; expected {n}")


def scrape_game(session: requests.Session, rank: int, tid: int, scraped_at: str) -> dict:
    thread_url = THREAD_URL.format(tid=tid)
    t = parse_thread(fetch(session, thread_url).decode("utf-8", "replace"))
    comments = t["comments"]
    for page in range(2, t["page_count"] + 1):
        comments += parse_comments(fetch(session, THREAD_PAGE_URL.format(page=page, tid=tid)).decode("utf-8", "replace"))
    comments = [c for c in comments if c["pid"] != t["pid"]]
    if not t["cart_url"]:
        raise ValueError("no cart image link in the first post")
    png = fetch(session, t["cart_url"])
    art = ART_DIR / f"{t['cart_id']}.p8.png"
    art.write_bytes(png)
    return {
        "rank": rank,
        "tid": tid,
        "cart_id": t["cart_id"],
        "game_name": t["game_name"],
        "author": t["author"],
        "artwork_url": t["cart_url"],
        "artwork_path": art.as_posix(),
        "license": t["license"],
        "like_count": t["like_count"],
        "description": t["description"],
        "code": cart_code(png),
        "top_comments": json.dumps(top_comments(comments), ensure_ascii=False),
        "thread_url": thread_url,
        "scraped_at": scraped_at,
    }


def validate(rows: list[dict]) -> list[str]:
    errors = []
    if len(rows) != N_GAMES:
        errors.append(f"expected {N_GAMES} rows, got {len(rows)}")
    if len({r["tid"] for r in rows}) != len(rows):
        errors.append("duplicate thread ids")
    for r in rows:
        for field in ("game_name", "author", "cart_id", "license", "code"):
            if not r[field]:
                errors.append(f"rank {r['rank']} (tid {r['tid']}): empty {field}")
        if not Path(r["artwork_path"]).is_file():
            errors.append(f"rank {r['rank']} (tid {r['tid']}): artwork file missing")
    return errors


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    start = time.time()
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    listing = collect_listing(session, N_GAMES)
    print(f"Listing: {len(listing)} carts (newest first)")

    rows, failures = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        jobs = {pool.submit(scrape_game, session, rank, tid, scraped_at): (rank, tid, title)
                for rank, (tid, title) in enumerate(listing, 1)}
        for job in as_completed(jobs):
            rank, tid, title = jobs[job]
            try:
                rows.append(job.result())
            except Exception as e:  # don't stop at the first bad game, collect them all and report at the end
                failures.append(f"rank {rank} tid {tid} {title!r}: {type(e).__name__}: {e}")
    rows.sort(key=lambda r: r["rank"])

    errors = failures + validate(rows)
    if errors:
        print("FAILED, CSV not written:")
        print("\n".join(f"  {e}" for e in errors))
        return 1

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_comments = sum(len(json.loads(r["top_comments"])) for r in rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH} in {time.time() - start:.1f}s")
    print(f"  'No License': {sum(r['license'] == 'No License' for r in rows)}  "
          f"no description: {sum(not r['description'] for r in rows)}  "
          f"no comments: {sum(r['top_comments'] == '[]' for r in rows)}  "
          f"top comments kept: {n_comments}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
