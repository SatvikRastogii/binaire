from pathlib import Path

import scrape

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_listing():
    games = scrape.parse_listing(read("lister_p1.html"))
    assert len(games) == 30
    assert len({tid for tid, _ in games}) == 30
    assert games[:3] == [(159194, '"The Zone"'), (159193, "Code Golf Snake"), (159182, "FlappyBird")]


def test_parse_thread_cart_post():
    t = scrape.parse_thread(read("thread_petal_quest.html"))
    assert t["pid"] == 160388
    assert t["game_name"] == "Petal Quest"
    assert t["author"] == "noelcody"
    assert t["cart_id"] == "petal_quest-12"
    assert t["cart_url"] == "https://www.lexaloffle.com/bbs/cposts/pe/petal_quest-12.p8.png"
    assert t["license"] == "CC4-BY-NC-SA"
    assert t["like_count"] == 106
    assert t["page_count"] == 1
    assert t["description"].startswith("Petal Quest is a tiny adventure in a forest of secrets.\nA game of discovery")
    for player_text in ("Copy and paste", "Cart #", "Code ▽", "License:"):
        assert player_text not in t["description"]


def test_parse_thread_comments():
    comments = scrape.parse_thread(read("thread_petal_quest.html"))["comments"]
    assert len(comments) == 26
    assert comments[0] == {"pid": 160390, "author": "Jaesoof", "date": "2025-01-08 20:40", "stars": 1, "text": "this is so cute"}
    for c in comments:
        assert "Mark as Spam" not in c["text"]


def test_top_comments_sorted_by_stars_then_thread_order():
    top = scrape.top_comments(scrape.parse_thread(read("thread_petal_quest.html"))["comments"])
    assert [(c["author"], c["stars"]) for c in top] == [
        ("RealShadowCaster", 3), ("noelcody", 2), ("Jaesoof", 1), ("Supernaut", 1), ("phil", 1),
    ]
    assert set(top[0]) == {"author", "stars", "date", "text"}


def test_top_comments_drops_duplicate_pids():
    c = {"pid": 1, "author": "a", "stars": 5, "date": "", "text": "x"}
    assert scrape.top_comments([c, dict(c), {**c, "pid": 2, "stars": 1}]) == [
        {"author": "a", "stars": 5, "date": "", "text": "x"},
        {"author": "a", "stars": 1, "date": "", "text": "x"},
    ]


def test_parse_thread_no_license_label():
    t = scrape.parse_thread(read("thread_no_license.html"))
    assert t["game_name"] == "Retro Rewind Volume 1"
    assert t["author"] == "mabbees"
    assert t["cart_id"] == "retro_rewind_vol1-4"
    assert t["license"] == "No License"
    assert "Note: This cartridge" not in t["description"]


def test_parse_thread_without_comments():
    t = scrape.parse_thread(read("thread_no_comments.html"))
    assert t["game_name"] == '"The Zone"'
    assert t["author"] == "squishyam"
    assert t["cart_id"] == "weyubomopo-0"
    assert t["comments"] == []
    assert scrape.top_comments(t["comments"]) == []
