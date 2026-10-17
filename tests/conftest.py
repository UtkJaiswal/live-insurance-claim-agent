import pytest

from app.schema import Book, BookReplacementCost, BookUsedValue, Item, ItemReplacementCost

SRC = {"source": "Example Store", "url": "https://example.com/p", "retrieved_at": "2026-10-18T10:00:00+00:00"}


@pytest.fixture
def sweep_dir(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "f_000001.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path


def make_book(i, status="identified", repl=None, used=None, thickness=2.0, title="A Book", **kw):
    return Book(
        id=f"b{i}",
        frame_ref="frames/f_000001.jpg",
        status=status,
        title=title if status != "unidentified" else None,
        spine_thickness_cm=thickness,
        replacement_cost=BookReplacementCost(amount=repl, **(SRC if repl is not None else {})),
        used_value=BookUsedValue(amount=used, condition_assumed="good" if used is not None else None,
                                 **(SRC if used is not None else {})),
        **kw,
    )


def make_item(i, status="priced", low=None, high=None, category="lighting"):
    priced = low is not None or high is not None
    return Item(
        id=f"i{i}",
        category=category,
        frame_ref="frames/f_000001.jpg",
        status=status,
        replacement_cost=ItemReplacementCost(low=low, high=high, **(SRC if priced else {})),
    )
