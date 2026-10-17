"""Scores a claim packet against hand-collected ground truth, one row per pass bar.

Matching rules (stated so the numbers can be challenged):
- Titles are normalised (lowercase, subtitle after ':' dropped, punctuation and leading articles
  removed) and matched one-to-one, greedily by fuzzy score >= TITLE_MATCH.
- Title recall uses legible spines as the denominator. "Confidently wrong" is every title the
  system asserted that matches no book in the room, divided by all titles it asserted.
- Spine dimensions and prices are compared only on sample books the system matched; coverage is
  reported next to the error so a small matched set cannot hide.
- Non-book items: found = sum over categories of min(true count, predicted count).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from rapidfuzz import fuzz

from app.schema import Book, ClaimPacket

TITLE_MATCH = 88


@dataclass
class Metric:
    measure: str
    target: str
    result: str
    passed: bool | None
    detail: str = ""


def norm_title(t: str) -> str:
    t = t.lower().split(":")[0]
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def title_score(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))


def match_titles(pred: list[str], truth: list[str], threshold: float = TITLE_MATCH) -> dict[int, int]:
    """One-to-one greedy matching. Returns {pred index: truth index}."""
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            s = title_score(p, t)
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            out[i] = j
            used_p.add(i)
            used_t.add(j)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pct(pred: float | None, true: float | None) -> float | None:
    if pred is None or not true:
        return None
    return abs(pred - true) / abs(true)


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def evaluate(packet: ClaimPacket, truth_dir: Path, sweep_meta: dict | None = None) -> list[Metric]:
    gt_books = _read_csv(truth_dir / "books.csv")
    gt_prices = _read_csv(truth_dir / "prices.csv") if (truth_dir / "prices.csv").exists() else []
    gt_items = _read_csv(truth_dir / "items.csv") if (truth_dir / "items.csv").exists() else []
    gt_room = json.loads((truth_dir / "room.json").read_text()) if (truth_dir / "room.json").exists() else {}
    metrics: list[Metric] = []

    # Book count
    true_n, pred_n = len(gt_books), len(packet.books)
    err = _pct(pred_n, true_n)
    metrics.append(Metric("Book count", "within 5%", f"{pred_n} vs {true_n} ({_fmt_pct(err)})",
                          None if err is None else err <= 0.05))

    # Titles
    titled_pred = [b for b in packet.books if b.title]
    titled_gt = [(i, r) for i, r in enumerate(gt_books) if r.get("title")]
    legible_idx = {i for i, r in titled_gt if r.get("legible", "").lower() in ("yes", "y", "true", "1")}
    m = match_titles([b.title for b in titled_pred], [r["title"] for _, r in titled_gt])
    gt_index_of = {k: titled_gt[v][0] for k, v in m.items()}  # pred idx -> row in gt_books
    correct = sum(1 for g in gt_index_of.values() if g in legible_idx)
    wrong = len(titled_pred) - len(m)
    recall = correct / len(legible_idx) if legible_idx else None
    wrong_rate = wrong / len(titled_pred) if titled_pred else 0.0
    metrics.append(Metric(
        "Title identification", ">= 70% of legible correct, <= 3% confidently wrong",
        f"{_fmt_pct(recall)} correct ({correct}/{len(legible_idx)}), {_fmt_pct(wrong_rate)} wrong ({wrong}/{len(titled_pred)})",
        None if recall is None else (recall >= 0.70 and wrong_rate <= 0.03),
    ))

    # Spine dimensions (on the hand-measured sample)
    book_for_gt_row: dict[int, Book] = {g: titled_pred[p] for p, g in gt_index_of.items()}
    sample = [(i, r) for i, r in enumerate(gt_books) if _f(r.get("height_cm", "")) is not None]
    h_err, t_err, matched = [], [], 0
    for i, r in sample:
        b = book_for_gt_row.get(i)
        if b is None:
            continue
        matched += 1
        e = _pct(b.spine_height_cm, _f(r["height_cm"]))
        if e is not None:
            h_err.append(e)
        e = _pct(b.spine_thickness_cm, _f(r.get("thickness_cm", "")))
        if e is not None:
            t_err.append(e)
    all_err = h_err + t_err
    metrics.append(Metric(
        "Spine dimensions", "within 15% (sample of 20)",
        f"height MAPE {_fmt_pct(mean(h_err) if h_err else None)}, thickness MAPE {_fmt_pct(mean(t_err) if t_err else None)}",
        None if not all_err else mean(all_err) <= 0.15,
        f"matched {matched}/{len(sample)} sample books",
    ))

    # Book prices
    p_err, p_matched = [], 0
    pred_titles = [b.title or "" for b in packet.books]
    for r in gt_prices:
        b = next((x for x in packet.books if r.get("isbn") and x.isbn == r["isbn"]), None)
        if b is None and r.get("title"):
            mm = match_titles(pred_titles, [r["title"]])
            b = packet.books[next(iter(mm))] if mm else None
        if b is None:
            continue
        p_matched += 1
        for pred, true in ((b.replacement_cost.amount, _f(r.get("replacement_cost", ""))),
                           (b.used_value.amount, _f(r.get("used_value", "")))):
            e = _pct(pred, true)
            if e is not None:
                p_err.append(e)
    metrics.append(Metric(
        "Book prices", "within 25% (sample of 15)", f"MAPE {_fmt_pct(mean(p_err) if p_err else None)}",
        None if not p_err else mean(p_err) <= 0.25, f"matched {p_matched}/{len(gt_prices)}, {len(p_err)} price pairs compared",
    ))

    # Non-book items
    gt_cats: dict[str, int] = {}
    for r in gt_items:
        gt_cats[r["category"]] = gt_cats.get(r["category"], 0) + 1
    pred_cats: dict[str, int] = {}
    for it in packet.items:
        pred_cats[it.category] = pred_cats.get(it.category, 0) + 1
    found = sum(min(n, pred_cats.get(c, 0)) for c, n in gt_cats.items())
    extra = sum(max(0, n - gt_cats.get(c, 0)) for c, n in pred_cats.items())
    rec = found / len(gt_items) if gt_items else None
    metrics.append(Metric("Non-book items", ">= 80% found, correct category",
                          f"{_fmt_pct(rec)} ({found}/{len(gt_items)})", None if rec is None else rec >= 0.80,
                          f"{extra} extra predictions"))

    # Room
    for label, key, tol in (("Floor area", "floor_area_m2", 0.10), ("Wall area", "wall_area_m2", 0.15)):
        e = _pct(getattr(packet.room, key), _f(str(gt_room.get(key, ""))))
        metrics.append(Metric(label, f"within {int(tol * 100)}%",
                              f"{getattr(packet.room, key)} vs {gt_room.get(key)} m2 ({_fmt_pct(e)})",
                              None if e is None else e <= tol))

    # Time to packet
    ttp = None
    if sweep_meta and sweep_meta.get("packet_ready_t") is not None and sweep_meta.get("sweep_ended_t") is not None:
        ttp = sweep_meta["packet_ready_t"] - sweep_meta["sweep_ended_t"]
    metrics.append(Metric("Time to packet", "under 5 minutes", "n/a" if ttp is None else f"{ttp:.0f} s",
                          None if ttp is None else ttp < 300))
    return metrics


def to_markdown(metrics: list[Metric]) -> str:
    mark = {True: "PASS", False: "FAIL", None: "n/a"}
    lines = ["| Measure | Pass bar | Result | Status | Detail |", "|---|---|---|---|---|"]
    lines += [f"| {m.measure} | {m.target} | {m.result} | {mark[m.passed]} | {m.detail} |" for m in metrics]
    return "\n".join(lines)


def to_json(metrics: list[Metric]) -> list[dict]:
    return [asdict(m) for m in metrics]
