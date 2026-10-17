"""Tools the live agent can call, and the session state they update.

The live agent only records structure and user statements. Nothing it logs becomes a priced or
identified line by itself: the offline pipeline re-detects everything from the saved frames and uses
these records as hints and as the audit trail of what the user said.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schema import ITEM_CATEGORIES

LOCALE_ANSWER_WINDOW_S = 30.0

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "set_locale",
        "description": "Record the policyholder's country and currency once they confirm them.",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "ISO 3166-1 alpha-2 code, e.g. IN, US, GB"},
                "currency": {"type": "string", "description": "ISO 4217 code, e.g. INR, USD, GBP"},
            },
            "required": ["country", "currency"],
        },
    },
    {"name": "begin_sweep", "description": "Mark the start of the continuous room sweep."},
    {
        "name": "mark_region",
        "description": "The camera has moved to a new bookcase, shelf or wall. Labels frames from now on.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Stable label, e.g. 'Unit A shelf 2' or 'North wall'"},
                "kind": {"type": "string", "enum": ["shelf", "wall", "floor", "other"]},
            },
            "required": ["label", "kind"],
        },
    },
    {
        "name": "log_item",
        "description": "Log one non-book item seen in the room. Call once per distinct item.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(ITEM_CATEGORIES)},
                "description": {"type": "string", "description": "Short visual description incl. material"},
                "brand_model": {"type": "string", "description": "Only if legible in the image"},
            },
            "required": ["category", "description"],
        },
    },
    {
        "name": "record_user_statement",
        "description": "Record something the user said that affects the claim: a correction, an answer, "
        "an exclusion (not theirs / skip), or a provenance fact (signed, first edition, original, print).",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["correction", "answer", "exclusion", "provenance"]},
                "subject": {"type": "string", "description": "What it is about, e.g. 'portrait above desk' or 'Unit B shelf 3'"},
                "statement": {"type": "string", "description": "What the user said, close to verbatim"},
            },
            "required": ["kind", "subject", "statement"],
        },
    },
    {"name": "end_sweep", "description": "The room is covered or the user says they are done."},
]


@dataclass
class LiveState:
    country: str = ""
    currency: str = ""
    sweep_started_t: float | None = None
    sweep_ended_t: float | None = None
    last_user_speech_t: float | None = None
    region: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    statements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sweeping(self) -> bool:
        return self.sweep_started_t is not None and self.sweep_ended_t is None


def run_tool(state: LiveState, name: str, args: dict[str, Any], t: float, frame_ref: str | None) -> tuple[dict, dict | None]:
    """Applies a tool call. Returns (response for the model, event for the UI or None)."""
    args = args or {}
    if name == "set_locale":
        # In the first real sweep the agent set US/USD 3 s in, before the user said anything.
        if state.last_user_speech_t is None or t - state.last_user_speech_t > LOCALE_ANSWER_WINDOW_S:
            return {
                "ok": False,
                "error": "The user has not answered yet. Ask them to confirm their country and currency, "
                "wait for their spoken answer, then call set_locale.",
            }, None
        state.country = str(args.get("country", "")).upper()[:2]
        state.currency = str(args.get("currency", "")).upper()[:3]
        return {"ok": True}, {"type": "locale", "country": state.country, "currency": state.currency}

    if name == "begin_sweep":
        if state.sweep_started_t is None:
            state.sweep_started_t = t
        return {"ok": True}, {"type": "sweep", "state": "started", "t": t}

    if name == "mark_region":
        state.region = str(args.get("label", "")).strip() or None
        return {"ok": True}, {"type": "region", "label": state.region, "kind": args.get("kind"), "t": t}

    if name == "log_item":
        category = args.get("category") if args.get("category") in ITEM_CATEGORIES else "other"
        item = {
            "id": f"live_item_{len(state.items) + 1:03d}",
            "category": category,
            "description": str(args.get("description", "")),
            "brand_model": args.get("brand_model") or None,
            "region": state.region,
            "frame_ref": frame_ref,
            "t": t,
        }
        state.items.append(item)
        return {"ok": True, "id": item["id"]}, {"type": "item", **item}

    if name == "record_user_statement":
        stmt = {
            "kind": args.get("kind", "answer"),
            "subject": str(args.get("subject", "")),
            "statement": str(args.get("statement", "")),
            "region": state.region,
            "frame_ref": frame_ref,
            "t": t,
        }
        state.statements.append(stmt)
        return {"ok": True}, {"type": "statement", **stmt}

    if name == "end_sweep":
        if state.sweep_ended_t is None:
            state.sweep_ended_t = t
        return {"ok": True}, {"type": "sweep", "state": "ended", "t": t}

    return {"ok": False, "error": f"unknown tool {name}"}, None
