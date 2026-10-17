"""The claim packet contract.

Every field from the brief's output contract is present. Unknown values are None (serialised as
null): an empty field is correct when the system does not know; a filled field without evidence
is a defect. Extra fields we add are marked "extension".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BookStatus = Literal["identified", "unidentified", "needs_appraisal"]
ItemStatus = Literal["priced", "range", "needs_appraisal"]

# Shared vocabulary for non-book items. Ground truth uses the same list so category checks are exact.
ITEM_CATEGORIES = (
    "shelving",
    "furniture",
    "appliance",
    "lighting",
    "art",
    "rug",
    "electronics",
    "decor",
    "other",
)


class Sweep(BaseModel):
    id: str
    captured_at: str
    device: str = ""
    duration_s: float = 0
    country: str = ""
    currency: str = ""


class Room(BaseModel):
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    wall_area_m2: float | None = None
    shelved_wall_area_m2: float | None = None
    scale_method: str = ""
    confidence: float = 0
    # extension
    floor_area_ft2: float | None = None
    wall_area_ft2: float | None = None
    shape_notes: str = ""


class BookReplacementCost(BaseModel):
    amount: float | None = None
    source: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    converted: bool = False
    # extension: original price before conversion, for audit
    original_amount: float | None = None
    original_currency: str | None = None


class BookUsedValue(BaseModel):
    amount: float | None = None
    source: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    condition_assumed: str | None = None
    # extension
    converted: bool = False
    original_amount: float | None = None
    original_currency: str | None = None


class Book(BaseModel):
    id: str
    shelf: str | None = None
    position: int | None = None
    frame_ref: str
    status: BookStatus
    title: str | None = None
    author: str | None = None
    edition: str | None = None
    isbn: str | None = None
    spine_height_cm: float | None = None
    spine_thickness_cm: float | None = None
    id_confidence: float = 0
    replacement_cost: BookReplacementCost = Field(default_factory=BookReplacementCost)
    used_value: BookUsedValue = Field(default_factory=BookUsedValue)
    # extension
    orientation: Literal["upright", "flat"] = "upright"
    spine_text_read: str | None = None
    catalogue_url: str | None = None
    measurement_method: str | None = None
    appraisal_reason: str | None = None


class Dimensions(BaseModel):
    w: float | None = None
    h: float | None = None
    d: float | None = None


class ItemReplacementCost(BaseModel):
    low: float | None = None
    high: float | None = None
    source: str | None = None
    url: str | None = None
    retrieved_at: str | None = None


class Item(BaseModel):
    id: str
    category: str
    description: str = ""
    brand_model: str | None = None
    frame_ref: str
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    status: ItemStatus
    replacement_cost: ItemReplacementCost = Field(default_factory=ItemReplacementCost)
    confidence: float = 0
    # extension
    material: str | None = None
    dimensions_method: str | None = None
    appraisal_reason: str | None = None


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    shelf_run_m: float = 0
    books_replacement_cost: float = 0
    books_used_value: float = 0
    items_replacement_cost_low: float = 0
    items_replacement_cost_high: float = 0
    excluded_from_totals: int = 0


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str


class Excluded(BaseModel):
    """extension: lines the policyholder asked to leave out ("those are not mine"), kept for audit."""

    ref_id: str
    frame_ref: str
    statement: str


class ClaimPacket(BaseModel):
    sweep: Sweep
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)
    # extension
    excluded_by_policyholder: list[Excluded] = Field(default_factory=list)
