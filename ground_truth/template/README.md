# Ground truth template

Copy this folder to `ground_truth/<room-name>/` and fill it in **before tuning anything**.

| File | One row per | Notes |
|---|---|---|
| `books.csv` | every physical book | `legible` = could a person read the title from the spine without touching it? Fill `height_cm`/`thickness_cm` for at least 20 books (hand-measured). Leave title blank for illegible spines. `orientation` is `upright` or `flat`. |
| `prices.csv` | 15 hand-checked books | Same source type the system uses (e.g., a shopping result from an Indian store). Price in the sweep currency. |
| `items.csv` | every non-book item | `category` must be one of: shelving, furniture, appliance, lighting, art, rug, electronics, decor, other. |
| `room.json` | the room | Tape-measured. `wall_area_m2` is gross (perimeter × height) unless `wall_area_rule` says otherwise. |

Run `uv run python scripts/evaluate.py --packet sweeps/<id>/claim_packet.json --truth ground_truth/<room-name>`.
