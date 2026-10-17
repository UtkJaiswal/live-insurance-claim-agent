"""Score a claim packet against ground truth.

  uv run python scripts/evaluate.py --packet sweeps/<id>/claim_packet.json --truth ground_truth/<room>

Writes results.md and results.json next to the packet and prints the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluate import evaluate, to_json, to_markdown  # noqa: E402
from app.schema import ClaimPacket  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    args = ap.parse_args()

    packet = ClaimPacket.model_validate_json(args.packet.read_text())
    meta_path = args.packet.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    metrics_path = args.packet.parent / "metrics.json"
    if meta is not None and meta.get("packet_ready_t") is None and metrics_path.exists():
        # offline runs have no live clock: use the pipeline's own processing time
        ttp = json.loads(metrics_path.read_text()).get("time_to_packet_s")
        if ttp is not None:
            meta = {**meta, "sweep_ended_t": 0.0, "packet_ready_t": float(ttp)}
    metrics = evaluate(packet, args.truth, meta)

    md = to_markdown(metrics)
    (args.packet.parent / "results.md").write_text(md + "\n")
    (args.packet.parent / "results.json").write_text(json.dumps(to_json(metrics), indent=2))
    print(md)


if __name__ == "__main__":
    main()
