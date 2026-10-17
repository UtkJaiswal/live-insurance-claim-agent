"""Run the post-sweep pipeline on a recorded (or simulated) sweep.

  uv run python scripts/process_sweep.py sweeps/<id>            # full run
  uv run python scripts/process_sweep.py sweeps/<id> --reuse    # reuse saved detections and spine reads
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.pipeline.run import process_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--reuse", action="store_true", help="reuse saved detect/read stage outputs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    packet, info = asyncio.run(process_sweep(args.sweep_dir, load_settings(), reuse=args.reuse))
    t = packet.totals
    print(f"\nbooks {t.book_count} (identified {t.books_identified}, unidentified {t.books_unidentified}); items {len(packet.items)}; "
          f"review {len(packet.review_queue)}; defects {len(info['defects'])}; cost ${info['metrics']['total_usd']}; "
          f"time {info['metrics']['processing_s']} s")
    for d in info["defects"]:
        print("DEFECT:", d)
    print("summary:", info["summary"])
    print("report:", args.sweep_dir / "report.html")


if __name__ == "__main__":
    main()
