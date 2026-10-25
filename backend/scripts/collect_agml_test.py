"""Sample an independent fresh/rotten test set from Project-AgML (CC-BY-4.0).

Sultana, Jahan & Uddin (2022), "An extensive dataset for successful recognition
of fresh and rotten fruits", Data in Brief 44:108552 -- photographed in
Bangladesh, separately from the training data. It is used only for testing:
held-out photos from the *training* source measure that source, not a kitchen,
which is how a head once scored 96% in testing and then called mouldy bread raw
meat. A model must hold up here before the app uses it.

The 3,200 raw rows are grouped by fruit and label, so taking ``--take`` rows at
every ``--step`` spreads the sample across all of them.

    python scripts/collect_agml_test.py --out ../data/agml
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import httpx

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "Project-AgML/fresh_rotten_fruit_classification"
STATES = {0: "fresh", 1: "rotten"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--take", type=int, default=15)
    parser.add_argument(
        "--offsets",
        help="comma-separated row offsets to read instead of every --step; "
        "used to draw training rows that do not overlap the test sample",
    )
    args = parser.parse_args()
    offsets = (
        [int(o) for o in args.offsets.split(",")]
        if args.offsets
        else list(range(0, 3200, args.step))
    )

    counts: collections.Counter[str] = collections.Counter()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for offset in offsets:
            response = client.get(
                ROWS,
                params={
                    "dataset": DATASET,
                    "config": "raw",
                    "split": "train",
                    "offset": offset,
                    "length": args.take,
                },
            )
            response.raise_for_status()
            for item in response.json()["rows"]:
                row = item["row"]
                label = f"{row['fruit_type'].lower()}__{STATES[row['label']]}"
                path = args.out / label / f"{item['row_idx']}.jpg"
                if not path.exists():
                    image = client.get(row["image"]["src"])
                    image.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(image.content)
                counts[label] += 1

    for label, n in sorted(counts.items()):
        print(f"  {label:22} {n}")


if __name__ == "__main__":
    main()
