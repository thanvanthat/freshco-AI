"""Train the head that tells a red apple from a pomegranate.

ImageNet-1k's only apple class is the green Granny Smith, so MobileNetV2 files
red apples under "pomegranate": 37 of 60 independently photographed apples, at
a median confidence of 0.87. This trains a two-class layer on the 1280-d
embedding to separate them; the classifier consults it only when ImageNet
answers "pomegranate".

DATA
----
Project-AgML (CC-BY-4.0) apple and pomegranate rows that do not overlap the
rows ``collect_agml_test.py`` samples for testing (that script takes rows 0-14
and 100-114 of each 200-row class block; these are rows 20-59 and 120-159)::

    python scripts/collect_agml_test.py --out ../data/agml_apple_pom \\
        --take 20 --offsets 20,40,120,140,1620,1640,1720,1740,1220,1240,1320,1340,2820,2840,2920,2940
    python scripts/train_apple_pomegranate_head.py --data ../data/agml_apple_pom

WHAT THE SHIPPED HEAD MEASURED (October 2026)
---------------------------------------------
Trained on 320 photos (fresh and rotten of each). Unlike fresh-vs-rotten, this
distinction held up on sources the head never saw:

- AgML held-out rows: apples 100%, pomegranates 100% (same source);
- 500 apples from Densu341/Fresh-rotten-fruit: 92% called apple;
- 15 whole-pomegranate photos from Wikimedia Commons: 14 called pomegranate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.vision.onnx_backend import APPLE_POMEGRANATE_HEAD_FILE, OnnxFoodModel  # noqa: E402
from train_food_head import train  # noqa: E402
from train_freshness_head import load  # noqa: E402

logger = logging.getLogger("train_apple_pomegranate_head")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = OnnxFoodModel()
    model.load()
    x, _, foods = load(model, args.data)
    keep = np.isin(foods, ["apple", "pomegranate"])
    y = (foods[keep] == "pomegranate").astype(int)
    logger.info("  %d apples, %d pomegranates", (y == 0).sum(), (y == 1).sum())

    w, b = train(x[keep], y, 2, epochs=1500, lr=0.5, weight_decay=args.weight_decay)
    out = args.out / APPLE_POMEGRANATE_HEAD_FILE
    np.savez_compressed(out, weights=w, bias=b)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
