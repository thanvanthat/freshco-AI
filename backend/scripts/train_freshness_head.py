"""Train the fresh-vs-rotten head that refines the freshness score.

A two-class logistic layer on MobileNetV2's 1280-d embedding (added by
``expose_embedding.py``), trained with the same gradient-descent routine as
``train_food_head.py``. Writes ``models/freshness_head.npz``; the API picks it
up on the next start and reports ``+freshness-model`` in ``scoring_method``.

DATA
----
Folders named ``<food>__fresh`` and ``<food>__rotten``, as written by
``collect_fresh_rotten.py`` (training) and ``collect_agml_test.py``
(independent test)::

    python scripts/collect_fresh_rotten.py --out ../data/fresh_rotten
    python scripts/collect_agml_test.py    --out ../data/agml
    python scripts/train_freshness_head.py --train ../data/fresh_rotten/train \\
        --test ../data/fresh_rotten/test --independent ../data/agml

WHAT THE SHIPPED HEAD MEASURED (October 2026)
---------------------------------------------
Trained on 2,700 photos of 9 produce types (Densu341/Fresh-rotten-fruit):

- same-source held-out photos: 96.5% -- which describes that dataset, not a
  kitchen, and is not the number to trust;
- 480 independently photographed fruits (Project-AgML, 8 types, 5 never seen
  in training): AUC 0.83 against 0.69 for the OpenCV formula alone.

Its probabilities are near 0 or 1 even when wrong, and on the independent set
it called ~40% of sound fruit rotten. So ``app/freshness.py`` never lets it
declare anything spoiled: a confident "rotten" caps the score at nearly
spoiled ("inspect it"), a confident "sound" lets an undamaged item read fresh.
Under that policy, rotten fruit shown as fresh fell from 29.5% to 5.2% with no
increase in sound fruit shown as spoiled (0.5%).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.vision.onnx_backend import FRESHNESS_HEAD_FILE, OnnxFoodModel, softmax  # noqa: E402
from train_food_head import IMAGE_SUFFIXES, train  # noqa: E402

logger = logging.getLogger("train_freshness_head")


def load(model: OnnxFoodModel, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings, rotten flags and food names for ``root/<food>__<state>/*``."""
    embeddings, rotten, foods = [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name):
        food, state = folder.name.split("__", 1)
        if state not in {"fresh", "rotten"}:
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None or min(image.shape[:2]) < 32:
                continue
            embeddings.append(model.embedding(image))
            rotten.append(state == "rotten")
            foods.append(food)
    logger.info("  %s: %d images", root, len(embeddings))
    return np.stack(embeddings), np.asarray(rotten, dtype=int), np.asarray(foods)


def report(name: str, w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    flagged = softmax(x @ w + b)[:, 1] >= 0.5
    logger.info(
        "  %-22s accuracy %5.1f%% | rotten caught %5.1f%% | sound kept %5.1f%% (n=%d)",
        name,
        (flagged == y).mean() * 100,
        flagged[y == 1].mean() * 100,
        (~flagged)[y == 0].mean() * 100,
        len(y),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="held-out photos from the same source")
    parser.add_argument("--independent", type=Path, help="photos from a different source")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = OnnxFoodModel()
    model.load()

    logger.info("Embedding...")
    x, y, _ = load(model, args.train)
    w, b = train(x, y, 2, epochs=1500, lr=0.5, weight_decay=args.weight_decay)

    logger.info("Results (rotten = probability >= 0.5):")
    for name, root in (("same-source test", args.test), ("independent test", args.independent)):
        if root is not None:
            tx, ty, _ = load(model, root)
            report(name, w, b, tx, ty)

    out = args.out / FRESHNESS_HEAD_FILE
    np.savez_compressed(out, weights=w, bias=b)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
