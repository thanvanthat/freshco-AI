"""Sample fresh/rotten produce photos from the Densu341/Fresh-rotten-fruit dataset.

The dataset is one 3 GB zip on the Hugging Face Hub (OpenRAIL licence). A zip
keeps its table of contents at the end, so this reads that over HTTP range
requests and fetches only the sampled members -- a few hundred megabytes rather
than the whole archive.

Writes ``<out>/<split>/<food>__<fresh|rotten>/<name>`` with the dataset's
misspelt folder names normalised ("tamto" -> tomato). The sample is seeded, so
re-running reproduces it.

    python scripts/collect_fresh_rotten.py --out ../data/fresh_rotten
"""

from __future__ import annotations

import argparse
import collections
import io
import random
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

URL = "https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit/resolve/main/freshness_fruit.zip"

#: The dataset's folder spellings -> FreshCO AI knowledge-base names.
FOODS = {
    "apples": "apple",
    "banana": "banana",
    "bittergroud": "bitter_gourd",
    "capsicum": "capsicum",
    "cucumber": "cucumber",
    "okra": "okra",
    "oranges": "orange",
    "patato": "potato",
    "potato": "potato",
    "tamto": "tomato",
    "tomato": "tomato",
}
FOLDER = re.compile(r"^dataset/(Train|Test)/(fresh|rotten)(\w+)/[^/]+$")


class HttpFile(io.RawIOBase):
    """A seekable, read-only file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        self._client = httpx.Client(follow_redirects=True, timeout=60)
        head = self._client.head(url)
        head.raise_for_status()
        self._url = str(head.url)
        self.size = int(head.headers["content-length"])
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = {0: offset, 1: self._pos + offset, 2: self.size + offset}[whence]
        return self._pos

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        if self._pos >= self.size:
            return 0
        end = min(self._pos + len(buffer), self.size) - 1
        response = self._client.get(self._url, headers={"Range": f"bytes={self._pos}-{end}"})
        response.raise_for_status()
        data = response.content
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def open_zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[zipfile.ZipInfo]] = collections.defaultdict(list)
    for info in open_zip().infolist():
        match = FOLDER.match(info.filename)
        if match and match.group(3) in FOODS:
            split, state, food = match.group(1).lower(), match.group(2), FOODS[match.group(3)]
            groups[(split, f"{food}__{state}")].append(info)

    rng = random.Random(args.seed)
    plan: list[tuple[zipfile.ZipInfo, Path]] = []
    for (split, label), members in sorted(groups.items()):
        quota = args.train_per_class if split == "train" else args.test_per_class
        for info in rng.sample(members, min(quota, len(members))):
            plan.append((info, args.out / split / label / Path(info.filename).name))

    todo = [(i, p) for i, p in plan if not p.exists()]
    print(f"{len(plan)} sampled, {len(todo)} to fetch")

    local = threading.local()

    def fetch(item: tuple[zipfile.ZipInfo, Path]) -> None:
        info, path = item
        if not hasattr(local, "zip"):
            local.zip = open_zip()  # one connection per worker thread
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(local.zip.read(info))

    with ThreadPoolExecutor(args.workers) as pool:
        for done, _ in enumerate(pool.map(fetch, todo), 1):
            if done % 250 == 0:
                print(f"  {done}/{len(todo)}")

    counts = collections.Counter((p.parts[-3], p.parts[-2]) for _, p in plan)
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:5} {label:22} {n}")


if __name__ == "__main__":
    main()
