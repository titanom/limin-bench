import json
import os

from .base import PregeneratedMultiTurnDataset


def load_mt_bench(categories: list[str] | None = None) -> PregeneratedMultiTurnDataset:
    current_dir = os.path.dirname(__file__)
    mt_bench_path = os.path.join(current_dir, "data/mt_bench.json")

    with open(mt_bench_path, "r") as f:
        data = json.load(f)

    rows = []

    if categories is None:
        categories = list(data.keys())

    for category in categories:
        if category in data:
            for row in data[category]["rows"]:
                rows.append(row)

    return PregeneratedMultiTurnDataset(rows=rows)
