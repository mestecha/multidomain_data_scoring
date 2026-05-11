"""sample seed rows from per-country train CSVs for negative generation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

TRAIN_DIR = Path("data/input/multicultural/countries/train")

COUNTRY_FILES = {
    "AUS": "Australia",
    "CHL": "Chile",
    "CHN": "China",
    "DEU": "Germany",
    "ESP": "Spain",
    "IND": "India",
    "JPN": "Japan",
    "KEN": "Kenya",
    "MEX": "Mexico",
    "NZL": "New Zealand",
    "POL": "Poland",
    "USA": "United States of America",
    "ZAF": "South Africa",
}


def load_country(code: str) -> list[dict]:
    """read all rows for one country."""
    path = TRAIN_DIR / f"{code}.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample(code: str, n: int, seed: int = 42) -> list[dict]:
    """sample n rows from a country's train CSV (without replacement)."""
    rows = load_country(code)
    rng = random.Random(seed)
    n = min(n, len(rows))
    return rng.sample(rows, n)
