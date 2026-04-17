"""HarmBench dataset loader.

Loads behaviors from the HarmBench benchmark dataset.
See: https://github.com/centerforaisafety/HarmBench
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HARMBENCH_REPO = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main"
BEHAVIORS_FILE = "data/behavior_datasets/harmbench_behaviors_text_all.csv"


def load_harmbench(data_dir: str = "data/harmbench") -> list[dict]:
    """Load HarmBench behaviors from local cache or download.

    Returns:
        List of dicts with keys: id, behavior, category, source
    """
    data_path = Path(data_dir)
    behaviors_path = data_path / "harmbench_behaviors_text_all.csv"

    if not behaviors_path.exists():
        logger.info("HarmBench data not found locally. Downloading...")
        _download_harmbench(data_path)

    behaviors = []
    with open(behaviors_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append({
                "id": f"harmbench_{row.get('BehaviorID', len(behaviors))}",
                "behavior": row["Behavior"],
                "category": row.get("SemanticCategory", "unknown"),
                "source": "harmbench",
                "functional_category": row.get("FunctionalCategory", ""),
            })

    logger.info(f"Loaded {len(behaviors)} behaviors from HarmBench")
    return behaviors


def _download_harmbench(data_path: Path):
    """Download HarmBench behaviors CSV from GitHub."""
    import urllib.request

    data_path.mkdir(parents=True, exist_ok=True)
    url = f"{HARMBENCH_REPO}/{BEHAVIORS_FILE}"
    dest = data_path / "harmbench_behaviors_text_all.csv"

    logger.info(f"Downloading HarmBench behaviors from {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Saved to {dest}")
