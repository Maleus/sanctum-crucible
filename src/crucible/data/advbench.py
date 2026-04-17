"""AdvBench dataset loader.

Loads harmful behaviors and strings from the AdvBench benchmark.
See: https://github.com/llm-attacks/llm-attacks
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ADVBENCH_REPO = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main"
BEHAVIORS_FILE = "data/advbench/harmful_behaviors.csv"
STRINGS_FILE = "data/advbench/harmful_strings.csv"


def load_advbench(data_dir: str = "data/advbench") -> list[dict]:
    """Load AdvBench harmful behaviors from local cache or download.

    Returns:
        List of dicts with keys: id, behavior, target, category, source
    """
    data_path = Path(data_dir)
    behaviors_path = data_path / "harmful_behaviors.csv"

    if not behaviors_path.exists():
        logger.info("AdvBench data not found locally. Downloading...")
        _download_advbench(data_path)

    behaviors = []
    with open(behaviors_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            behaviors.append({
                "id": f"advbench_{i}",
                "behavior": row["goal"],
                "target": row.get("target", ""),
                "category": "advbench",
                "source": "advbench",
            })

    logger.info(f"Loaded {len(behaviors)} behaviors from AdvBench")
    return behaviors


def _download_advbench(data_path: Path):
    """Download AdvBench data from GitHub."""
    import urllib.request

    data_path.mkdir(parents=True, exist_ok=True)

    for filename, url_path in [
        ("harmful_behaviors.csv", BEHAVIORS_FILE),
        ("harmful_strings.csv", STRINGS_FILE),
    ]:
        url = f"{ADVBENCH_REPO}/{url_path}"
        dest = data_path / filename
        logger.info(f"Downloading {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info(f"Saved to {dest}")
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")
