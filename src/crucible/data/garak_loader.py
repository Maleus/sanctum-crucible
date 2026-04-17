"""Garak dataset loader.

Loads probe data from Garak (LLM vulnerability scanner by NVIDIA).
See: https://github.com/NVIDIA/garak

Garak probes are organized by category. This loader extracts the prompt
templates from Garak's probe definitions for use as training data.
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def load_garak(
    data_dir: str = "data/garak",
    probe_families: list[str] | None = None,
) -> list[dict]:
    """Load Garak probes as behavior definitions.

    This runs garak's probe listing and extracts prompt templates.
    Garak must be installed: pip install garak

    Args:
        data_dir: Directory to cache extracted probes.
        probe_families: Optional list of probe families to include.
            Defaults to all available probes.

    Returns:
        List of dicts with keys: id, behavior, category, source
    """
    data_path = Path(data_dir)
    cache_file = data_path / "garak_probes.json"

    if cache_file.exists():
        with open(cache_file) as f:
            behaviors = json.load(f)
        logger.info(f"Loaded {len(behaviors)} probes from Garak cache")
        return behaviors

    logger.info("Extracting probes from Garak installation...")
    behaviors = _extract_garak_probes(probe_families)

    # Cache for future runs
    data_path.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(behaviors, f, indent=2)

    logger.info(f"Extracted and cached {len(behaviors)} Garak probes")
    return behaviors


def _extract_garak_probes(probe_families: list[str] | None = None) -> list[dict]:
    """Extract probe prompts from Garak's installed probes."""
    try:
        import garak.probes
        import importlib
        import pkgutil
    except ImportError:
        logger.error(
            "Garak not installed. Install with: pip install garak"
        )
        return _load_garak_fallback()

    behaviors = []
    probe_idx = 0

    for importer, modname, ispkg in pkgutil.walk_packages(
        garak.probes.__path__, prefix="garak.probes."
    ):
        if probe_families and not any(f in modname for f in probe_families):
            continue

        try:
            module = importlib.import_module(modname)
        except Exception as e:
            logger.debug(f"Skipping probe module {modname}: {e}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and hasattr(attr, "prompts")
                and attr_name != "Probe"
            ):
                try:
                    probe_instance = attr()
                    prompts = getattr(probe_instance, "prompts", [])
                    category = modname.replace("garak.probes.", "")

                    for prompt in prompts:
                        if isinstance(prompt, str) and prompt.strip():
                            behaviors.append({
                                "id": f"garak_{probe_idx}",
                                "behavior": prompt.strip(),
                                "category": category,
                                "source": "garak",
                                "probe_class": f"{modname}.{attr_name}",
                            })
                            probe_idx += 1
                except Exception as e:
                    logger.debug(f"Skipping probe {attr_name}: {e}")

    return behaviors


def _load_garak_fallback() -> list[dict]:
    """Fallback: load from pre-exported file if Garak isn't installed."""
    fallback_path = Path("data/garak/garak_probes_export.json")
    if fallback_path.exists():
        with open(fallback_path) as f:
            return json.load(f)

    logger.warning(
        "No Garak data available. Install Garak or provide "
        "data/garak/garak_probes_export.json"
    )
    return []
