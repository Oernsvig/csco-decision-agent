"""Load *.spec.yaml files into typed Spec objects."""

import logging
from pathlib import Path

import yaml

from csco.specs.schema import DisruptionType, Spec

logger = logging.getLogger(__name__)

_SPECS_DIR = Path(__file__).parent.parent.parent.parent / "specs"

_ALL_TYPES: list[DisruptionType] = [
    "geopolitical",
    "natural_disaster",
    "economic",
    "cyber",
    "labour",
]


def load_spec(disruption_type: DisruptionType, specs_dir: Path = _SPECS_DIR) -> Spec:
    path = specs_dir / f"{disruption_type}.spec.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = Spec.model_validate(raw)
    logger.debug("Loaded spec '%s' from %s", disruption_type, path)
    return spec


def load_all_specs(specs_dir: Path = _SPECS_DIR) -> dict[DisruptionType, Spec]:
    result: dict[DisruptionType, Spec] = {}
    for dt in _ALL_TYPES:
        result[dt] = load_spec(dt, specs_dir)
    logger.info("Loaded %d specs from %s", len(result), specs_dir)
    return result
