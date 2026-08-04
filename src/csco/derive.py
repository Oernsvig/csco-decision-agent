"""Pure, stateless feature derivation from raw pipeline scores + spec bands.

Public API
----------
derive(supplier, spec) -> DerivedSupplier
parse_band_expr(expr)  -> (op_str, threshold)   # exposed for unit testing
score_to_band(score, bands_dict, high_key, low_key) -> band_name

Band expression format
----------------------
The specs use two expression shapes:
  ">= 0.55"         -- simple comparison (HIGH / LOW thresholds)
  "0.45-0.54"       -- range descriptor (MEDIUM bands; NOT parsed as a condition,
                        risk_level derivation only consumes HIGH and LOW thresholds)

sub_score_bands use the same pair of simple expressions for ``high`` and ``low``;
the middle band (``med``) is implicit.
"""

import re
from typing import Literal

from csco.models import DerivedSupplier, Tier1Supplier
from csco.specs.schema import Spec

_EXPR_RE = re.compile(r"^(>=|<=|>|<|==)\s*([0-9]+(?:\.[0-9]+)?)$")

_OP_MAP = {">=": "gte", "<=": "lte", ">": "gt", "<": "lt", "==": "eq"}


def parse_band_expr(expr: str) -> tuple[str, float]:
    """Parse a simple comparison expression into (operator_name, threshold).

    Only call this on the HIGH and LOW band expressions.  Range expressions like
    '0.45-0.54' are not valid inputs and will raise ValueError.

    >>> parse_band_expr(">=0.55")
    ('gte', 0.55)
    >>> parse_band_expr("<0.45")
    ('lt', 0.45)
    """
    m = _EXPR_RE.match(expr.strip())
    if not m:
        raise ValueError(
            f"Unparseable band expression: {expr!r}  "
            "(only simple comparisons like '>=0.55' or '<0.45' are supported)"
        )
    return _OP_MAP[m.group(1)], float(m.group(2))


def _satisfies(value: float, op: str, threshold: float) -> bool:
    match op:
        case "gte":
            return value >= threshold
        case "lte":
            return value <= threshold
        case "gt":
            return value > threshold
        case "lt":
            return value < threshold
        case "eq":
            return value == threshold
        case _:
            raise ValueError(f"Unknown operator: {op!r}")


def score_to_band(
    score: float,
    bands: dict[str, str],
    high_key: str = "high",
    low_key: str = "low",
) -> str:
    """Classify a 0–1 score as high / med / low (or HIGH / LOW) using spec band expressions.

    Evaluation order: high → low → middle (implicit).
    Only the HIGH and LOW band expressions are consumed; the middle band expression
    is a range descriptor used for documentation only.
    """
    high_op, high_thresh = parse_band_expr(bands[high_key])
    if _satisfies(score, high_op, high_thresh):
        return high_key

    low_op, low_thresh = parse_band_expr(bands[low_key])
    if _satisfies(score, low_op, low_thresh):
        return low_key

    # Middle band — derive the key name from the available keys
    all_keys = set(bands.keys()) - {high_key, low_key, "calibration"}
    middle_keys = [k for k in all_keys if k not in (high_key, low_key)]
    return middle_keys[0] if middle_keys else "med"


# ---------------------------------------------------------------------------
# Derived field: risk_level
# ---------------------------------------------------------------------------

def derive_risk_level(
    risk_score: float,
    risk_bands: dict[str, str],
) -> Literal["HIGH", "MEDIUM", "LOW"]:
    """Map a 0–1 composite risk score to HIGH / MEDIUM / LOW using spec thresholds.

    risk_bands keys are uppercase: HIGH, MEDIUM, LOW.
    Only the HIGH and LOW expressions are parsed; MEDIUM is implicit.

    Boundary test values (geopolitical, HIGH>=0.55, LOW<0.45):
        0.55 -> HIGH,  0.54 -> MEDIUM,  0.45 -> MEDIUM,  0.44 -> LOW
    Boundary test values (others, HIGH>=0.60, LOW<0.45):
        0.60 -> HIGH,  0.59 -> MEDIUM,  0.45 -> MEDIUM,  0.44 -> LOW
    """
    band = score_to_band(risk_score, risk_bands, high_key="HIGH", low_key="LOW")
    if band == "HIGH":
        return "HIGH"
    if band == "LOW":
        return "LOW"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def derive(supplier: Tier1Supplier, spec: Spec) -> DerivedSupplier:
    """Return a DerivedSupplier with risk_level computed.

    Policy is customised by the general risk bands and the disruption-type
    strategic priority; the type-specific supplier rules consume the structural
    sub-scores directly via the oracle.
    """
    return DerivedSupplier(
        supplier=supplier,
        risk_level=derive_risk_level(supplier.risk_score, spec.risk_bands),
    )
