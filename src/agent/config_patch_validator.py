from __future__ import annotations

from typing import Any, Dict, List, Tuple


DEFAULT_PARAMETER_RANGES: Dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.2),
    "max_depth": (3, 12),
    "num_leaves": (16, 256),
    "min_child_samples": (5, 100),
}

INTEGER_KEYS = {"max_depth", "num_leaves", "min_child_samples"}


def validate_config_patch(
    patch: dict,
    policy: dict | None = None,
    base_config: dict | None = None,
    decision: str | None = None,
) -> Tuple[dict, List[str], bool]:
    """Validate Gemini-proposed LightGBM parameters.

    The validator is intentionally small: allowed key, numeric value, range.
    Gemini handles reasoning; this function only protects the config file.
    """
    _ = base_config, decision
    ranges = _parameter_ranges(policy)
    warnings: List[str] = []
    valid_patch: Dict[str, Any] = {}

    if not isinstance(patch, dict) or not patch:
        return {}, ["Config proposal must be a non-empty object."], False

    for key, value in patch.items():
        if key == "reason":
            continue
        if key not in ranges:
            warnings.append(f"Unsupported parameter: {key}.")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            warnings.append(f"Parameter {key} must be numeric.")
            continue

        low, high = ranges[key]
        if numeric_value < low or numeric_value > high:
            warnings.append(f"Parameter {key}={numeric_value} outside allowed range [{low}, {high}].")
            continue

        valid_patch[key] = int(round(numeric_value)) if key in INTEGER_KEYS else float(numeric_value)

    required = set(ranges)
    missing = sorted(required - set(valid_patch))
    if missing:
        warnings.append(f"Missing required parameters: {', '.join(missing)}.")

    return valid_patch, warnings, not warnings


def _parameter_ranges(policy: dict | None) -> Dict[str, tuple[float, float]]:
    allowed = (policy or {}).get("allowed_patch_keys", {})
    if not isinstance(allowed, dict) or not allowed:
        return dict(DEFAULT_PARAMETER_RANGES)

    ranges: Dict[str, tuple[float, float]] = {}
    for key in DEFAULT_PARAMETER_RANGES:
        bounds = allowed.get(key)
        if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
            ranges[key] = (float(bounds["min"]), float(bounds["max"]))
        else:
            ranges[key] = DEFAULT_PARAMETER_RANGES[key]
    return ranges
