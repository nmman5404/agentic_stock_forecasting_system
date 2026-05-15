from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


FORBIDDEN_KEY_FRAGMENTS = (
    "path",
    "file",
    "import",
    "code",
    "env",
    "exec",
    "eval",
    "subprocess",
    "os.",
    "sys.",
)


def validate_config_patch(
    patch: dict,
    policy: dict,
    base_config: dict,
    decision: Optional[str] = None,
) -> Tuple[dict, List[str], bool]:
    warnings: List[str] = []
    validated_patch: Dict[str, Any] = {}

    patch = patch or {}
    if not isinstance(patch, dict):
        return {}, ["Config patch must be a flat dictionary."], False

    allowed_patch_keys = policy.get("allowed_patch_keys", {})
    policy_rules = policy.get("policy", {})
    if not isinstance(allowed_patch_keys, dict):
        return {}, ["Config patch policy is missing allowed_patch_keys."], False
    if not isinstance(policy_rules, dict):
        policy_rules = {}

    require_non_empty = bool(policy_rules.get("require_non_empty_patch_for_train_challenger", False))
    if decision in {"TRAIN_CHALLENGER", "RETRAIN_RECENT_WINDOW"} and require_non_empty and not patch:
        return {}, ["TRAIN_CHALLENGER/RETRAIN_RECENT_WINDOW requires a non-empty config patch."], False

    reject_unknown = bool(policy_rules.get("reject_unknown_keys", True))
    clamp_out_of_range = bool(policy_rules.get("clamp_out_of_range_values", False))

    for key, value in patch.items():
        key_text = str(key)
        key_lower = key_text.lower()

        if any(fragment in key_lower for fragment in FORBIDDEN_KEY_FRAGMENTS):
            warnings.append(f"Rejected unsafe patch key: {key_text}.")
            return {}, warnings, False

        if key_text not in allowed_patch_keys:
            message = f"Unknown config patch key rejected: {key_text}."
            warnings.append(message)
            if reject_unknown:
                return {}, warnings, False
            continue

        if isinstance(value, (dict, list, tuple, set)):
            warnings.append(f"Nested or collection patch value rejected for key: {key_text}.")
            return {}, warnings, False

        if isinstance(value, bool) or value is None:
            warnings.append(f"Non-numeric patch value rejected for key: {key_text}.")
            return {}, warnings, False

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            warnings.append(f"Non-numeric patch value rejected for key: {key_text}.")
            return {}, warnings, False

        bounds = allowed_patch_keys.get(key_text, {})
        min_value = bounds.get("min")
        max_value = bounds.get("max")
        if min_value is not None and numeric_value < float(min_value):
            if not clamp_out_of_range:
                warnings.append(f"Patch value for {key_text} below minimum {min_value}: {numeric_value}.")
                return {}, warnings, False
            numeric_value = float(min_value)
            warnings.append(f"Patch value for {key_text} clamped to minimum {min_value}.")

        if max_value is not None and numeric_value > float(max_value):
            if not clamp_out_of_range:
                warnings.append(f"Patch value for {key_text} above maximum {max_value}: {numeric_value}.")
                return {}, warnings, False
            numeric_value = float(max_value)
            warnings.append(f"Patch value for {key_text} clamped to maximum {max_value}.")

        validated_patch[key_text] = _preserve_numeric_type(key_text, numeric_value, base_config)

    return validated_patch, warnings, True


def _preserve_numeric_type(key: str, value: float, base_config: Dict[str, Any]) -> Any:
    if key == "train_window_days":
        return int(round(value))

    params = base_config.get("lightgbm_params", {}) if isinstance(base_config, dict) else {}
    current_value = params.get(key) if isinstance(params, dict) else None
    integer_keys = {"max_depth", "num_leaves", "n_estimators", "min_child_samples"}
    if key in integer_keys or isinstance(current_value, int):
        return int(round(value))
    return float(value)
