from __future__ import annotations

import json
from typing import Any, Dict


CONFIG_PROPOSAL_PROMPT = """You are a quantitative forecasting assistant.

Given monitoring context, propose one safe LightGBM Quantile configuration for a retrain attempt.

Rules:
1. Return valid JSON only.
2. Do not include markdown.
3. Do not reveal chain-of-thought.
4. Do not write or modify Python code.
5. Only choose values inside the provided parameter ranges.
6. Keep the reason concise and evidence-based.

Required JSON schema:
{
  "learning_rate": 0.03,
  "max_depth": 6,
  "num_leaves": 64,
  "min_child_samples": 25,
  "reason": "..."
}
"""


CONFIG_REPAIR_PROMPT = """The previous LightGBM configuration JSON was invalid.

Return a corrected JSON object using only the allowed parameter ranges.
Return valid JSON only, without markdown.

Required JSON schema:
{
  "learning_rate": 0.03,
  "max_depth": 6,
  "num_leaves": 64,
  "min_child_samples": 25,
  "reason": "..."
}
"""


def build_agent_diagnosis_prompt(
    diagnostics: Dict[str, Any],
    improvement_config: Dict[str, Any],
    config_patch_policy: Dict[str, Any],
) -> str:
    _ = improvement_config
    payload = {
        "monitoring_context": diagnostics,
        "parameter_ranges": _parameter_ranges(config_patch_policy),
    }
    return CONFIG_PROPOSAL_PROMPT + "\n\nContext:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_config_patch_repair_prompt(
    previous_plan: Dict[str, Any],
    validation_warnings: list[str],
    diagnostics: Dict[str, Any],
    config_patch_policy: Dict[str, Any],
    base_config: Dict[str, Any],
) -> str:
    _ = base_config
    payload = {
        "previous_config": previous_plan,
        "validation_warnings": validation_warnings,
        "monitoring_context": diagnostics,
        "parameter_ranges": _parameter_ranges(config_patch_policy),
    }
    return CONFIG_REPAIR_PROMPT + "\n\nContext:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _parameter_ranges(policy: Dict[str, Any]) -> Dict[str, list[float]]:
    allowed = policy.get("allowed_patch_keys", {}) if isinstance(policy, dict) else {}
    ranges = {}
    for key in ("learning_rate", "max_depth", "num_leaves", "min_child_samples"):
        bounds = allowed.get(key, {}) if isinstance(allowed, dict) else {}
        ranges[key] = [bounds.get("min"), bounds.get("max")]
    return ranges
