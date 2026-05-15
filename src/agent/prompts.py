from __future__ import annotations

import json
from typing import Any, Dict


AGENT_RETRAIN_PLAN_PROMPT = """You are a Quant Model Improvement Agent.

Your job is to diagnose evidence and propose a safe retrain plan for a LightGBM Quantile challenger.

Use only this evidence:
- walk-forward validation metrics
- directional accuracy
- interval coverage
- prediction bias
- quantile crossing
- forecast distribution
- drift report
- regime report
- risk report
- news context
- governance history
- deterministic retrain policy output
- allowed config patch keys and ranges

Important:
1. Your decision is advisory. Deterministic policy decides whether retrain planning is opened.
2. Do not interpret ACCURACY_OK as model cleared for use.
3. Use overall_trust_status for final reliability interpretation.
4. Risk, drift, regime, uncertainty calibration, and news/event context may override an accuracy pass.
5. Do not hallucinate news or causal explanations.
6. Do not modify Python source code.
7. Only use allowed config_patch keys from the policy.
8. Config values must be numeric and within allowed min/max.
9. If decision is TRAIN_CHALLENGER, config_patch must be non-empty.
10. Return valid JSON only.
11. Do not include markdown.
12. Do not reveal chain-of-thought.
13. Provide concise evidence-based reason.

Required JSON schema:
{
  "diagnosis": "...",
  "decision": "MONITOR|TRAIN_CHALLENGER|MANUAL_REVIEW",
  "strategy": "NO_ACTION|RETRAIN_RECENT_WINDOW|WIDEN_INTERVAL|STABILIZE_MODEL|REDUCE_OVERFIT|INCREASE_MODEL_CAPACITY|ADD_REGULARIZATION",
  "config_patch": {},
  "reason": "...",
  "confidence": 0.0,
  "evidence_used": []
}
"""


CONFIG_PATCH_REPAIR_PROMPT = """You are repairing an invalid LightGBM config patch.

Rules:
1. Previous config_patch was rejected by validation.
2. Use validation warnings to fix the patch.
3. Only use allowed keys and numeric values inside min/max ranges.
4. Keep diagnosis, decision, and strategy unless they are inconsistent with producing a safe patch.
5. If no safe patch can be produced, return decision MANUAL_REVIEW and an empty config_patch.
6. Return valid JSON only.
7. Do not include markdown.
8. Do not reveal chain-of-thought.

Required JSON schema:
{
  "diagnosis": "...",
  "decision": "MONITOR|TRAIN_CHALLENGER|MANUAL_REVIEW",
  "strategy": "NO_ACTION|RETRAIN_RECENT_WINDOW|WIDEN_INTERVAL|STABILIZE_MODEL|REDUCE_OVERFIT|INCREASE_MODEL_CAPACITY|ADD_REGULARIZATION",
  "config_patch": {},
  "reason": "...",
  "confidence": 0.0,
  "evidence_used": []
}
"""


def build_agent_diagnosis_prompt(
    diagnostics: Dict[str, Any],
    improvement_config: Dict[str, Any],
    config_patch_policy: Dict[str, Any],
) -> str:
    allowed_patch_keys = list((config_patch_policy.get("allowed_patch_keys") or {}).keys())
    payload = {
        "diagnostics": diagnostics,
        "trigger_rules": improvement_config.get("trigger_rules", {}),
        "allowed_decisions": ["MONITOR", "TRAIN_CHALLENGER", "MANUAL_REVIEW"],
        "allowed_diagnoses": improvement_config.get("allowed_diagnoses", []),
        "allowed_strategies": improvement_config.get("allowed_strategies", []),
        "allowed_config_patch_keys": allowed_patch_keys,
        "config_patch_policy": config_patch_policy,
    }
    return (
        AGENT_RETRAIN_PLAN_PROMPT
        + "\n\nUse this evidence and policy. Return one JSON object only:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_config_patch_repair_prompt(
    previous_plan: Dict[str, Any],
    validation_warnings: list[str],
    diagnostics: Dict[str, Any],
    config_patch_policy: Dict[str, Any],
    base_config: Dict[str, Any],
) -> str:
    payload = {
        "previous_plan": previous_plan,
        "validation_warnings": validation_warnings,
        "diagnostics": diagnostics,
        "config_patch_policy": config_patch_policy,
        "base_config": base_config,
    }
    return (
        CONFIG_PATCH_REPAIR_PROMPT
        + "\n\nRepair this plan. Return one JSON object only:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
