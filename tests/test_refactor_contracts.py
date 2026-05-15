from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.agent.config_patch_validator import validate_config_patch
from src.agent.governance import compare_model_candidates
from src.modeling.validation import WalkForwardConfig, run_walk_forward_validation
from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.risk.risk_engine import calculate_risk_report


def _market_frame(rows: int = 120, drift: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120 + drift, rows)
    volume = np.full(rows, 1_000_000.0)
    df = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    df["ma_7"] = df["close"].rolling(7).mean().bfill()
    df["ma_14"] = df["close"].rolling(14).mean().bfill()
    return df


class RefactorContractsTest(unittest.TestCase):
    def test_drift_report_has_evidence_scores(self):
        reference = _market_frame().tail(60).copy()
        current = reference.copy()
        validation = {"metrics": {"mape": 0.02, "directional_accuracy": 0.65, "interval_95_coverage": 0.85}}

        report = detect_drift(reference, current, validation)

        self.assertIn("feature_score", report)
        self.assertIn("target_score", report)
        self.assertIn("concept_score", report)
        self.assertIn("feature_drift_detected", report)
        self.assertEqual(report["severity"], "LOW")

    def test_strong_concept_drift_increases_severity(self):
        reference = _market_frame().tail(60).copy()
        current = reference.copy()
        validation = {"metrics": {"mape": 0.09, "directional_accuracy": 0.42, "interval_95_coverage": 0.50}}

        report = detect_drift(reference, current, validation)

        self.assertTrue(report["concept_drift_detected"])
        self.assertEqual(report["severity"], "HIGH")

    def test_regime_components_and_labels(self):
        report = detect_regime(_market_frame())

        self.assertIn(report["trend_regime"], {"UPTREND", "DOWNTREND", "SIDEWAYS", "MIXED_TREND"})
        self.assertIn("volume_regime", report)
        self.assertIn("final_regime_label", report)
        self.assertNotEqual(report["trend_regime"], "REVERSAL" + "_RISK")
        self.assertEqual(report["liquidity_regime"], report["volume_regime"])

    def test_risk_engine_is_pure_measurement(self):
        forecast = {
            "current_price": 100,
            "forecasts": [{"q_0.025": 97, "q_0.5": 102, "q_0.975": 108}],
        }
        report = calculate_risk_report(forecast, drift_report={"severity": "LOW"}, regime_report={})

        self.assertIn("risk_level", report)
        self.assertNotIn("preliminary" + "_signal", report)
        self.assertNotIn("signal" + "_confidence", report)
        self.assertEqual(calculate_risk_report({}, None)["risk_level"], "EXTREME_RISK")
        self.assertEqual(
            calculate_risk_report(forecast, drift_report={"severity": "HIGH"}, regime_report={})["risk_level"],
            "EXTREME_RISK",
        )

    def test_walk_forward_validation_output_contract(self):
        df = _market_frame(rows=90)
        result = run_walk_forward_validation(
            df,
            config=WalkForwardConfig(initial_train_size=40, validation_window=5, step_size=10, max_windows=2),
            model_params={"n_estimators": 5, "learning_rate": 0.05, "num_leaves": 8, "verbose": -1},
        )

        self.assertEqual(result["evaluation_method"], "walk_forward")
        self.assertIn("metrics", result)
        self.assertIn("folds", result)
        self.assertGreaterEqual(result["fold_count"], 1)

    def test_config_patch_validator_rejects_unknown_and_accepts_valid(self):
        policy = {
            "allowed_patch_keys": {"learning_rate": {"min": 0.005, "max": 0.1}},
            "policy": {"reject_unknown_keys": True, "require_non_empty_patch_for_train_challenger": True},
        }
        base = {"lightgbm_params": {"learning_rate": 0.05}}

        _, warnings, valid = validate_config_patch({"unknown": 1}, policy, base, decision="TRAIN_CHALLENGER")
        self.assertFalse(valid)
        self.assertTrue(warnings)

        patch, warnings, valid = validate_config_patch({"learning_rate": 0.03}, policy, base, decision="TRAIN_CHALLENGER")
        self.assertTrue(valid)
        self.assertEqual(patch["learning_rate"], 0.03)
        self.assertEqual(warnings, [])

    def test_governance_uses_walk_forward_metrics_and_risk_labels(self):
        champion = {"mape": 0.04, "directional_accuracy": 0.55, "interval_95_coverage": 0.82, "risk_level": "LOW_RISK"}
        challenger = {"mape": 0.03, "directional_accuracy": 0.57, "interval_95_coverage": 0.86, "risk_level": "LOW_RISK"}
        config = {
            "promotion_rules": {
                "max_allowed_mape_for_promotion": 0.05,
                "min_directional_accuracy_for_promotion": 0.50,
                "min_interval_95_coverage_for_promotion": 0.80,
            }
        }

        decision = compare_model_candidates(champion, challenger, config)

        self.assertEqual(decision["decision"], "PROMOTE_CHALLENGER")
        self.assertTrue(decision["accepted_challenger"])


if __name__ == "__main__":
    unittest.main()
