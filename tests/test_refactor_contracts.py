from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.agent.config_patch_validator import validate_config_patch
from src.agent.governance import compare_model_candidates
from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.monitoring.risk_engine import calculate_risk_report


def _market_frame(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120, rows)
    df = pd.DataFrame({"close": close, "volume": np.full(rows, 1_000_000.0)}, index=idx)
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    return df


class RefactorContractsTest(unittest.TestCase):
    def test_drift_report_uses_labels(self):
        reference = _market_frame().tail(60)
        validation = {"metrics": {"mape": 0.09, "directional_accuracy": 0.42, "interval_95_coverage": 0.50}}

        report = detect_drift(reference, reference.copy(), validation)

        self.assertEqual(report["concept_drift_level"], "HIGH")
        self.assertIn("final_drift_label", report)
        self.assertNotIn("recommended_action", report)

    def test_regime_report_has_no_confidence_heuristic(self):
        report = detect_regime(_market_frame())

        self.assertIn("final_regime_label", report)
        self.assertNotIn("regime_confidence", report)

    def test_risk_engine_raises_on_invalid_forecast(self):
        with self.assertRaises(ValueError):
            calculate_risk_report({})

    def test_config_validator_is_range_only(self):
        patch = {"learning_rate": 0.03, "max_depth": 6, "num_leaves": 64, "min_child_samples": 25}
        valid_patch, warnings, valid = validate_config_patch(patch, {}, {})

        self.assertTrue(valid)
        self.assertEqual(warnings, [])
        self.assertEqual(valid_patch["num_leaves"], 64)

    def test_metric_comparison_prefers_lower_mape(self):
        current = {"mape": 0.03, "directional_accuracy": 0.55, "interval_95_coverage": 0.82}
        candidate = {"mape": 0.025, "directional_accuracy": 0.52, "interval_95_coverage": 0.80}

        result = compare_model_candidates(current, candidate)

        self.assertEqual(result["decision"], "SAVE_CANDIDATE_CONFIG")
        self.assertTrue(result["accepted_candidate"])


if __name__ == "__main__":
    unittest.main()
