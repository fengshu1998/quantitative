import json
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_analysis_utils
import signal_utils
from portfolio_selection_utils import allocate_positions, select_ranked_portfolio_candidates


class SignalFusionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.factor_selection_path = self.root / "factor_selection.json"
        self.prediction_path = self.root / "transformer_predictions.csv"
        signal_utils.ALPHA_FACTOR_SELECTION_PATH = self.factor_selection_path
        signal_utils.TRANSFORMER_PREDICTION_PATH = self.prediction_path
        signal_utils.clear_signal_caches()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_factor_selection(self):
        self.factor_selection_path.write_text(
            json.dumps(
                {
                    "factor_selection": [
                        {
                            "factor": "return_20d",
                            "validity": "effective",
                            "direction": "positive",
                            "factor_weight": 2.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_generate_signal_outputs_raw_subsignals_without_fusion_score(self):
        self._write_factor_selection()
        pd.DataFrame(
            [
                {
                    "symbol": "sh600000",
                    "prediction": 0.8,
                    "prediction_zscore": 2.0,
                    "prediction_rank": 1,
                }
            ]
        ).to_csv(self.prediction_path, index=False)

        df = pd.DataFrame(
            [
                {
                    "symbol": "sh600000",
                    "trend": "uptrend",
                    "return_20d": 0.10,
                    "price_vs_ma20": 0.05,
                    "volatility_20d": 20.0,
                    "max_drawdown_20d": -5.0,
                    "volume": 1000000,
                    "pe": 20,
                    "pb": 2,
                }
            ]
        )
        scored = signal_utils.generate_signal(df).iloc[0]

        self.assertEqual(scored["transformer_score"], 2.0)
        self.assertNotIn("signal_score", scored.index)
        self.assertNotIn("fusion_score", scored.index)
        self.assertNotIn("raw_fusion_score", scored.index)
        self.assertNotIn("fusion_weights", scored.index)
        self.assertEqual(scored["signal"], "BUY")

    def test_missing_transformer_does_not_break_signal_generation(self):
        self._write_factor_selection()
        df = pd.DataFrame(
            [
                {
                    "symbol": "sh600001",
                    "trend": "uptrend",
                    "return_20d": 0.10,
                    "price_vs_ma20": 0.05,
                    "volatility_20d": 20.0,
                    "max_drawdown_20d": -5.0,
                    "volume": 1000000,
                }
            ]
        )
        scored = signal_utils.generate_signal(df).iloc[0]

        self.assertEqual(scored["transformer_score"], 0.0)
        self.assertIn(scored["signal"], {"BUY", "HOLD", "SELL"})

    def test_candidate_selection_uses_cross_section_score_and_position_has_no_transformer_overlay(self):
        all_factor_data = {
            "sh600000": {
                "name": "A",
                "data": pd.DataFrame(
                    [{"signal": "BUY", "cross_section_score": 3.0, "volatility_20d": 10.0}]
                ),
            },
            "sh600001": {
                "name": "B",
                "data": pd.DataFrame(
                    [{"signal": "BUY", "cross_section_score": 5.0, "volatility_20d": 20.0}]
                ),
            },
        }

        selected = select_ranked_portfolio_candidates(all_factor_data)[:2]
        self.assertEqual(selected[0]["symbol"], "sh600001")

        allocated = allocate_positions(selected, max_position_per_stock=0.10, max_total_position=0.60)
        self.assertTrue(all(item["transformer_weight_adjustment"] == 0.0 for item in allocated))
        self.assertTrue(all(item["target_weight"] <= 0.10 for item in allocated))

    def test_cross_sectional_standardization_adds_rank_and_buckets(self):
        self.factor_selection_path.write_text(
            json.dumps(
                {
                    "factor_selection": [
                        {
                            "factor": "return_20d",
                            "validity": "effective",
                            "direction": "positive",
                            "factor_weight": 1.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        signal_utils.clear_signal_caches()
        all_factor_data = {}
        for idx in range(10):
            symbol = f"sh6000{idx:02d}"
            all_factor_data[symbol] = {
                "name": symbol,
                "data": pd.DataFrame(
                    [
                        {
                            "date": "2026-06-08",
                            "symbol": symbol,
                            "return_20d": idx if idx < 9 else 1000,
                            "alpha_score": idx,
                            "transformer_score": idx % 3,
                            "risk_liquidity_score": 9 - idx,
                            "industry": "A" if idx < 5 else "B",
                            "float_market_cap": 100 + idx * 100,
                            "total_market_cap": 100 + idx * 100,
                        }
                    ]
                ),
            }

        scored = signal_utils.apply_cross_sectional_signal_scores(all_factor_data)
        latest = pd.concat([item["data"].tail(1) for item in scored.values()], ignore_index=True)

        self.assertTrue(latest["cross_section_rank_pct"].between(0, 1).all())
        self.assertNotIn("signal_score", latest.columns)
        self.assertNotIn("fusion_score", latest.columns)
        self.assertNotIn("raw_fusion_score", latest.columns)
        for column in [
            "alpha_cross_section_score",
            "transformer_cross_section_score",
            "risk_liquidity_cross_section_score",
        ]:
            self.assertIn(column, latest.columns)
        expected = (
            latest["alpha_cross_section_score"] * 0.7
            + latest["transformer_cross_section_score"] * 0.2
            + latest["risk_liquidity_cross_section_score"] * 0.1
        ).round(6)
        pd.testing.assert_series_equal(latest["cross_section_score"], expected, check_names=False)
        self.assertEqual(int(latest["long_bucket"].sum()), 2)
        self.assertEqual(int(latest["short_bucket"].sum()), 2)
        self.assertAlmostEqual(float(latest["return_20d_zscore"].mean()), 0.0, places=6)
        industry_means = latest.groupby("industry")["return_20d_zscore"].mean().abs()
        self.assertTrue((industry_means < 1e-6).all())
        cap_corr = latest["return_20d_neutralized"].corr(latest["float_market_cap"].map(lambda value: np.log(value)))
        self.assertLess(abs(float(cap_corr)), 0.05)

    def test_ranked_selection_and_constraints_use_top_quantile_without_shorting(self):
        all_factor_data = {}
        for idx in range(10):
            symbol = f"sh6001{idx:02d}"
            all_factor_data[symbol] = {
                "name": symbol,
                "data": pd.DataFrame(
                    [
                        {
                            "signal": "SELL" if idx < 2 else "HOLD",
                            "cross_section_score": idx / 10,
                            "cross_section_rank_pct": (idx + 1) / 10,
                            "long_bucket": idx >= 8,
                            "short_bucket": idx < 2,
                            "volatility_20d": 100.0 if idx == 9 else 20.0,
                            "industry": "Tech" if idx >= 8 else "Other",
                        }
                    ]
                ),
            }

        selected = select_ranked_portfolio_candidates(all_factor_data)
        self.assertEqual([item["symbol"] for item in selected], ["sh600109", "sh600108"])
        self.assertTrue(all(not item["short_bucket"] for item in selected))

        allocated = allocate_positions(selected, max_position_per_stock=0.10, max_total_position=0.60)
        self.assertTrue(all(item["target_weight"] <= 0.10 for item in allocated))
        self.assertLessEqual(sum(item["target_weight"] for item in allocated), 0.60)
        high_vol = next(item for item in allocated if item["symbol"] == "sh600109")
        low_vol = next(item for item in allocated if item["symbol"] == "sh600108")
        self.assertLessEqual(high_vol["volatility_weight_adjustment"], low_vol["volatility_weight_adjustment"])


class AlphaFeedbackTest(unittest.TestCase):
    def test_live_feedback_multiplier_adjusts_weight_without_direction_change(self):
        stats = [
            {
                "factor": "return_20d",
                "sample_count": 1000,
                "missing_rate": 0.0,
                "ic": 0.1,
                "rank_ic": 0.1,
                "ic_ir": 0.5,
                "rank_ic_ir": 0.5,
                "long_short_return": 0.02,
            }
        ]
        feedback = {"factors": {"return_20d": {"feedback_multiplier": 0.8}}}

        selection = alpha_analysis_utils.build_factor_selection(stats, live_feedback=feedback)[0]

        self.assertEqual(selection["direction"], "positive")
        self.assertLess(selection["factor_weight"], selection["base_factor_weight"])
        self.assertEqual(selection["feedback_multiplier"], 0.8)


if __name__ == "__main__":
    unittest.main()
