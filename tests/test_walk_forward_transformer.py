import tempfile
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

sys.modules.setdefault("akshare", types.SimpleNamespace())

import qlib_backtest_utils
import qlib_training_utils


class TransformerLabelFeatureEngineeringTest(unittest.TestCase):
    def test_cross_sectional_labels_keep_raw_and_center_excess(self):
        panel = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-01-02")] * 3,
                "instrument": ["SH600000", "SH600001", "SH600002"],
                "label_5d_forward_return": [0.10, 0.04, -0.02],
            }
        )

        out = qlib_training_utils._add_cross_sectional_labels(panel)

        self.assertIn("label_5d_forward_return", out.columns)
        self.assertAlmostEqual(out["label_5d_cs_excess_return"].mean(), 0.0)
        self.assertTrue(out["label_5d_rank_pct"].between(0, 1).all())

    def test_cross_sectional_zscore_features_are_centered_by_date(self):
        panel = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-01-02")] * 3,
                "instrument": ["SH600000", "SH600001", "SH600002"],
                "return_5d": [1.0, 2.0, 100.0],
            }
        )
        for col in qlib_training_utils.FEATURE_COLUMNS:
            if col not in panel.columns:
                panel[col] = 1.0

        features = qlib_training_utils._build_cross_sectional_feature_frame(panel)

        self.assertIn("cs_z_return_5d", features.columns)
        self.assertAlmostEqual(features["cs_z_return_5d"].mean(), 0.0, places=6)

    def test_feature_selection_removes_missing_constant_and_weak_rankic(self):
        old_threshold = qlib_training_utils.TRANSFORMER_FEATURE_MIN_ABS_RANKIC
        old_missing = qlib_training_utils.TRANSFORMER_FEATURE_MISSING_MAX
        try:
            qlib_training_utils.TRANSFORMER_FEATURE_MIN_ABS_RANKIC = 0.95
            qlib_training_utils.TRANSFORMER_FEATURE_MISSING_MAX = 0.25
            index = pd.MultiIndex.from_product(
                [pd.date_range("2026-01-02", periods=4), ["SH600000", "SH600001", "SH600002", "SH600003"]],
                names=["datetime", "instrument"],
            )
            label = pd.Series([0.4, 0.3, 0.2, 0.1] * 4, index=index)
            feature_df = pd.DataFrame(
                {
                    "good": [4, 3, 2, 1] * 4,
                    "mostly_missing": [None, None, None, 1] * 4,
                    "constant": [1] * len(index),
                    "weak": [1, 3, 2, 4] * 4,
                },
                index=index,
            )

            selected, diagnostics = qlib_training_utils._select_transformer_features(feature_df, label, index)

            self.assertEqual(selected, ["good"])
            self.assertIn("mostly_missing", diagnostics["missing_rate_filter"]["removed"])
            self.assertIn("constant", diagnostics["zero_std_filter"]["removed"])
            self.assertIn("weak", diagnostics["rank_ic_filter"]["removed"])
        finally:
            qlib_training_utils.TRANSFORMER_FEATURE_MIN_ABS_RANKIC = old_threshold
            qlib_training_utils.TRANSFORMER_FEATURE_MISSING_MAX = old_missing

    def test_portfolio_prediction_metrics_include_top20_and_monthly_rankic(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-01-02"] * 5 + ["2026-02-02"] * 5,
                "symbol": [f"sh60000{i}" for i in range(5)] * 2,
                "prediction": [5, 4, 3, 2, 1] * 2,
                "label": [0.05, 0.02, 0.0, -0.01, -0.03] * 2,
                "raw_forward_return": [0.06, 0.03, 0.01, 0.0, -0.02] * 2,
            }
        )

        metrics = qlib_training_utils._portfolio_prediction_metrics(frame)

        self.assertIn("top20_bottom20_return", metrics)
        self.assertIn("top20_excess_return", metrics)
        self.assertIn("monthly_rank_ic_mean", metrics)
        self.assertGreater(metrics["top20_bottom20_return"], 0)
        self.assertGreater(metrics["monthly_rank_ic_positive_rate"], 0)


class WalkForwardWindowTest(unittest.TestCase):
    def test_year_windows_match_production_walk_forward_plan(self):
        windows = qlib_training_utils.build_walk_forward_year_windows(
            start_year=2010,
            end_year=2019,
            train_years=6,
            valid_years=1,
            test_years=1,
        )

        self.assertEqual(windows[0]["train_start"], "2010-01-01")
        self.assertEqual(windows[0]["train_end"], "2015-12-31")
        self.assertEqual(windows[0]["valid_start"], "2016-01-01")
        self.assertEqual(windows[0]["valid_end"], "2016-12-31")
        self.assertEqual(windows[0]["test_start"], "2017-01-01")
        self.assertEqual(windows[0]["test_end"], "2017-12-31")
        self.assertEqual(windows[1]["train_start"], "2011-01-01")
        self.assertEqual(windows[1]["valid_start"], "2017-01-01")
        self.assertEqual(windows[1]["test_start"], "2018-01-01")
        self.assertEqual(windows[2]["train_start"], "2012-01-01")
        self.assertEqual(windows[2]["valid_start"], "2018-01-01")
        self.assertEqual(windows[2]["test_start"], "2019-01-01")

    def test_live_production_segments_use_latest_completed_year_for_validation(self):
        segments = qlib_training_utils.build_live_production_segments(
            latest_date="2026-06-08",
            train_years=6,
            valid_years=1,
        )

        self.assertEqual(segments["train"], ("2019-01-01", "2024-12-31"))
        self.assertEqual(segments["valid"], ("2025-01-01", "2025-12-31"))


class QlibWalkForwardSignalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.static_path = self.root / "transformer_predictions.csv"
        self.walk_path = self.root / "transformer_walk_forward_predictions.csv"
        self.live_dir = self.root / "live_predictions"
        self.old_static = qlib_backtest_utils.TRANSFORMER_PREDICTION_PATH
        self.old_walk = qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_PREDICTION_PATH
        self.old_enabled = qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_ENABLED
        self.old_live_dir = qlib_backtest_utils.TRANSFORMER_LIVE_PREDICTION_DIR
        self.old_live_min = qlib_backtest_utils.TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS
        qlib_backtest_utils.TRANSFORMER_PREDICTION_PATH = self.static_path
        qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_PREDICTION_PATH = self.walk_path
        qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_ENABLED = True
        qlib_backtest_utils.TRANSFORMER_LIVE_PREDICTION_DIR = self.live_dir
        qlib_backtest_utils.TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS = 2

    def tearDown(self):
        qlib_backtest_utils.TRANSFORMER_PREDICTION_PATH = self.old_static
        qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_PREDICTION_PATH = self.old_walk
        qlib_backtest_utils.TRANSFORMER_WALK_FORWARD_ENABLED = self.old_enabled
        qlib_backtest_utils.TRANSFORMER_LIVE_PREDICTION_DIR = self.old_live_dir
        qlib_backtest_utils.TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS = self.old_live_min
        qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA = {}
        self.tmp.cleanup()

    def test_transformer_signal_prefers_walk_forward_predictions(self):
        pd.DataFrame(
            [{"date": "2018-01-02", "symbol": "sh600000", "prediction": 1.0}]
        ).to_csv(self.static_path, index=False)
        pd.DataFrame(
            [
                {
                    "date": "2018-01-02",
                    "symbol": "sh600000",
                    "prediction": 2.0,
                    "fold_id": "test_2018",
                    "test_start": "2018-01-01",
                    "test_end": "2018-12-31",
                }
            ]
        ).to_csv(self.walk_path, index=False)

        signal = qlib_backtest_utils._transformer_prediction_series()

        self.assertEqual(float(signal.iloc[0]), 2.0)
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["prediction_mode"], "walk_forward")
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["fold_count"], 1)
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["first_test_year"], 2018)

    def test_static_prediction_is_fallback_when_walk_forward_missing(self):
        pd.DataFrame(
            [{"date": "2018-01-02", "symbol": "sh600000", "prediction": 1.0}]
        ).to_csv(self.static_path, index=False)

        signal = qlib_backtest_utils._transformer_prediction_series()

        self.assertEqual(float(signal.iloc[0]), 1.0)
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["prediction_mode"], "static")

    def test_live_prediction_snapshots_build_signal_series(self):
        self.live_dir.mkdir()
        pd.DataFrame(
            [{"date": "2026-06-08", "symbol": "sh600000", "prediction": 1.0, "generated_at": "2026-06-08T15:00:00"}]
        ).to_csv(self.live_dir / "2026-06-08.csv", index=False)
        pd.DataFrame(
            [{"date": "2026-06-09", "symbol": "sh600000", "prediction": 2.0, "generated_at": "2026-06-09T15:00:00"}]
        ).to_csv(self.live_dir / "2026-06-09.csv", index=False)

        signal = qlib_backtest_utils._transformer_live_prediction_series()

        self.assertEqual(len(signal), 2)
        self.assertEqual(float(signal.loc[("SH600000", pd.Timestamp("2026-06-09"))]), 2.0)
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["prediction_mode"], "live_snapshot")
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["snapshot_day_count"], 2)

    def test_live_prediction_snapshots_require_min_days(self):
        self.live_dir.mkdir()
        pd.DataFrame(
            [{"date": "2026-06-09", "symbol": "sh600000", "prediction": 2.0}]
        ).to_csv(self.live_dir / "2026-06-09.csv", index=False)

        signal = qlib_backtest_utils._transformer_live_prediction_series()

        self.assertTrue(signal.empty)
        self.assertEqual(qlib_backtest_utils.LAST_TRANSFORMER_SIGNAL_METADATA["snapshot_day_count"], 1)

    def test_live_prediction_snapshot_save_adds_metadata(self):
        old_dir = qlib_training_utils.TRANSFORMER_LIVE_PREDICTION_DIR
        qlib_training_utils.TRANSFORMER_LIVE_PREDICTION_DIR = self.live_dir
        try:
            pred = pd.Series(
                [1.0],
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2026-06-09"), "SH600000")],
                    names=["datetime", "instrument"],
                ),
            )

            path = qlib_training_utils.save_transformer_live_prediction_snapshot(
                pred,
                {"feature_scaler_path": "scaler.json"},
                "2026-06-09T15:01:02",
            )
            saved = pd.read_csv(path)

            self.assertEqual(path.name, "2026-06-09.csv")
            self.assertEqual(saved.iloc[0]["symbol"], "sh600000")
            self.assertEqual(saved.iloc[0]["prediction_mode"], "live_snapshot")
            self.assertEqual(saved.iloc[0]["generated_at"], "2026-06-09T15:01:02")
            self.assertEqual(saved.iloc[0]["feature_scaler_path"], "scaler.json")
        finally:
            qlib_training_utils.TRANSFORMER_LIVE_PREDICTION_DIR = old_dir


class DailyPipelineBehaviorTest(unittest.TestCase):
    def test_main_daily_pipeline_does_not_call_walk_forward_training(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertNotIn("run_transformer_walk_forward_training", source)
        self.assertNotIn("run_transformer_training", source)


class LiveProductionTrainingCompatibilityTest(unittest.TestCase):
    def test_legacy_training_wrapper_delegates_to_live_production_training(self):
        old = qlib_training_utils.run_transformer_live_production_training
        try:
            qlib_training_utils.run_transformer_live_production_training = lambda force=False: {
                "status": "ok",
                "training_mode": "live_production",
                "force": force,
            }

            report = qlib_training_utils.run_transformer_training(force=True)

            self.assertTrue(report["deprecated_alias"])
            self.assertEqual(report["alias_for"], "run_transformer_live_production_training")
            self.assertEqual(report["training_mode"], "live_production")
            self.assertTrue(report["force"])
        finally:
            qlib_training_utils.run_transformer_live_production_training = old


if __name__ == "__main__":
    unittest.main()
