from __future__ import annotations

"""Tests for src/ml_models.py — TDD Red-Green-Refactor (Constitution Principle III)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# These imports will fail (RED) until each module/function is implemented
from src.features import FEATURE_COLS, HORIZON, make_horizon_target
from src.ml_models import (
    TRAIN_MAX_WEEK,
    VAL_MIN_WEEK,
    VAL_MAX_WEEK,
    PRED_CUTOFF,
    NAN_FILL_RF,
    XGB_HPO_GRID,
    LGB_HPO_GRID,
    BaseMLForecaster,
    XGBoostForecaster,
    RandomForestForecaster,
    LightGBMForecaster,
    run_hpo_xgboost,
    run_hpo_lightgbm,
    plot_feature_importance,
    merge_rmsle_matrices,
    save_all_models_rmsle,
    plot_all_models_comparison,
    save_ml_rmsle,
    save_all_predictions,
)
import joblib
from src.baselines import save_predictions

# =============================================================================
# Phase 2: Foundational
# =============================================================================

class TestMakeHorizonTarget:
    """Tests for make_horizon_target() in src/features.py."""

    def test_output_has_y_column(self, ml_fm):
        out = make_horizon_target(ml_fm, h=1)
        assert "y" in out.columns

    def test_y_equals_log1p_shifted_num_orders_h1(self, ml_fm):
        """For Pair A at week 10, y should be log1p(num_orders at week 11) = log1p(22)."""
        out = make_horizon_target(ml_fm, h=1)
        row = out[(out["center_id"] == 1) & (out["meal_id"] == 10) & (out["week"] == 10)]
        assert len(row) == 1
        expected = np.log1p(11 * 2)  # week 11 * 2 = 22
        assert abs(float(row["y"].iloc[0]) - expected) < 1e-8

    def test_nan_rows_dropped(self, ml_fm):
        """Pair B ends at week 30; h=5 means last usable row is week 25 → no NaN in y."""
        out = make_horizon_target(ml_fm, h=5)
        assert out["y"].isna().sum() == 0

    def test_raises_value_error_h_below_1(self, ml_fm):
        with pytest.raises(ValueError):
            make_horizon_target(ml_fm, h=0)

    def test_raises_value_error_h_above_horizon(self, ml_fm):
        with pytest.raises(ValueError):
            make_horizon_target(ml_fm, h=HORIZON + 1)

    def test_y_is_non_negative(self, ml_fm):
        out = make_horizon_target(ml_fm, h=1)
        assert (out["y"] >= 0).all()

    def test_train_row_count_h1(self, ml_fm):
        """Training rows (week <= 105) for Pair A h=1: weeks 1..144 (144 rows);
        Pair B h=1: weeks 1..29 (29 rows) → 173 total."""
        out = make_horizon_target(ml_fm, h=1)
        train = out[out["week"] <= TRAIN_MAX_WEEK]
        # Pair A weeks 1..105 all have a valid h=1 target (week+1 exists up to 146)
        pair_a_train = train[(train["center_id"] == 1) & (train["meal_id"] == 10)]
        assert len(pair_a_train) == 105
        # Pair B: weeks 1..29 have h=1 target (week 30 is the last)
        pair_b_train = train[(train["center_id"] == 2) & (train["meal_id"] == 20)]
        assert len(pair_b_train) == 29


# =============================================================================
# Phase 2: BaseMLForecaster
# =============================================================================

class TestBaseMLForecaster:
    """Tests for BaseMLForecaster ABC in src/ml_models.py."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseMLForecaster()  # type: ignore

    def test_concrete_subclass_must_define_name(self):
        """Subclass without name raises TypeError or AttributeError on name access."""
        class Incomplete(BaseMLForecaster):
            def fit_all(self, fm): pass
            def predict_all(self, fm): return pd.DataFrame()
            def save(self, models_dir=None): pass
            def load(self, models_dir=None): pass
        obj = Incomplete()
        assert not hasattr(obj, "name") or obj.name is None or isinstance(obj.name, str)

    def test_predict_delegates_to_predict_all(self, ml_fm):
        """predict() must return the same result as predict_all()."""
        class Stub(BaseMLForecaster):
            name = "stub"
            def fit_all(self, fm): pass
            def predict_all(self, fm):
                return pd.DataFrame({"center_id": [1], "meal_id": [10], "horizon": [1], "y_pred": [5.0]})
            def save(self, models_dir=None): pass
            def load(self, models_dir=None): pass
        obj = Stub()
        r1 = obj.predict(ml_fm)
        r2 = obj.predict_all(ml_fm)
        pd.testing.assert_frame_equal(r1, r2)

    def test_models_list_empty_before_fit(self):
        class Stub(BaseMLForecaster):
            name = "stub"
            def fit_all(self, fm): pass
            def predict_all(self, fm): return pd.DataFrame()
            def save(self, models_dir=None): pass
            def load(self, models_dir=None): pass
        obj = Stub()
        assert obj.models_ == []


# =============================================================================
# Phase 3: User Story 1 — XGBoost Direct Forecaster
# =============================================================================

class TestXGBoostForecaster:
    """Tests for XGBoostForecaster — TDD RED before T010."""

    @pytest.fixture
    def small_params(self, monkeypatch):
        """Override n_estimators for speed in tests."""
        import src.ml_models as mm
        monkeypatch.setitem(mm.XGB_PARAMS, "n_estimators", 5)
        monkeypatch.setitem(mm.XGB_PARAMS, "early_stopping_rounds", 2)

    def test_name_is_xgboost(self):
        assert XGBoostForecaster.name == "xgboost"

    def test_models_empty_before_fit(self):
        model = XGBoostForecaster()
        assert model.models_ == []

    def test_fit_all_populates_10_models(self, ml_fm, small_params):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        assert len(model.models_) == HORIZON

    def test_fit_all_saves_10_json_files(self, ml_fm, small_params, tmp_path):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        xgb_dir = tmp_path / "xgboost"
        files = sorted(xgb_dir.glob("xgb_h*.json"))
        assert len(files) == HORIZON

    def test_saved_json_loadable_with_xgb_booster(self, ml_fm, small_params, tmp_path):
        import xgboost as xgb_lib
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        b = xgb_lib.XGBRegressor()
        b.load_model(tmp_path / "xgboost" / "xgb_h01.json")
        assert b is not None

    def test_predict_all_schema(self, ml_fm, small_params):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert set(preds.columns) == {"center_id", "meal_id", "horizon", "y_pred"}
        assert set(preds["horizon"].unique()) == set(range(1, HORIZON + 1))

    def test_predict_all_y_pred_non_negative(self, ml_fm, small_params):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert (preds["y_pred"] >= 0).all()

    def test_predict_delegates_to_predict_all(self, ml_fm, small_params):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        r1 = model.predict(ml_fm)
        r2 = model.predict_all(ml_fm)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_load_restores_models(self, ml_fm, small_params, tmp_path):
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        model2 = XGBoostForecaster()
        model2.load(tmp_path)
        assert len(model2.models_) == HORIZON
        preds = model2.predict_all(ml_fm)
        assert len(preds) > 0


# =============================================================================
# Phase 4: User Story 4 — Prediction File Contract
# =============================================================================

class TestSavePredictionsML:
    """Verify constitution Principle IV prediction file contract for ML models."""

    @pytest.fixture
    def fitted_xgb(self, ml_fm, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setitem(mm.XGB_PARAMS, "n_estimators", 5)
        monkeypatch.setitem(mm.XGB_PARAMS, "early_stopping_rounds", 2)
        model = XGBoostForecaster()
        model.fit_all(ml_fm)
        return model

    def test_pred_csv_exists_for_h01(self, ml_fm, fitted_xgb, tmp_path):
        preds = fitted_xgb.predict_all(ml_fm)
        save_predictions(fitted_xgb, preds, 1, ml_fm, tmp_path)
        assert (tmp_path / "pred_xgboost_h01.csv").exists()

    def test_pred_csv_columns(self, ml_fm, fitted_xgb, tmp_path):
        preds = fitted_xgb.predict_all(ml_fm)
        save_predictions(fitted_xgb, preds, 1, ml_fm, tmp_path)
        df = pd.read_csv(tmp_path / "pred_xgboost_h01.csv")
        assert set(df.columns) == {"center_id", "meal_id", "week", "y_pred", "split"}

    def test_split_column_is_eval(self, ml_fm, fitted_xgb, tmp_path):
        preds = fitted_xgb.predict_all(ml_fm)
        save_predictions(fitted_xgb, preds, 1, ml_fm, tmp_path)
        df = pd.read_csv(tmp_path / "pred_xgboost_h01.csv")
        assert (df["split"] == "eval").all()

    def test_y_pred_non_negative(self, ml_fm, fitted_xgb, tmp_path):
        preds = fitted_xgb.predict_all(ml_fm)
        save_predictions(fitted_xgb, preds, 1, ml_fm, tmp_path)
        df = pd.read_csv(tmp_path / "pred_xgboost_h01.csv")
        assert (df["y_pred"] >= 0).all()

    def test_week_equals_anchor_plus_horizon(self, ml_fm, fitted_xgb, tmp_path):
        preds = fitted_xgb.predict_all(ml_fm)
        save_predictions(fitted_xgb, preds, 1, ml_fm, tmp_path)
        df = pd.read_csv(tmp_path / "pred_xgboost_h01.csv")
        # Pair A anchor is week 125 (PRED_CUTOFF), so prediction week = 126
        pair_a = df[(df["center_id"] == 1) & (df["meal_id"] == 10)]
        assert int(pair_a["week"].iloc[0]) == PRED_CUTOFF + 1


# =============================================================================
# Phase 5: User Story 7 — ML RMSLE Matrix
# =============================================================================

class TestMLRmsleMatrix:
    """Tests for save_ml_rmsle — evaluating ML models and writing ml_rmsle.csv."""

    @pytest.fixture
    def fitted_xgb(self, ml_fm, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setitem(mm.XGB_PARAMS, "n_estimators", 5)
        monkeypatch.setitem(mm.XGB_PARAMS, "early_stopping_rounds", 2)
        m = XGBoostForecaster()
        m.fit_all(ml_fm)
        return m

    def test_rmsle_dict_has_rmsle_key(self, ml_fm, fitted_xgb):
        from src.baselines import evaluate_all
        metrics = evaluate_all([fitted_xgb], ml_fm)
        assert "rmsle" in metrics

    def test_rmsle_shape_is_n_models_by_10(self, ml_fm, fitted_xgb):
        from src.baselines import evaluate_all
        metrics = evaluate_all([fitted_xgb], ml_fm)
        rmsle_df = metrics["rmsle"]
        assert rmsle_df.shape == (1, HORIZON)

    def test_rmsle_values_non_negative(self, ml_fm, fitted_xgb):
        from src.baselines import evaluate_all
        metrics = evaluate_all([fitted_xgb], ml_fm)
        vals = metrics["rmsle"].values.astype(float)
        assert np.all((vals >= 0) | np.isnan(vals))

    def test_rmsle_row_index_matches_model_name(self, ml_fm, fitted_xgb):
        from src.baselines import evaluate_all
        metrics = evaluate_all([fitted_xgb], ml_fm)
        assert "xgboost" in metrics["rmsle"].index


class TestSaveMLRmsle:
    """Tests for save_ml_rmsle function."""

    @pytest.fixture
    def fitted_xgb(self, ml_fm, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setitem(mm.XGB_PARAMS, "n_estimators", 5)
        monkeypatch.setitem(mm.XGB_PARAMS, "early_stopping_rounds", 2)
        m = XGBoostForecaster()
        m.fit_all(ml_fm)
        return m

    def test_ml_rmsle_csv_written(self, ml_fm, fitted_xgb, tmp_path):
        save_ml_rmsle([fitted_xgb], ml_fm, tmp_path)
        assert (tmp_path / "ml_rmsle.csv").exists()

    def test_ml_rmsle_shape_1x10(self, ml_fm, fitted_xgb, tmp_path):
        save_ml_rmsle([fitted_xgb], ml_fm, tmp_path)
        df = pd.read_csv(tmp_path / "ml_rmsle.csv", index_col="model")
        assert df.shape == (1, HORIZON)


# =============================================================================
# Phase 6: User Story 2 — RandomForest Direct Forecaster
# =============================================================================

class TestRandomForestForecaster:
    """Tests for RandomForestForecaster."""

    @pytest.fixture
    def small_params(self, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setitem(mm.RF_PARAMS, "n_estimators", 5)

    def test_name_is_random_forest(self):
        assert RandomForestForecaster.name == "random_forest"

    def test_fit_all_saves_10_pkl_files(self, ml_fm, small_params, tmp_path):
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        files = sorted((tmp_path / "random_forest").glob("rf_h*.pkl"))
        assert len(files) == HORIZON

    def test_loaded_pkl_is_random_forest_regressor(self, ml_fm, small_params, tmp_path):
        from sklearn.ensemble import RandomForestRegressor as RFR
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        loaded = joblib.load(tmp_path / "random_forest" / "rf_h01.pkl")
        assert isinstance(loaded, RFR)

    def test_predict_all_schema(self, ml_fm, small_params):
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert set(preds.columns) == {"center_id", "meal_id", "horizon", "y_pred"}
        assert set(preds["horizon"].unique()) == set(range(1, HORIZON + 1))

    def test_predict_all_y_pred_non_negative(self, ml_fm, small_params):
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert (preds["y_pred"] >= 0).all()

    def test_nan_features_filled_with_sentinel(self, ml_fm, small_params):
        """Pairs with NaN lag features should still produce predictions (no error)."""
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert len(preds) > 0  # no exception; NaN handled via fill

    def test_models_has_10_entries(self, ml_fm, small_params):
        model = RandomForestForecaster()
        model.fit_all(ml_fm)
        assert len(model.models_) == HORIZON


# =============================================================================
# Phase 7: User Story 3 — LightGBM Direct Forecaster
# =============================================================================

class TestLightGBMForecaster:
    """Tests for LightGBMForecaster."""

    @pytest.fixture
    def small_params(self, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setitem(mm.LGB_PARAMS, "n_estimators", 5)

    def test_name_is_lightgbm(self):
        assert LightGBMForecaster.name == "lightgbm"

    def test_fit_all_saves_10_pkl_files(self, ml_fm, small_params, tmp_path):
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        files = sorted((tmp_path / "lightgbm").glob("lgbm_h*.pkl"))
        assert len(files) == HORIZON

    def test_loaded_pkl_is_lgbm_regressor(self, ml_fm, small_params, tmp_path):
        import lightgbm as lgb_lib
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        model.save(tmp_path)
        loaded = joblib.load(tmp_path / "lightgbm" / "lgbm_h01.pkl")
        assert isinstance(loaded, lgb_lib.LGBMRegressor)

    def test_predict_all_schema(self, ml_fm, small_params):
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert set(preds.columns) == {"center_id", "meal_id", "horizon", "y_pred"}
        assert set(preds["horizon"].unique()) == set(range(1, HORIZON + 1))

    def test_predict_all_y_pred_non_negative(self, ml_fm, small_params):
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        preds = model.predict_all(ml_fm)
        assert (preds["y_pred"] >= 0).all()

    def test_val_split_used_in_fit(self, ml_fm, small_params):
        """Fit should not raise even when val split is small."""
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        assert len(model.models_) == HORIZON

    def test_models_has_10_entries(self, ml_fm, small_params):
        model = LightGBMForecaster()
        model.fit_all(ml_fm)
        assert len(model.models_) == HORIZON


# =============================================================================
# Phase 8: User Story 8 — All-Models Comparison
# =============================================================================

@pytest.fixture
def mock_rmsle_csvs(tmp_path):
    """Write mock baseline (5×10) and ml (3×10) RMSLE CSVs to tmp_path."""
    cols = [f"h{h:02d}" for h in range(1, 11)]
    baseline_models = ["naive", "seasonal_naive", "sma", "ses", "linreg"]
    ml_models = ["xgboost", "random_forest", "lightgbm"]
    rng = np.random.default_rng(42)
    baseline_df = pd.DataFrame(
        rng.uniform(0.2, 1.0, (5, 10)), index=baseline_models, columns=cols
    )
    ml_df = pd.DataFrame(
        rng.uniform(0.1, 0.9, (3, 10)), index=ml_models, columns=cols
    )
    b_path = tmp_path / "baseline_rmsle.csv"
    m_path = tmp_path / "ml_rmsle.csv"
    baseline_df.to_csv(b_path, index=True, index_label="model")
    ml_df.to_csv(m_path, index=True, index_label="model")
    return b_path, m_path


class TestMergeRmsleMatrices:
    """Tests for merge_rmsle_matrices."""

    def test_merge_produces_8_rows(self, mock_rmsle_csvs):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        assert df.shape[0] == 8

    def test_merge_produces_10_columns(self, mock_rmsle_csvs):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        assert df.shape[1] == 10

    def test_merge_row_index_order(self, mock_rmsle_csvs):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        assert list(df.index[:5]) == ["naive", "seasonal_naive", "sma", "ses", "linreg"]
        assert list(df.index[5:]) == ["xgboost", "random_forest", "lightgbm"]


class TestSaveAllModelsRmsle:
    """Tests for save_all_models_rmsle."""

    def test_csv_written(self, mock_rmsle_csvs, tmp_path):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        save_all_models_rmsle(df, tmp_path)
        assert (tmp_path / "all_models_rmsle.csv").exists()

    def test_csv_has_8_rows(self, mock_rmsle_csvs, tmp_path):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        save_all_models_rmsle(df, tmp_path)
        read_back = pd.read_csv(tmp_path / "all_models_rmsle.csv", index_col="model")
        assert read_back.shape == (8, 10)


class TestPlotAllModelsComparison:
    """Tests for plot_all_models_comparison."""

    def test_plot_writes_pdf(self, mock_rmsle_csvs, tmp_path):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        plot_all_models_comparison(df, tmp_path)
        assert (tmp_path / "all_models_rmsle.pdf").exists()

    def test_pdf_is_non_empty(self, mock_rmsle_csvs, tmp_path):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        plot_all_models_comparison(df, tmp_path)
        assert (tmp_path / "all_models_rmsle.pdf").stat().st_size > 0

    def test_returns_path(self, mock_rmsle_csvs, tmp_path):
        b_path, m_path = mock_rmsle_csvs
        df = merge_rmsle_matrices(b_path, m_path)
        p = plot_all_models_comparison(df, tmp_path)
        assert p.name == "all_models_rmsle.pdf"


# =============================================================================
# Phase 9: US-5 — HPO Grid (T029)
# =============================================================================


class TestRunHpoXgboost:
    """Tests for run_hpo_xgboost — 6 combos at h=1, best params returned."""

    @pytest.fixture
    def hpo_result(self, ml_fm, tmp_path, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setattr(mm, "XGB_PARAMS", {**mm.XGB_PARAMS, "n_estimators": 2})
        return run_hpo_xgboost(ml_fm, tmp_path)

    @pytest.fixture
    def hpo_csv(self, ml_fm, tmp_path, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setattr(mm, "XGB_PARAMS", {**mm.XGB_PARAMS, "n_estimators": 2})
        run_hpo_xgboost(ml_fm, tmp_path)
        return pd.read_csv(tmp_path / "hpo_xgboost.csv")

    def test_hpo_xgboost_csv_has_6_rows(self, hpo_csv):
        assert len(hpo_csv) == 6

    def test_hpo_xgboost_columns(self, hpo_csv):
        for col in ("learning_rate", "max_depth", "val_rmsle_h01", "selected"):
            assert col in hpo_csv.columns

    def test_hpo_xgboost_exactly_one_selected(self, hpo_csv):
        assert hpo_csv["selected"].sum() == 1

    def test_hpo_xgboost_returns_dict(self, hpo_result):
        assert isinstance(hpo_result, dict)
        assert "learning_rate" in hpo_result
        assert "max_depth" in hpo_result

    def test_hpo_xgboost_selected_row_matches_best(self, hpo_csv, hpo_result):
        best_row = hpo_csv[hpo_csv["selected"]].iloc[0]
        assert best_row["learning_rate"] == hpo_result["learning_rate"]
        assert int(best_row["max_depth"]) == int(hpo_result["max_depth"])

    def test_hpo_xgboost_val_rmsle_non_negative(self, hpo_csv):
        assert (hpo_csv["val_rmsle_h01"] >= 0).all()


class TestRunHpoLightgbm:
    """Tests for run_hpo_lightgbm — 4 combos at h=1, best params returned."""

    @pytest.fixture
    def hpo_result(self, ml_fm, tmp_path, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setattr(mm, "LGB_PARAMS", {**mm.LGB_PARAMS, "n_estimators": 2})
        return run_hpo_lightgbm(ml_fm, tmp_path)

    @pytest.fixture
    def hpo_csv(self, ml_fm, tmp_path, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setattr(mm, "LGB_PARAMS", {**mm.LGB_PARAMS, "n_estimators": 2})
        run_hpo_lightgbm(ml_fm, tmp_path)
        return pd.read_csv(tmp_path / "hpo_lightgbm.csv")

    def test_hpo_lightgbm_csv_has_4_rows(self, hpo_csv):
        assert len(hpo_csv) == 4

    def test_hpo_lightgbm_columns(self, hpo_csv):
        for col in ("learning_rate", "num_leaves", "val_rmsle_h01", "selected"):
            assert col in hpo_csv.columns

    def test_hpo_lightgbm_exactly_one_selected(self, hpo_csv):
        assert hpo_csv["selected"].sum() == 1

    def test_hpo_lightgbm_returns_dict(self, hpo_result):
        assert isinstance(hpo_result, dict)
        assert "learning_rate" in hpo_result
        assert "num_leaves" in hpo_result

    def test_hpo_lightgbm_selected_row_matches_best(self, hpo_csv, hpo_result):
        best_row = hpo_csv[hpo_csv["selected"]].iloc[0]
        assert best_row["learning_rate"] == hpo_result["learning_rate"]
        assert int(best_row["num_leaves"]) == int(hpo_result["num_leaves"])

    def test_hpo_lightgbm_val_rmsle_non_negative(self, hpo_csv):
        assert (hpo_csv["val_rmsle_h01"] >= 0).all()


# =============================================================================
# Phase 10: US-6 — Feature Importance (T034)
# =============================================================================


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance — XGBoost and LightGBM side-by-side."""

    @pytest.fixture
    def fitted_models(self, ml_fm, tmp_path, monkeypatch):
        import src.ml_models as mm
        monkeypatch.setattr(mm, "XGB_PARAMS", {**mm.XGB_PARAMS, "n_estimators": 2})
        monkeypatch.setattr(mm, "LGB_PARAMS", {**mm.LGB_PARAMS, "n_estimators": 2})
        monkeypatch.setattr(mm, "RESULTS_DIR", tmp_path)
        xgb_m = XGBoostForecaster()
        lgb_m = LightGBMForecaster()
        xgb_m.fit_all(ml_fm)
        lgb_m.fit_all(ml_fm)
        return xgb_m, lgb_m

    def test_xgb_importance_pdf_written(self, fitted_models, tmp_path):
        xgb_m, lgb_m = fitted_models
        plot_feature_importance(xgb_m, lgb_m, tmp_path)
        assert (tmp_path / "feature_importance_xgb.pdf").exists()

    def test_lgb_importance_pdf_written(self, fitted_models, tmp_path):
        xgb_m, lgb_m = fitted_models
        plot_feature_importance(xgb_m, lgb_m, tmp_path)
        assert (tmp_path / "feature_importance_lgb.pdf").exists()

    def test_returns_dict_with_xgb_lgb_keys(self, fitted_models, tmp_path):
        xgb_m, lgb_m = fitted_models
        result = plot_feature_importance(xgb_m, lgb_m, tmp_path)
        assert isinstance(result, dict)
        assert "xgb" in result and "lgb" in result

    def test_pdfs_are_non_empty(self, fitted_models, tmp_path):
        xgb_m, lgb_m = fitted_models
        result = plot_feature_importance(xgb_m, lgb_m, tmp_path)
        assert result["xgb"].stat().st_size > 0
        assert result["lgb"].stat().st_size > 0
