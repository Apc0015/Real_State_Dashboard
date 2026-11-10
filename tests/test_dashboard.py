import os
import pytest
import numpy as np
import pandas as pd
import app


@pytest.fixture(scope="module")
def raw_dataset():
    """Load the raw dataset once for all tests."""
    df = app.load_data()
    assert df is not None, "Real_Estate.csv could not be loaded."
    return df


@pytest.fixture(scope="module")
def preprocessed_dataset(raw_dataset):
    """Preprocess the dataset once for tests."""
    return app.preprocess_data(raw_dataset)


@pytest.fixture(scope="module")
def trained_artifacts(preprocessed_dataset):
    """Train the model once and return all returned components."""
    model, scaler, feature_names, metrics, y_test, y_pred = app.train_model(preprocessed_dataset)
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred
    }


class TestDataLoading:
    """Test suite for data loading and environment resolution."""

    def test_load_data_success(self, raw_dataset):
        assert isinstance(raw_dataset, pd.DataFrame)
        assert len(raw_dataset) == 414
        expected_columns = [
            'Transaction date',
            'House age',
            'Distance to the nearest MRT station',
            'Number of convenience stores',
            'Latitude',
            'Longitude',
            'House price of unit area'
        ]
        for col in expected_columns:
            assert col in raw_dataset.columns

    def test_load_data_invalid_path(self):
        missing_df = app.load_data(filepath='non_existent_file.csv')
        assert missing_df is None

    def test_get_data_path_env_override(self, monkeypatch):
        custom_path = "/tmp/custom_dataset.csv"
        monkeypatch.setenv("DATA_PATH", custom_path)
        assert app.get_data_path() == custom_path


class TestFeaturePreprocessing:
    """Test suite for feature engineering and transformation pipeline."""

    def test_temporal_feature_extraction(self, preprocessed_dataset):
        assert 'year' in preprocessed_dataset.columns
        assert 'month' in preprocessed_dataset.columns
        assert 'day' in preprocessed_dataset.columns
        assert set(preprocessed_dataset['year'].unique()).issubset({2012, 2013})
        assert preprocessed_dataset['month'].between(1, 12).all()
        assert preprocessed_dataset['day'].between(1, 31).all()

    def test_distance_category_binning(self, preprocessed_dataset):
        assert 'distance_category' in preprocessed_dataset.columns
        expected_categories = {'Very Close', 'Close', 'Moderate', 'Far'}
        assert set(preprocessed_dataset['distance_category'].dropna().unique()).issubset(expected_categories)

        # Boundary check
        very_close = preprocessed_dataset[preprocessed_dataset['Distance to the nearest MRT station'] <= 500]
        assert (very_close['distance_category'] == 'Very Close').all()

    def test_convenience_score_normalization(self, preprocessed_dataset):
        assert 'convenience_score' in preprocessed_dataset.columns
        assert preprocessed_dataset['convenience_score'].min() >= 0.0
        assert preprocessed_dataset['convenience_score'].max() <= 1.0

    def test_spatial_zone_binning(self, preprocessed_dataset):
        assert 'lat_zone' in preprocessed_dataset.columns
        assert 'long_zone' in preprocessed_dataset.columns
        assert len(preprocessed_dataset['lat_zone'].cat.categories) == 5
        assert len(preprocessed_dataset['long_zone'].cat.categories) == 5

    def test_prepare_features_structure(self, preprocessed_dataset):
        X, y = app.prepare_features(preprocessed_dataset)
        assert len(X) == len(preprocessed_dataset)
        assert len(y) == len(preprocessed_dataset)
        assert X.isna().sum().sum() == 0
        assert 'House price of unit area' not in X.columns
        # 9 numerical + 4 distance bins + 5 lat bins + 5 long bins = 23 columns
        assert X.shape[1] == 23


class TestModelTrainingAndEvaluation:
    """Test suite for model fitting, scoring, and metrics."""

    def test_train_model_artifacts(self, trained_artifacts):
        assert trained_artifacts['model'] is not None
        assert trained_artifacts['scaler'] is not None
        assert len(trained_artifacts['feature_names']) == 23
        assert len(trained_artifacts['y_test']) == len(trained_artifacts['y_pred'])

    def test_model_metrics_validity(self, trained_artifacts):
        metrics = trained_artifacts['metrics']
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R²' in metrics

        # Realistic bounds for the Sindian real estate dataset
        assert np.isfinite(metrics['RMSE']) and metrics['RMSE'] > 0
        assert np.isfinite(metrics['MAE']) and metrics['MAE'] > 0
        assert 0.40 <= metrics['R²'] <= 0.85

    def test_feature_importances(self, trained_artifacts):
        model = trained_artifacts['model']
        feature_names = trained_artifacts['feature_names']
        importances = model.feature_importances_
        assert len(importances) == len(feature_names)
        assert np.isclose(np.sum(importances), 1.0)

        # Distance to MRT and House Age are historically the top factors
        top_feature_indices = np.argsort(importances)[::-1][:3]
        top_feature_names = [feature_names[i] for i in top_feature_indices]
        assert any('MRT' in name or 'House age' in name for name in top_feature_names)


class TestPredictionPipeline:
    """Test suite for single-record valuation input preparation and inference."""

    def test_build_prediction_input_distance_categories(self, trained_artifacts):
        feature_names = trained_artifacts['feature_names']

        # 300m -> Very Close
        inp_vc = app.build_prediction_input(
            house_age=10, mrt_distance=300, convenience_stores=5,
            latitude=24.98, longitude=121.53, year=2013, month=6, day=15,
            feature_names=feature_names, max_stores=10
        )
        assert inp_vc['distance_category_Very Close'].iloc[0] == 1
        assert inp_vc['distance_category_Far'].iloc[0] == 0

        # 800m -> Close
        inp_c = app.build_prediction_input(
            house_age=10, mrt_distance=800, convenience_stores=5,
            latitude=24.98, longitude=121.53, year=2013, month=6, day=15,
            feature_names=feature_names, max_stores=10
        )
        assert inp_c['distance_category_Close'].iloc[0] == 1

        # 1500m -> Moderate
        inp_m = app.build_prediction_input(
            house_age=10, mrt_distance=1500, convenience_stores=5,
            latitude=24.98, longitude=121.53, year=2013, month=6, day=15,
            feature_names=feature_names, max_stores=10
        )
        assert inp_m['distance_category_Moderate'].iloc[0] == 1

        # 3500m -> Far
        inp_f = app.build_prediction_input(
            house_age=10, mrt_distance=3500, convenience_stores=5,
            latitude=24.98, longitude=121.53, year=2013, month=6, day=15,
            feature_names=feature_names, max_stores=10
        )
        assert inp_f['distance_category_Far'].iloc[0] == 1

    def test_predict_price_realistic_value(self, trained_artifacts):
        model = trained_artifacts['model']
        scaler = trained_artifacts['scaler']
        feature_names = trained_artifacts['feature_names']

        sample_input = app.build_prediction_input(
            house_age=5.0,
            mrt_distance=350.0,
            convenience_stores=7,
            latitude=24.97,
            longitude=121.54,
            year=2013,
            month=5,
            day=10,
            feature_names=feature_names,
            max_stores=10
        )

        price = app.predict_price(model, scaler, sample_input)
        assert isinstance(price, float)
        assert 10.0 <= price <= 120.0

    def test_predict_price_distance_sensitivity(self, trained_artifacts):
        """Properties closer to transit should generally command higher valuations."""
        model = trained_artifacts['model']
        scaler = trained_artifacts['scaler']
        feature_names = trained_artifacts['feature_names']

        close_to_transit = app.build_prediction_input(
            house_age=10.0,
            mrt_distance=200.0,
            convenience_stores=5,
            latitude=24.98,
            longitude=121.53,
            year=2013,
            month=6,
            day=1,
            feature_names=feature_names,
            max_stores=10
        )
        far_from_transit = app.build_prediction_input(
            house_age=10.0,
            mrt_distance=4000.0,
            convenience_stores=5,
            latitude=24.98,
            longitude=121.53,
            year=2013,
            month=6,
            day=1,
            feature_names=feature_names,
            max_stores=10
        )

        price_close = app.predict_price(model, scaler, close_to_transit)
        price_far = app.predict_price(model, scaler, far_from_transit)
        assert price_close > price_far

    def test_predict_price_determinism(self, trained_artifacts):
        model = trained_artifacts['model']
        scaler = trained_artifacts['scaler']
        feature_names = trained_artifacts['feature_names']

        inp = app.build_prediction_input(
            house_age=15.0,
            mrt_distance=1200.0,
            convenience_stores=3,
            latitude=24.96,
            longitude=121.51,
            year=2012,
            month=11,
            day=20,
            feature_names=feature_names,
            max_stores=10
        )

        p1 = app.predict_price(model, scaler, inp)
        p2 = app.predict_price(model, scaler, inp)
        assert p1 == p2
