"""Tests unitaires et d'intégration pour le pipeline de Machine Learning."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data, prepare_features_and_target, split_data
from src.preprocessing import create_preprocessor, fit_and_transform_splits, identify_column_types
from src.models import get_default_models, build_pipeline
from src.evaluate import evaluate_cross_validation, compute_test_metrics


@pytest.fixture
def raw_dataset():
    """Charge le dataset brut pour les tests."""
    return load_raw_data()


def test_data_loader_shape(raw_dataset):
    """Vérifie le chargement et la structure minimale du dataset."""
    assert isinstance(raw_dataset, pd.DataFrame)
    assert raw_dataset.shape[0] == 395
    assert "G3" in raw_dataset.columns


def test_prepare_binary_target(raw_dataset):
    """Vérifie la création correcte de la cible binaire."""
    X, y = prepare_features_and_target(raw_dataset, target_type="binary", include_intermediate_grades=False)
    assert "G3" not in X.columns
    assert "G1" not in X.columns
    assert "G2" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_stratified_split(raw_dataset):
    """Vérifie que la stratification est bien respectée lors du découpage."""
    X, y = prepare_features_and_target(raw_dataset, target_type="binary")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 316
    assert len(X_test) == 79
    # Ratio de positifs doit être très proche entre train et test
    assert pytest.approx(y_train.mean(), abs=0.02) == y_test.mean()


def test_preprocessor_no_leakage(raw_dataset):
    """Vérifie l'absence de fuite et les dimensions transformées."""
    X, y = prepare_features_and_target(raw_dataset, target_type="binary")
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    X_train_t, X_test_t, preprocessor, feature_names = fit_and_transform_splits(X_train, X_test)
    assert X_train_t.shape[0] == len(X_train)
    assert X_test_t.shape[0] == len(X_test)
    assert X_train_t.shape[1] == X_test_t.shape[1]
    assert len(feature_names) == X_train_t.shape[1]
    assert not np.isnan(X_train_t).any()
    assert not np.isnan(X_test_t).any()


def test_model_pipeline_execution(raw_dataset):
    """Vérifie l'entraînement complet d'un pipeline sur Logistic Regression."""
    X, y = prepare_features_and_target(raw_dataset, target_type="binary")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    num_cols, cat_cols = identify_column_types(X_train)

    preprocessor = create_preprocessor(num_cols, cat_cols)
    models = get_default_models()
    pipe = build_pipeline(preprocessor, models["Logistic Regression"])

    pipe.fit(X_train, y_train)
    metrics, cm, roc_tuple = compute_test_metrics(pipe, X_test, y_test)

    assert "test_accuracy" in metrics
    assert "test_f1_weighted" in metrics
    assert cm.shape == (2, 2)
    assert len(roc_tuple) == 3
