"""Model loading and prediction utilities."""

import joblib
import json


def load_model(model_path):
    """Load XGBoost model from pickle file.

    Args:
        model_path: Path to model .pkl file

    Returns:
        Trained XGBoost model
    """
    return joblib.load(model_path)


def load_scaler(scaler_path):
    """Load fitted StandardScaler.

    Args:
        scaler_path: Path to scaler .pkl file

    Returns:
        Fitted StandardScaler
    """
    return joblib.load(scaler_path)


def load_feature_columns(features_path):
    """Load feature column names from JSON.

    Args:
        features_path: Path to feature_columns.json

    Returns:
        List of feature names
    """
    with open(features_path, "r") as f:
        return json.load(f)


def load_config(config_path):
    """Load model configuration from JSON.

    Args:
        config_path: Path to model_config_full.json

    Returns:
        Dictionary with model config and metrics
    """
    with open(config_path, "r") as f:
        return json.load(f)


def predict(model, scaler, X):
    """Generate predictions with proper scaling.

    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        X: Feature array (n_samples, n_features)

    Returns:
        Predictions array
    """
    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    return predictions
