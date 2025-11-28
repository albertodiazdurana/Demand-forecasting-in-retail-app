"""Tests for model_utils module."""

import json
import numpy as np
from unittest.mock import Mock

from model.model_utils import (
    load_feature_columns,
    load_config,
    predict,
)


class TestLoadFeatureColumns:
    """Tests for load_feature_columns function."""

    def test_loads_json_list(self, tmp_path):
        """Should load feature columns from JSON file."""
        features = ["feature1", "feature2", "feature3"]
        file_path = tmp_path / "features.json"
        file_path.write_text(json.dumps(features))

        result = load_feature_columns(file_path)
        assert result == features

    def test_returns_list(self, tmp_path):
        """Should return a list."""
        features = ["col1", "col2"]
        file_path = tmp_path / "features.json"
        file_path.write_text(json.dumps(features))

        result = load_feature_columns(file_path)
        assert isinstance(result, list)

    def test_handles_empty_list(self, tmp_path):
        """Should handle empty feature list."""
        file_path = tmp_path / "features.json"
        file_path.write_text("[]")

        result = load_feature_columns(file_path)
        assert result == []


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_config_dict(self, tmp_path):
        """Should load config from JSON file."""
        config = {
            "model_type": "xgboost",
            "metrics": {"rmse": 6.4, "mae": 3.2},
            "training_samples": 100000,
        }
        file_path = tmp_path / "config.json"
        file_path.write_text(json.dumps(config))

        result = load_config(file_path)
        assert result == config

    def test_returns_dict(self, tmp_path):
        """Should return a dictionary."""
        config = {"key": "value"}
        file_path = tmp_path / "config.json"
        file_path.write_text(json.dumps(config))

        result = load_config(file_path)
        assert isinstance(result, dict)

    def test_preserves_nested_structure(self, tmp_path):
        """Should preserve nested config structure."""
        config = {
            "model": {
                "type": "xgboost",
                "params": {"max_depth": 6, "learning_rate": 0.1},
            }
        }
        file_path = tmp_path / "config.json"
        file_path.write_text(json.dumps(config))

        result = load_config(file_path)
        assert result["model"]["params"]["max_depth"] == 6


class TestPredict:
    """Tests for predict function."""

    def test_scales_and_predicts(self):
        """Should scale features and return predictions."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([10.0, 20.0])

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])

        X = np.array([[100.0, 200.0], [300.0, 400.0]])

        result = predict(mock_model, mock_scaler, X)

        mock_scaler.transform.assert_called_once()
        mock_model.predict.assert_called_once()
        np.testing.assert_array_equal(result, np.array([10.0, 20.0]))

    def test_passes_scaled_features_to_model(self):
        """Should pass scaled features to model.predict."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([5.0])

        mock_scaler = Mock()
        scaled_X = np.array([[0.5, -0.5]])
        mock_scaler.transform.return_value = scaled_X

        X = np.array([[100.0, 200.0]])
        predict(mock_model, mock_scaler, X)

        # Verify model received the scaled values
        call_args = mock_model.predict.call_args[0][0]
        np.testing.assert_array_equal(call_args, scaled_X)

    def test_returns_numpy_array(self):
        """Should return predictions as numpy array."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1], [2], [3]])

        X = np.array([[10], [20], [30]])
        result = predict(mock_model, mock_scaler, X)

        assert isinstance(result, np.ndarray)
