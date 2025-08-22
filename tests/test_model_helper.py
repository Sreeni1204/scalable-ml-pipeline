import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score

from scalable_ml_pipeline.model_helper.model_helper import ModelHelper


@pytest.fixture
def setup():
    # Sample data for testing
    x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[1, 2], [4, 5]])
    y_test = np.array([0, 1])
    
    # Create an instance of ModelHelper
    model_helper = ModelHelper()
    
    return model_helper, x_train, y_train, x_test, y_test

def test_initialization(setup):
    model_helper, _, _, _, _ = setup
    assert isinstance(model_helper.model, RandomForestClassifier)

def test_train_model(setup):
    model_helper, x_train, y_train, _, _ = setup
    trained_model = model_helper.train_model(x_train, y_train)
    assert trained_model is not None
    assert hasattr(trained_model, "predict")

def test_model_inference(setup):
    model_helper, x_train, y_train, x_test, _ = setup
    model_helper.train_model(x_train, y_train)
    predictions = model_helper.model_inference(x_test)
    assert len(predictions) == len(x_test)
    assert all(pred in [0, 1] for pred in predictions)  # Assuming binary classification

def test_evaluate_model(setup):
    model_helper, x_train, y_train, x_test, y_test = setup
    model_helper.train_model(x_train, y_train)
    predictions = model_helper.model_inference(x_test)
    precision, recall, fbeta = model_helper.evaluate_model(y_test, predictions)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
