import os
import pandas as pd
import numpy as np
import pickle
import pytest
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from scalable_ml_pipeline.data.data_processor import DataProcessor


current_dir = os.path.dirname(__file__)
FILE_PATH = os.path.dirname(current_dir)
ENCODER = pickle.load(open(os.path.join(FILE_PATH, "/model/encoder.pkl"), "rb"))
LABEL_BINARIZER = pickle.load(open(os.path.join(FILE_PATH, "model/label_binarizer.pkl"), "rb"))
TEST_DATA_PATH = "tests/test_data/census.csv"


@pytest.fixture
def processor():
    # Use a small subset for testing
    df = pd.read_csv(TEST_DATA_PATH)
    test_path = "data/test_census.csv"
    df.to_csv(test_path, index=False)
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    label = "salary"
    proc = DataProcessor(
        data_path=test_path,
        categorical_features=categorical_features,
        label=label,
        label_binarizer=LABEL_BINARIZER,
        one_hot_encoder=ENCODER
    )
    yield proc
    os.remove(test_path)


def test_load_data(processor):
    df = processor.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_split_data(processor):
    df = processor.load_data()
    train, test = processor.split_data(df, test_size=0.2, random_state=1)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(df)


def test_preprocess_data_train(processor):
    df = processor.load_data()
    x, y, encoder, lb = processor.preprocess_data(df, train=True)
    assert x.shape[0] == df.shape[0]
    assert isinstance(y, np.ndarray)
    assert encoder is not None
    assert lb is not None


def test_preprocess_data_test(processor):
    df = processor.load_data()
    x, y, _, _ = processor.preprocess_data(df, train=False)
    assert x.shape[0] == df.shape[0]
    assert isinstance(y, np.ndarray)


def test_process_method(processor):
    result = processor.process()
    # Should return x_train, y_train, encoder, label_binarizer, x_test, y_test
    assert len(result) == 6
    x_train, y_train, encoder, lb, x_test, y_test = result
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)