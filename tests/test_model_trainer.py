import os
import pytest

from scalable_ml_pipeline.model_trainer import ModelTrain


@pytest.fixture
def model_trainer():
    return ModelTrain(
        data_path="tests/test_data/census.csv",
        model_path="tests/test_models",
    )


def test_perform_model_training(model_trainer):
    model_trainer.perform_model_training()
    # Check if model files are created
    assert os.path.exists(
        os.path.join(model_trainer.model_path, "trained_model.pkl")
    )
    assert os.path.exists(
        os.path.join(model_trainer.model_path, "encoder.pkl")
    )
    assert os.path.exists(
        os.path.join(model_trainer.model_path, "label_binarizer.pkl")
    )


def test_invalid_data_path():
    with pytest.raises(FileNotFoundError):
        trainer = ModelTrain(
            data_path="invalid/path/to/data.csv",
            model_path="temp",
        )
        trainer.perform_model_training()


def test_invalid_model_path():
    with pytest.raises(Exception):
        trainer = ModelTrain(
            data_path="tests/test_data/census.csv",
            model_path="/invalid/path/to/model",
        )
        trainer.perform_model_training()


def test_perform_slice_eval(tmp_path):
    model_trainer = ModelTrain(
        data_path="tests/test_data/census.csv",
        model_path=tmp_path
    )
    model_trainer.perform_model_training()
    model_trainer.perform_slice_eval(slice_category="education")
    output_file = os.path.join(tmp_path, "slices_education_output.txt")
    assert os.path.exists(output_file)
    with open(output_file, "r") as f:
        content = f.read()
        assert "Slice Category: education" in content
