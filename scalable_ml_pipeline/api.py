import argparse
from importlib.resources import files
import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from scalable_ml_pipeline.data.data_processor import DataProcessor
from scalable_ml_pipeline.model_helper.model_helper import ModelHelper


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

MODEL_DIR = files("scalable-ml-pipeline").joinpath("models")
MODEL = pickle.load(open(MODEL_DIR.joinpath("trained_model.pkl"), "rb"))
ENCODER = pickle.load(open(MODEL_DIR.joinpath("encoder.pkl"), "rb"))
LABEL_BINARIZER = pickle.load(open(MODEL_DIR.joinpath("label_binarizer.pkl"), "rb"))

class InputData(BaseModel):
    # Using the random row of census.csv as sample
    age: int = Field(None, example=49)
    workclass: str = Field(None, example='Private')
    fnlgt: int = Field(None, example=160187)
    education: str = Field(None, example='9th')
    education_num: int = Field(None, example=5)
    marital_status: str = Field(None, example='Married-spouse-absent')
    occupation: str = Field(None, example='Other-service')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='Black')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=16)
    native_country: str = Field(None, example='Jamaica')


@app.post("/predict")
async def predict(input_data: InputData):
    """
    Endpoint to predict the salary based on input data.
    """
    if not input_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input data is required."
        )
    
    input_df = pd.DataFrame([input_data.dict()])
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processor = DataProcessor(
        data_path=None,  # Not used here, but required for initialization
        categorical_features=categorical_features,
        label=None,  # Not used here, but required for initialization
        label_binarizer=LABEL_BINARIZER,
        one_hot_encoder=ENCODER
    )

    processed_data = data_processor.process_test_data(input_df)

    prediction = ModelHelper.model_inference(MODEL, processed_data[0])[0]

    predicted_salary = '<=50k' if prediction == 0 else '>50k'

    return JSONResponse(
        content={
            "predicted_salary": predicted_salary,
        },
        status_code=status.HTTP_200_OK
    )


def main():

    parser = argparse.ArgumentParser("Scalable ML Pipeline on Census Data")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for the webservice 'default: localhost'"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="port for the sebservice 'default: 8000'"
    )
    args = parser.parse_args()

    uvicorn.run(
        "scalable_ml_pipeline.api:app",
        host=args.host,
        port=args.port,
    )