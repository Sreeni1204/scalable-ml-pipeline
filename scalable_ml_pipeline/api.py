import argparse
import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional

from scalable_ml_pipeline.data.data_processor import DataProcessor
from scalable_ml_pipeline.model_helper.model_helper import ModelHelper
from scalable_ml_pipeline.model_helper.s3_utils import pull_model_from_dvc


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

class AppConfig(BaseModel):
    run_location: str = "render"  # Default to 'render', can be set to 'local' or other environments

# Initialize app configuration
config = AppConfig()

@app.on_event("startup")
async def startup_event():
    """
    Startup event to load the model and encoders.
    Event pulls the model from DVC remote storage.
    """
    print("Starting up and loading model...")
    try:
        pull_model_from_dvc()
        # Set model path based on run location
        if config.run_location == "render":
            model_path = "/opt/render/project/src/model"
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../model")
            model_path = os.path.abspath(model_path)

        # Check if the model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Set the model path as a global variable
        global MODEL_PATH
        MODEL_PATH = model_path

        global MODEL, ENCODER, LABEL_BINARIZER
        MODEL = pickle.load(open(os.path.join(MODEL_PATH, "trained_model.pkl"), "rb"))
        ENCODER = pickle.load(open(os.path.join(MODEL_PATH, "encoder.pkl"), "rb"))
        LABEL_BINARIZER = pickle.load(open(os.path.join(MODEL_PATH, "label_binarizer.pkl"), "rb"))

        print(f"Model path set to: {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to pull model from DVC remote: {e}")


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
async def predict(input_data: List[InputData]):
    """
    Endpoint to predict the salary based on input data.
    """
    if not input_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input data is empty."
        )
    
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded"
        )
    
    input_df = pd.DataFrame([data.dict() for data in input_data])
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    data_processor = DataProcessor(
        data_path=None,  # Not used here, but required for initialization
        categorical_features=categorical_features,
        label=None,  # Not used here, but required for initialization
        label_binarizer=LABEL_BINARIZER,
        one_hot_encoder=ENCODER
    )

    processed_data = data_processor.process_test_data(input_df)

    model_helper = ModelHelper(model=MODEL)
    predictions = model_helper.model_inference(processed_data[0])

    predicted_salary = ['<=50k' if pred == 0 else '>50k' for pred in predictions]

    return JSONResponse(
        content={
            "predicted_salary": predicted_salary,
        },
        status_code=status.HTTP_200_OK
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok"}


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
    parser.add_argument(
        "--run_location",
        type=str,
        default="render",
        help="Run location for the application 'default: render', other option is 'local'"
    )
    args = parser.parse_args()

    # Update the config with the parsed arguments
    config.run_location = args.run_location

    uvicorn.run(
        "scalable_ml_pipeline.api:app",
        host=args.host,
        port=args.port,
    )