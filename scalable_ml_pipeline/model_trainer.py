# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for training machine learning models using preprocessed data.
"""

import argparse
import os
import pickle
import pandas as pd
import sys

from scalable_ml_pipeline.data.data_processor import DataProcessor
from scalable_ml_pipeline.model_helper.model_helper import ModelHelper


class ModelTrain():
    """
    A class to handle the training of machine learning models.
    This class initializes with the data path and model path, processes the data,
    trains the model, evaluates it, and saves the trained model along with encoders
    and label binarizers to the specified model path.
    """
    def __init__(
            self,
            data_path: str,
            model_path: str,
    ) -> None:
        """
        Initialize the ModelTrain class.
        Args:
            data_path (str): Path to the data csv file.
            model_path (str): Path to save the trained models.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.label = "salary"
        self.data_preprocessor = DataProcessor(
            data_path=self.data_path,
            categorical_features=self.categorical_features,
            label=self.label,
        )
        self.model_helper = ModelHelper()
    

    def perform_model_training(
            self
    ) -> None:
        """
        Function to perform model training.
        """
        x_train, y_train, encoder, label_binarizer, x_test, y_test = self.data_preprocessor.process()

        
        trained_model = self.model_helper.train_model(x_train, y_train)
        evaluate_model = self.model_helper.evaluate_model(y_test, self.model_helper.model_inference(x_test))

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        model_file_path = os.path.join(self.model_path, "trained_model.pkl")
        with open(model_file_path, "wb") as model_file:
            pickle.dump(trained_model, model_file)
        encoder_file_path = os.path.join(self.model_path, "encoder.pkl")
        with open(encoder_file_path, "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)
        label_binarizer_file_path = os.path.join(self.model_path, "label_binarizer.pkl")
        with open(label_binarizer_file_path, "wb") as label_binarizer_file:
            pickle.dump(label_binarizer, label_binarizer_file)
        print(f"Model trained and saved to {model_file_path}")

        print(f"Evaluation Metrics: Precision: {evaluate_model[0]}, Recall: {evaluate_model[1]}, F-beta Score: {evaluate_model[2]}")
    

    def perform_slice_training(
            self,
            slice_category: str
    ) -> None:
        """
        Function to perform model training with a specific slice category.
        
        Args:
            slice_category (str): The category to slice the data for training.
        """
        x_train, y_train, encoder, label_binarizer, x_test, y_test = self.data_preprocessor.process()

        model_file = os.path.join(self.model_path, f"trained_model.pkl")
        if os.path.exists(model_file):
            with open(model_file, "rb") as model_file:
                trained_model = pickle.load(model_file)
        else:
            print(f"No pre-trained model found at {model_file}. Training a new model.")
            trained_model = None
        
        predictions = self.model_helper.model_inference(trained_model, x_test)

        data = self.data_preprocessor.get_data()

        standard_output = sys.stdout
        slices_output_file = os.path.join(self.model_path, f"slices_{slice_category}_output.txt")

        with open(slices_output_file, "w") as slices_output:
            sys.stdout = slices_output
            print(f"Slice Category: {slice_category}")
            for value in data[slice_category].unique():
                index = data.index[data[slice_category] == value]

                print(f"Slice: {value}")
                print(f"Number of samples: {len(index)}")
                predictions_slice = self.model_helper.evaluate_model(y_test[index], predictions[index])
                print(f"Evaluation Metrics for {value}: Precision: {predictions_slice[0]}, Recall: {predictions_slice[1]}, F-beta Score: {predictions_slice[2]}")
                print("-" * 50)

        sys.stdout = standard_output


def main():
    """
    Main function to parse command line arguments and initiate model training.
    """
    parser = argparse.ArgumentParser(description="Run the model trainer.")
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "slice"],
        help="Action to perform: 'train' for training the model, 'slice' for slicing the data.",
    )
    parser.add_argument(
        "data_path",
        type=str,
        required=True,
        help="Path to the data csv file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/",
        help="Path to save the models",
    )
    parser.add_argument(
        "--slice_category",
        type=str,
        default="education",
        help="Provide the slice category for model training",
    )
    args = parser.parse_args()

    model_trainer = ModelTrain(
        data_path=args.data_path,
        model_path=args.model_path,
    )
    if args.action == "train":
        model_trainer.perform_model_training()
    elif args.action == "slice":
        model_trainer.perform_slice_training(
            slice_category=args.slice_category,
        )
