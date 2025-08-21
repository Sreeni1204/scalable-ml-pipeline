# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_helper.py
Helper class for model training, inference, and evaluation.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


class ModelHelper:
    """
    A helper class for managing machine learning models.
    This class provides methods for training a model, making predictions,
    and evaluating model performance using precision, recall, and F-beta score.
    """
    def __init__(
            self,
            model=None
    ) -> None:
        """
        Initializes the ModelHelper with a specified model.
        If no model is provided, a RandomForestClassifier is used by default.
        Attributes:
            model: An instance of a machine learning model (default is RandomForestClassifier).
        """

        self.model = model if model else RandomForestClassifier()

    def train_model(
            self,
            x_train,
            y_train
    ) -> RandomForestClassifier:
        """
        Trains the model using the provided training data.
        Args:
            x_train: Features for training the model.
            y_train: Target labels for training the model.
        Returns:
            The trained model instance.
        """
        return self.model.fit(x_train, y_train)

    def model_inference(
            self,
            x
    ) -> list:
        """
        Makes predictions using the trained model.
        Args:
            x: Features for which predictions are to be made.
        Returns:
            A list of predictions made by the model.
        """
        return self.model.predict(x)

    def evaluate_model(
            self,
            y_true,
            y_pred
    ) -> tuple:
        """
        Evaluates the model's performance using precision, recall, and F-beta score.
        Args:
            y_true: True labels for the data.
            y_pred: Predicted labels by the model.
        Returns:
            A tuple containing precision, recall, and F-beta score.
        """
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        fbeta = fbeta_score(y_true, y_pred, beta=1.0, zero_division=1)

        return precision, recall, fbeta
