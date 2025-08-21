# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for processing data for machine learning tasks, including loading, preprocessing, and splitting datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class DataProcessor:
    """
    A class to handle data loading, preprocessing, and splitting for machine learning tasks.
    """
    def __init__(
            self,
            data_path,
            categorical_features=[],
            label=None,
    ) -> None:
        """
        Initialize the DataProcessor with the path to the data, categorical features, and label.
        Attributes:
            data_path (str): Path to the data file (CSV or Parquet).
            categorical_features (list): List of categorical feature names.
            label (str): Name of the label column for supervised learning.
        """
        self.data_path = data_path
        self.categorical_features = categorical_features
        self.label = label
        self.label_binarizer = LabelBinarizer()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    
    def load_data(
            self
    ) -> pd.DataFrame:
        """
        Load data from the specified path.
        """
        if self.data_path.endswith('.csv'):
            return pd.read_csv(self.data_path)
        elif self.data_path.endswith('.parquet'):
            return pd.read_parquet(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
    def preprocess_data(
            self,
            data,
            train=True
    ) -> tuple:
        """
        Preprocess the data by handling categorical features and labels.
        Args:
            data (pd.DataFrame): The data to preprocess.
            train (bool): Whether the data is for training or testing.
        Returns:
            tuple: Processed features (X), labels (y), one-hot encoder, and label binarizer.
        """

        if self.label is not None:
            y = data[self.label]
            x = data.drop(columns=[self.label], axis=1)
        else:
            y = np.array([])
            x = data
        
        x_categorical = x[self.categorical_features].values
        x_continuous = x.drop(columns=self.categorical_features, axis=1)

        if train:
            x_categorical_encoded = self.one_hot_encoder.fit_transform(x_categorical)
            y_binarized = self.label_binarizer.fit_transform(y.values).ravel()
        else:
            x_categorical_encoded = self.one_hot_encoder.transform(x_categorical)
            try:
                y_binarized = self.label_binarizer.transform(y.values).ravel()
            except AttributeError:
                pass

        x = np.concatenate(
            [x_continuous.values, x_categorical_encoded], axis=1
        )     
        
        return x, y_binarized, self.one_hot_encoder, self.label_binarizer
    

    def split_data(
            self,
            data,
            test_size=0.2,
            random_state=42
    ) -> tuple:
        """
        Split the data into training and testing sets.
        Args:
            data (pd.DataFrame): The data to split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        Returns:
            tuple: Training and testing data as DataFrames.
        """
        train, test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        return train, test
    

    def process(
            self
    ) -> tuple:
        """
        Main method to process the data.
        Loads the data, splits it into training and testing sets, and preprocesses it.
        Returns:
            tuple: Processed training and testing features (X) and labels (y).
        """
        data = self.load_data()
        train, test = self.split_data(data)
        
        x_train, y_train, encoder, label_binarizer = self.preprocess_data(train, train=True)
        x_test, y_test, _, _ = self.preprocess_data(test, train=False)
        
        return x_train, y_train, encoder, label_binarizer, x_test, y_test
