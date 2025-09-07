# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A utility module for interacting with S3 storage using DVC.
Includes functions to pull model files from remote storage.
"""

import subprocess


def pull_model_from_dvc():
    """
    Pull the model file from the remote storage using DVC.
    Assumes that the DVC remote is already configured.
    """
    try:
        # Run the DVC pull command
        subprocess.run(["dvc", "pull"], check=True)
        print("Model pulled successfully from DVC remote.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull model from DVC remote: {e}")
