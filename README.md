# Scalable ML Pipeline

This repository contains a scalable machine learning pipeline designed to process data, train models, and deploy them in a production environment. The pipeline is built using modern tools and frameworks to ensure scalability, maintainability, and ease of use.

## Features
- **Data Processing**: Includes utilities for loading, preprocessing, and splitting datasets.
- **Model Training**: Supports training and evaluation of machine learning models.
- **Deployment**: Deploys the trained model as a FastAPI application.
- **CI/CD Integration**: Automated testing, linting, and deployment using GitHub Actions and Render.
- **DVC Integration**: Tracks and manages datasets and model files using DVC with S3 as the remote storage.

## Requirements
- Python 3.10
- Poetry v2.1.4 for dependency management
- AWS credentials for accessing S3

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sreeni1204/scalable-ml-pipeline.git
   cd scalable-ml-pipeline
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Set Up DVC**:
   ```bash
   dvc pull
   ```
   Ensure your AWS credentials are configured for accessing the S3 bucket.

4. **Run the Application**:
   ```bash
   poetry run uvicorn scalable_ml_pipeline.api:app --host 0.0.0.0 --port 8000
   ```

## Testing
Run the test suite using:
```bash
poetry run pytest
```

## Linting
Check code quality using:
```bash
poetry run flake8
```

## Deployment
The application is deployed using Render. Ensure the following environment variables are set in Render:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `RENDER_API_KEY`

## CI/CD Pipeline
The repository includes a GitHub Actions workflow for:
- Running tests and linting
- Pulling data and models using DVC
- Deploying the application to Render

## Directory Structure
```
scalable-ml-pipeline/
├── scalable_ml_pipeline/
│   ├── api.py          # FastAPI application
│   ├── data/           # Data processing utilities
│   ├── model_helper/   # Model utilities
├── tests/              # Test cases
├── .github/workflows/  # CI/CD workflows
├── pyproject.toml      # Poetry configuration
├── dvc.yaml            # DVC pipeline configuration
```

## License
This project is licensed under the MIT License.