# Fraud Detection System on Amazon SageMaker

This project builds and deploys an end-to-end credit card fraud detection system using Amazon SageMaker, AWS Lambda, and Amazon API Gateway.

The work in this repository goes beyond model training. It includes:

- exploratory data analysis on an imbalanced fraud dataset
- preprocessing and train/test split
- XGBoost model training and evaluation
- threshold optimization for business-friendly fraud decisions
- model packaging and deployment to a real SageMaker endpoint
- Lambda integration to expose the model as an API-friendly service
- API Gateway setup for public HTTP access
- autoscaling configuration and production-oriented endpoint testing
- latency and error-handling validation

The notebook used for the full workflow is [fraud_detection_project.ipynb](/home/snehangshu/projects/sagemaker/fraud_detection_project.ipynb).

## Project Overview

The goal of this project is to detect fraudulent credit card transactions with strong recall while keeping false alerts under control. Because the dataset is extremely imbalanced, the focus is not on raw accuracy. Instead, the pipeline is designed around:

- `AUC-PR` for model quality on rare fraud events
- class imbalance handling with `scale_pos_weight`
- threshold tuning to improve fraud-class `F1-score`
- production deployment through AWS managed services

This project was deployed as a complete inference system in AWS:

1. The trained XGBoost model was packaged as `model.tar.gz`.
2. The model artifact was uploaded to Amazon S3.
3. A SageMaker real-time inference endpoint was deployed.
4. An AWS Lambda function was created to validate requests and call the SageMaker endpoint.
5. API Gateway was configured to expose the Lambda function as an HTTP API.
6. SageMaker autoscaling was configured to support production traffic more reliably.
7. The deployed API was tested with real payloads, error scenarios, and load tests.

## Architecture

The deployed architecture is:

`Client -> API Gateway -> AWS Lambda -> SageMaker Endpoint -> Prediction Response`

### Component Responsibilities

- `Amazon SageMaker`
  Hosts the real-time XGBoost fraud detection model.

- `AWS Lambda`
  Accepts incoming JSON payloads, validates feature completeness and types, converts input into CSV format, invokes the SageMaker endpoint, applies the fraud threshold, and returns a structured JSON response.

- `Amazon API Gateway`
  Exposes the Lambda function through a public HTTPS endpoint and enables application or frontend integration.

- `Amazon S3`
  Stores training data, test data, model artifacts, and threshold configuration.

## Dataset

The project uses the well-known credit card fraud detection dataset with anonymized PCA-transformed features.

### Dataset Characteristics

- Total rows: `284,807`
- Features: `30`
- Target column: `Class`
- Legitimate transactions: `284,315`
- Fraud transactions: `492`
- Fraud ratio: `0.1727%`
- Imbalance ratio: about `578:1`

### Input Features

The model expects these 30 features in this exact order:

- `Time`
- `V1` to `V28`
- `Amount`

The feature name order is also stored in [datasets/features_names.json](/home/snehangshu/projects/sagemaker/datasets/features_names.json).

## Exploratory Data Analysis

The notebook shows several important findings before training:

- there are no missing values in the dataset
- fraud is extremely rare, so accuracy alone is misleading
- `Time` and `Amount` require scaling
- PCA-based features `V1` to `V28` are already transformed
- the features most correlated with fraud are `V17`, `V14`, `V12`, and `V10`

### Business-Relevant Statistics

- Mean legitimate transaction amount: `EUR 88.29`
- Mean fraud transaction amount: `EUR 122.21`
- Median legitimate transaction amount: `EUR 22.00`
- Median fraud transaction amount: `EUR 9.25`

## Data Preparation

The preprocessing pipeline in the notebook follows a production-safe approach:

1. Remove temporary analysis-only columns if present.
2. Split the data into features and target.
3. Apply stratified train/test split to preserve the fraud ratio.
4. Scale only `Amount` and `Time` using `StandardScaler`.
5. Fit the scaler on training data only to avoid leakage.
6. Export headerless CSV files for SageMaker built-in XGBoost compatibility.

### Split Details

- Training rows: `227,845`
- Test rows: `56,962`
- Training fraud cases: `394`
- Test fraud cases: `98`

## Model Training

The model used is `XGBoost` for binary classification.

### Training Configuration

- objective: `binary:logistic`
- eval metric: `aucpr`
- estimators: `500`
- max depth: `6`
- learning rate: `0.1`
- gamma: `1`
- min child weight: `5`
- subsample: `0.8`
- colsample by tree: `0.8`
- early stopping rounds: `30`
- random state: `42`
- class imbalance handling: `scale_pos_weight = 577.3`

### Why XGBoost

XGBoost is a strong choice here because it:

- performs well on tabular classification problems
- handles non-linear patterns effectively
- supports imbalance-aware weighting
- integrates cleanly with SageMaker built-in serving

## Model Performance

### Initial Evaluation at Threshold `0.50`

- ROC-AUC: `0.9553`
- PR-AUC: `0.7171`
- True negatives: `56,614`
- False positives: `250`
- False negatives: `16`
- True positives: `82`

Fraud-class performance at threshold `0.50`:

- Precision: `0.2470`
- Recall: `0.8367`
- F1-score: `0.3814`

This threshold catches many fraud cases, but produces too many false alarms.

### Threshold Optimization

The fraud decision threshold was tuned across a range of values to maximize fraud-class `F1-score`.

- Best threshold found: `0.73`
- Best fraud-class F1-score: `0.8000`

Fraud-class performance at threshold `0.73`:

- Precision: `0.8506`
- Recall: `0.7551`
- F1-score: `0.8000`

This threshold gives a much better balance between catching fraud and limiting unnecessary reviews.

The selected threshold is stored in [optimal_threshold.json](/home/snehangshu/projects/sagemaker/optimal_threshold.json).

## Business Impact Estimate

Using the optimized threshold and projecting the result to the full dataset:

- Fraud caught: `370` transactions
- Fraud missed: `120` transactions
- False alarms: `65` transactions
- Estimated fraud prevented: `EUR 45,218`
- Estimated fraud missed: `EUR 14,665`
- Estimated investigation cost: `EUR 130`
- Estimated net benefit: `EUR 30,422`

These numbers show why threshold tuning matters. In fraud systems, the best technical threshold is not always the best operational threshold unless business cost is considered.

## SageMaker Deployment

After training, the model was exported in native XGBoost format and packaged for SageMaker hosting.

### Deployment Flow

1. Save the booster as `xgboost-model`.
2. Compress it into `model.tar.gz`.
3. Upload the artifact to S3.
4. Create a SageMaker `XGBoostModel`.
5. Deploy a real-time endpoint.

### Deployment Details Verified from the Notebook

- AWS Region: `ap-south-1`
- SageMaker bucket: `sagemaker-ap-south-1-883244340451`
- S3 prefix: `fraud-detection`
- Execution role: `arn:aws:iam::883244340451:role/SageMakerExecutionRole`
- Framework version: `1.7-1`
- Instance type used in deployment: `ml.m5.large`
- Endpoint name: `fraud-detector-prod`

Model artifact location:

- `s3://sagemaker-ap-south-1-883244340451/fraud-detection/model-output/model.tar.gz`

Relevant local files:

- [xgboost-model](/home/snehangshu/projects/sagemaker/xgboost-model)
- [model.tar.gz](/home/snehangshu/projects/sagemaker/model.tar.gz)

## SageMaker Autoscaling

This project also includes production-oriented endpoint scaling work in SageMaker.

Autoscaling was configured for the deployed endpoint so the real-time inference service can better respond to variable traffic without requiring manual instance management. In a real deployment, this typically means:

- registering the SageMaker endpoint variant as a scalable target
- setting minimum and maximum instance counts
- using an AWS Application Auto Scaling policy
- scaling based on metrics such as invocation load per instance

This matters because a notebook deployment alone is not enough for production readiness. Autoscaling helps the fraud detection endpoint stay available and cost-aware under changing request volume.

## Lambda Inference Layer

The Lambda code is in [lambda_function.py](/home/snehangshu/projects/sagemaker/lambda_function.py).

The Lambda function acts as the business logic layer between the public API and SageMaker.

### Lambda Responsibilities

- read environment variables for `ENDPOINT_NAME` and `THRESHOLD`
- validate that all 30 required features are present
- reject missing or non-numeric inputs
- convert the JSON payload into the exact CSV order expected by SageMaker
- invoke the SageMaker real-time endpoint
- parse the fraud probability from the endpoint response
- apply threshold-based classification
- return a structured JSON result
- support CORS for browser-based access

### Risk Mapping Used in Lambda

- probability `>= 0.70`: `HIGH`, recommended action `BLOCK`
- probability `>= 0.30` and `< 0.70`: `MEDIUM`, recommended action `REVIEW`
- probability `< 0.30`: `LOW`, recommended action `APPROVE`

### Expected Lambda Environment Variables

- `ENDPOINT_NAME=fraud-detector-prod`
- `THRESHOLD=0.73`

## API Gateway Integration

API Gateway was created in AWS Console to expose the Lambda function through HTTPS.

The notebook tests this deployed endpoint:

- `https://qpdin21s7f.execute-api.ap-south-1.amazonaws.com/prod/predict`

This makes the fraud model accessible to external applications without directly exposing SageMaker.

## API Contract

### Request Format

The API expects a JSON body containing all required numeric features:

```json
{
  "Time": 0,
  "V1": -1.359807134,
  "V2": -0.072781173,
  "V3": 2.536346738,
  "V4": 1.378155224,
  "V5": -0.33832077,
  "V6": 0.462387778,
  "V7": 0.239598554,
  "V8": 0.098697901,
  "V9": 0.36378697,
  "V10": 0.090794172,
  "V11": -0.551599533,
  "V12": -0.617800856,
  "V13": -0.991389847,
  "V14": -0.311169354,
  "V15": 1.468176972,
  "V16": -0.470400525,
  "V17": 0.207971242,
  "V18": 0.02579058,
  "V19": 0.40399296,
  "V20": 0.251412098,
  "V21": -0.018306778,
  "V22": 0.277837576,
  "V23": -0.11047391,
  "V24": 0.066928075,
  "V25": 0.128539358,
  "V26": -0.189114844,
  "V27": 0.133558377,
  "V28": -0.021053053,
  "Amount": 149.62
}
```

### Example Success Response

```json
{
  "fraud_detected": true,
  "fraud_probability": 0.9866,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK",
  "threshold_used": 0.73,
  "model_endpoint": "fraud-detector-prod"
}
```

### Error Handling

The deployed API was tested for invalid inputs and returns proper errors for:

- missing required fields
- invalid JSON payloads
- non-numeric feature values
- model invocation errors
- unexpected internal exceptions

## End-to-End Validation

The notebook validates the system at multiple levels.

### Direct SageMaker Endpoint Test

Using one known fraud sample and one legitimate sample:

- fraud sample score: `0.9866`
- legitimate sample score: `0.0136`

### Full API Test Through API Gateway

Verified results from the deployed public API:

- known fraud request returned `fraud_detected = true`
- known legitimate request returned `fraud_detected = false`
- response included probability, risk level, action, threshold, and endpoint name

## Performance and Load Testing

This repository also includes [locustfile.py](/home/snehangshu/projects/sagemaker/locustfile.py) for API load testing.

### Batch API Test Results

From 50 API requests:

- Correct predictions: `46/50`
- Accuracy on batch sample: `92%`
- Average latency: `702 ms`
- P95 latency: `725 ms`
- Minimum latency: `677 ms`
- Maximum latency: `872 ms`

These results are useful because they reflect the full deployed path:

`client -> API Gateway -> Lambda -> SageMaker -> Lambda -> API response`

## Repository Structure

```text
.
├── README.md
├── fraud_detection_project.ipynb
├── lambda_function.py
├── locustfile.py
├── model.tar.gz
├── xgboost-model
├── optimal_threshold.json
├── pyproject.toml
├── uv.lock
└── datasets
    ├── creditcard.csv
    ├── features_names.json
    ├── train.csv
    └── test.csv
```

## Local Files and Their Purpose

- [fraud_detection_project.ipynb](/home/snehangshu/projects/sagemaker/fraud_detection_project.ipynb)
  Full workflow from EDA to deployment and API validation.

- [lambda_function.py](/home/snehangshu/projects/sagemaker/lambda_function.py)
  Lambda inference handler for API Gateway integration.

- [locustfile.py](/home/snehangshu/projects/sagemaker/locustfile.py)
  Load test configuration for the deployed API.

- [optimal_threshold.json](/home/snehangshu/projects/sagemaker/optimal_threshold.json)
  Saved decision threshold used in production inference.

- [pyproject.toml](/home/snehangshu/projects/sagemaker/pyproject.toml)
  Python project dependencies.

## Dependencies

Main Python dependencies currently listed in the project:

- `kaggle`
- `locust`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`

The notebook itself also installs:

- `xgboost`
- `boto3`
- `sagemaker`

## How to Run the Project

### 1. Install dependencies

Use your preferred environment setup, then install the required packages from the project configuration.

### 2. Run the notebook

Open [fraud_detection_project.ipynb](/home/snehangshu/projects/sagemaker/fraud_detection_project.ipynb) and execute the workflow step by step.

### 3. Train and export the model

The notebook:

- preprocesses the data
- trains XGBoost
- evaluates the model
- saves `xgboost-model`
- creates `model.tar.gz`

### 4. Deploy to SageMaker

The notebook deploys the trained model to the endpoint:

- `fraud-detector-prod`

### 5. Configure Lambda

Deploy [lambda_function.py](/home/snehangshu/projects/sagemaker/lambda_function.py) as an AWS Lambda function and set:

- `ENDPOINT_NAME`
- `THRESHOLD`

### 6. Connect API Gateway

Create an API Gateway route for the Lambda function so the model is accessible through HTTPS.

### 7. Test the API

You can test:

- direct sample requests from the notebook
- error-handling scenarios
- load testing with `locust`

## Key Production Features Completed

This project is not only a machine learning experiment. It includes practical deployment tasks typically expected in an MLOps-style demo or production prototype:

- trained an imbalanced fraud detection model with XGBoost
- tuned classification threshold for better business outcomes
- uploaded model artifacts and data to S3
- deployed a real-time SageMaker endpoint
- built a Lambda inference wrapper
- created a public API using API Gateway in AWS Console
- enabled CORS support
- validated good and bad request behavior
- tested end-to-end latency
- performed API load testing
- configured SageMaker autoscaling

## Future Improvements

Possible next steps for this system:

- move preprocessing into a reusable training pipeline
- store the scaler artifact explicitly for retraining consistency
- add CloudWatch dashboards and alarms
- version models and endpoints more formally
- secure API access with authentication and rate limiting
- automate deployment with Terraform, CloudFormation, or AWS CDK
- add CI/CD for notebook-to-endpoint promotion

## Conclusion

This project demonstrates a complete machine learning deployment workflow on AWS, starting from raw fraud data and ending with a production-style prediction API. It shows model development, threshold optimization, SageMaker hosting, Lambda integration, API Gateway exposure, autoscaling considerations, and end-to-end testing in one practical fraud detection system.
