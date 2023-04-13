# transactions-anomalies-detection

## 1. Files/ folders comments:

ml-cdk-app: The AWS CDK App folder containing all crucial part of a web application;

ml_cdk_app_stack.py: Located at ./ml-cdk-app/ml_cdk_app/ml_cdk_app_stack.py, it created a AWS CDK Stack class
(containing construction of Lambda Function, IAM role, RESTful API endpoint of API Gateway);

lambda_function.py: Located at ./ml-cdk-app/lambda_function/lambda_function.py, it defined the Lambda Function to perform the ML inference;

transactions-ml-sagemaker.ipynb: The SageMaker Jupyter Notebook that performs EDA, feature engineering, model training on AWS instance,
model deployment on a SageMaker endpoint for further configuration;

som-train.py: The Python script that defines the neural network - SOM (Self-Organizing Map) and explicit training codes;

inference.py: The Python script for SageMaker configuration, defining basic requirements like model loading, model saving, model prediction,
model prediciton, model output, etc.

## 2. Relevent links

Github Repository: https://github.com/JingyangYU63/transactions-anomalies-detection

Lambda Function API Address: https://usoi5v3cozoyu36rmracdmqpjy0gxuxu.lambda-url.us-east-2.on.aws/

## 3. Model intro

