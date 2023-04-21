# transactions-anomalies-detection

## 1. Files/ folders comments:

ml-cdk-app: The AWS CDK App folder containing all crucial part of a web application;

ml_cdk_app_stack.py: Located at ./ml-cdk-app/ml_cdk_app/ml_cdk_app_stack.py, it created a AWS CDK Stack class (containing construction of Lambda Function, IAM role, RESTful API endpoint of API Gateway);

lambda_function.py: Located at ./ml-cdk-app/lambda_function/lambda_function.py, it defined the Lambda Function to perform the ML inference;

transactions-ml-sagemaker.ipynb: The SageMaker Jupyter Notebook that performs EDA, feature engineering, model training on AWS instance, model deployment on a SageMaker endpoint for further configuration;

som-train.py: The Python script that defines the neural network - SOM (Self-Organizing Map) and explicit training codes;

inference.py: The Python script for SageMaker configuration, defining basic requirements like model loading, model saving, model prediction, model prediciton, model output, etc.

## 2. Relevent links

Github Repository: https://github.com/JingyangYU63/transactions-anomalies-detection

Lambda Function API Address: https://usoi5v3cozoyu36rmracdmqpjy0gxuxu.lambda-url.us-east-2.on.aws/

## 3. Model & Deployment

Before to fit in the data right into my model, I performed data preprocessing to read in the json file and fill in the NaN values. Then I computed aggregated statistics based on user's account info as feature engineering to generate useful features and transform the data into a form that the model can consume after standardization (this is crucial in distance-based custering algorithms). Then I did some EDA and visualization using basic ML clustering algorithms like KNN, DBSCAN, etc. After that, I started constructing my main model - SOM (Self-Organizing Map) in PyTorch. Since this algorithm updates weights once per input datapoint, I passed the data directly into the model instead of using data loaders. When the model is fully trained, I picked the data records that have a large distance to the cluster center above a center threshold (e.g., 95% quantile for the total distances) as the anomalies.

![image](https://user-images.githubusercontent.com/73151841/233254442-062ecd14-fc54-4040-a38b-42db809a9511.png)

Before to fit in the data right into my model, I performed data preprocessing to read in the json file and fill in the NaN values. Then I computed aggregated statistics based on user's account info as feature engineering to generate useful features and transform the data into a form that the model can consume after standardization (this is crucial in distance-based custering algorithms). Then I did some EDA and visualization using basic ML clustering algorithms like KNN, DBSCAN, etc. After that, I started constructing my main model - SOM (Self-Organizing Map) in PyTorch. Since this algorithm updates weights once per input datapoint, I passed the data directly into the model instead of using data loaders. When the model is fully trained, I picked the data records that have a large distance to the cluster center above a center threshold (e.g., 95% quantile for the total distances) as the anomalies.

Cluster centers pre-training

<img width="545" alt="image" src="https://user-images.githubusercontent.com/73151841/233545422-d4a8e12f-58df-406b-bd10-fe621b938c21.png">

Cluster centers after training

<img width="553" alt="image" src="https://user-images.githubusercontent.com/73151841/233545473-f1f8728d-3d87-484f-a025-288c989b0e1b.png">

I began to deploy the model after training. First, I deployed the model to a SageMaker endpoint. Then I created an AWS CDK App with a Lambda Function, served as the backend for the CDK app to perform ML inference. To retrieve my model, I configured my Lambda Function with an IAM role to grant permission to access the SageMaker endpoint where I deployed the model and an API Gateway RESTful API endpoint to provide a secure and scalable way of exposing the Lambda function to the internet. The Lambda function is triggered by API Gateway when a user makes a request to the CDK app, and it is responsible for handling the request and generating a response.
<img width="1359" alt="image" src="https://user-images.githubusercontent.com/73151841/231852488-baa9518b-7f13-4350-a663-17d88cc3a659.png">

Some test imputs on my deployed API:

<img width="662" alt="image" src="https://user-images.githubusercontent.com/73151841/233172779-0d8c5337-2ccd-49f8-a7a3-f089f24e9aa8.png">

<img width="1281" alt="image" src="https://user-images.githubusercontent.com/73151841/231853808-75cdb27e-dc94-4cf3-9377-be6a97b82cb8.png">
