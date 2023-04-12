import boto3
import csv
import json

# Define the AWS region and SageMaker endpoint name
region_name = 'us-east-2'
endpoint_name = 'pytorch-inference-2023-04-11-17-34-07-154'

# Create a SageMaker runtime client to invoke the endpoint
runtime_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    # Retrieve the input data from the Lambda event
    input_data = event['body']

    # Invoke the SageMaker endpoint with the input data
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=input_data
    )
    print(response)

    # Extract the predictions from the SageMaker endpoint response
    predictions = json.loads(response['Body'].read().decode())

    # Return the predictions as a JSON response
    return {
        'statusCode': 200,
        'body': json.dumps(predictions)
    }
