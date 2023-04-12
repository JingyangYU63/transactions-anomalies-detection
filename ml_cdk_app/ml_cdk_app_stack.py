from aws_cdk import (
    aws_apigateway as apigateway,
    aws_iam as iam,
    aws_lambda as lambda_,
    core,
)

class MlCdkAppStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the IAM role for the Lambda function
        lambda_role = iam.Role(self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Allows Lambda function to call Amazon SageMaker endpoint",
        )

        # Grant the necessary permissions to the Lambda function
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=["arn:aws:sagemaker:us-east-2:216657257678:endpoint/pytorch-inference-2023-04-11-17-34-07-154"]
        ))
        
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=["arn:aws:s3:::flagright-test-transactions/*"]
        ))
        
        # Define a Lambda function to execute your model and connect to the endpoint configuration
        lambda_fn = lambda_.Function(
            self, "MLPredictLambdaFunction",
            runtime=lambda_.Runtime.PYTHON_3_7,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset("lambda_function"),
            role=lambda_role,
        )

        # Create the API Gateway RESTful API endpoint
        api = apigateway.LambdaRestApi(
            self, "Endpoint",
            handler=lambda_fn,
            proxy=True,
            rest_api_name="MlCdkAppEndpoint",
            deploy_options={
                "stage_name": "dev"
            },
            default_cors_preflight_options={
                "allow_origins": apigateway.Cors.ALL_ORIGINS,
                "allow_methods": apigateway.Cors.ALL_METHODS
            },
            endpoint_types=[apigateway.EndpointType.REGIONAL]
        )

        # Print the API Gateway endpoint URL for testing
        core.CfnOutput(
            self, "MyApiUrl",
            value=api.url,
            description="My API Gateway endpoint URL",
        )
