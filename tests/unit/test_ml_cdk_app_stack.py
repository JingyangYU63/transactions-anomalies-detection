import aws_cdk as core
import aws_cdk.assertions as assertions

from ml_cdk_app.ml_cdk_app_stack import MlCdkAppStack

# example tests. To run these tests, uncomment this file along with the example
# resource in ml_cdk_app/ml_cdk_app_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MlCdkAppStack(app, "ml-cdk-app")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
