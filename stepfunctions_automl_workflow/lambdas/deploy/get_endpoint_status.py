"""AWS Lambda function to retrieve endpoint depoyment status"""
import boto3

sm_client = boto3.client("sagemaker")


def lambda_handler(event, context):

    return {
        "EndpointStatus": sm_client.describe_endpoint(
            EndpointName=event["EndpointName"]
        )["EndpointStatus"]
    }
