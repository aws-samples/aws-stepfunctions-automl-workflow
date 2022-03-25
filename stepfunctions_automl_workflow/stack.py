"""Stack code definition to create state machines, evaluation lambda and s3 bucket"""
import os
import shutil

from aws_cdk import (
    core as cdk,
    aws_s3 as s3,
    aws_s3_deployment as s3_deployment,
)
from stepfunctions_automl_workflow.stack_factory import (
    BuildMainStateMachine,
    BuildTrainStateMachine,
    BuildDeployStateMachine,
)

WORKDIR = "stepfunctions_automl_workflow/"
CONFIG_ASSET = "config.json"

REGION = os.environ["CDK_DEFAULT_REGION"]
# TODO make this retrievable at runtime (maybe boto3?)
AUTOGLUON_BASE_URI = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/autogluon-{scope}:0.3.1-cpu-py37"
)
AUTOGLUON_TRAINING_URI = AUTOGLUON_BASE_URI.format(region=REGION, scope="training")
AUTOGLUON_INFERENCE_URI = AUTOGLUON_BASE_URI.format(region=REGION, scope="inference")

AUTOGLUON_SCRIPTS_DIR = f"{WORKDIR}scripts/autogluon/"
AUTOGLUON_ARCHIVE_DIR = f"{WORKDIR}archives/"


def create_automl_workflow(scope: cdk.Construct, construct_id: str) -> None:
    """Generates autoML workflow

    This contains code archives, instantiate state machines, S3 bucket, IAM roles and
    AWS Lambda function for evaluation

    Parameters
    ----------
    scope : cdk.Construct
        scope of the application
    construct_id : str
        name of the application
    """
    stack = cdk.Stack(scope, construct_id)

    archive_path = f"{AUTOGLUON_ARCHIVE_DIR}sourcedir"
    archive_name = shutil.make_archive(archive_path, "gztar", AUTOGLUON_SCRIPTS_DIR)
    shutil.make_archive(archive_path, "zip", AUTOGLUON_SCRIPTS_DIR)

    bucket = s3.Bucket(
        stack, 
        "bucket",
        removal_policy=cdk.RemovalPolicy.DESTROY,
        auto_delete_objects=True
    )
    s3_deployment.BucketDeployment(
        scope=stack,
        id="BucketDeployment",
        sources=[s3_deployment.Source.asset(AUTOGLUON_ARCHIVE_DIR)],
        destination_bucket=bucket,
    )

    autogluon_scripts_uri = f"s3://{bucket.bucket_name}/{archive_name.split('/')[-1]}"

    train_sm_builder = BuildTrainStateMachine(
        stack,
        CONFIG_ASSET,
        autogluon_scripts_uri,
        AUTOGLUON_TRAINING_URI,
        AUTOGLUON_INFERENCE_URI,
    )
    deployment_sm_builder = BuildDeployStateMachine(stack, CONFIG_ASSET, 1)
    main_sm_builder = BuildMainStateMachine(
        stack, CONFIG_ASSET, train_sm_builder.build(), deployment_sm_builder.build()
    )
    main_sm_builder.build()
