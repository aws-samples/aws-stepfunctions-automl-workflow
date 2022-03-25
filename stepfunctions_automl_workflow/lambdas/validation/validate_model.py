"""AWS Lambda that loads model perfomances from S3 and compare then against threshold"""
import json
import os
import boto3
import tarfile

REQUIRED_PARAMS = ("ModelPath", "EvalThreshold", "EvalMetric")
ARCHIVE_NAME = "output.tar.gz"
METRICS_FILE_NAME = "scores.json"
LOCAL_WORKDIR = "/tmp/"
LOCAL_ARCHIVE_PATH = os.path.join(LOCAL_WORKDIR, ARCHIVE_NAME)


def parse_s3_uri(uri):

    splits = uri.replace("s3://", "").split("/")
    bucket = splits[0]
    object_key = os.path.join("/".join(splits[1:]))

    return bucket, object_key


def lambda_handler(event, context):

    # Validate input parameters
    for param in REQUIRED_PARAMS:
        if param not in event:
            raise IOError(f"Missing required parameter {param}")
        print(f"\t[{param}] - {event[param]}")

    model_path = event["ModelPath"]
    threshold = float(event["EvalThreshold"])
    metric = event["EvalMetric"]

    bucket, model_object_key = parse_s3_uri(model_path)
    archive_path = "/".join(model_object_key.split("/")[:-1] + [ARCHIVE_NAME])

    # Download archive to local disk
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(archive_path, LOCAL_ARCHIVE_PATH)
    print(f"Output at {archive_path} saved locally at {LOCAL_ARCHIVE_PATH}")

    # Extract files from archive
    with tarfile.open(LOCAL_ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(LOCAL_WORKDIR)

    with open(os.path.join(LOCAL_WORKDIR, METRICS_FILE_NAME), "r") as file:
        scores = json.load(file)

    return {"IsValid": (scores[metric] >= threshold), "Scores": scores}
