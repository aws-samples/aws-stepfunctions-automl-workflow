"""App entrypoint"""

from aws_cdk import core as cdk
from stepfunctions_automl_workflow.stack import create_automl_workflow


app = cdk.App()
create_automl_workflow(app, "StepFunctionsAutoMLWorkflow")

app.synth()
