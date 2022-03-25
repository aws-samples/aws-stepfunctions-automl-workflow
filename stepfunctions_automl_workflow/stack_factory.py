"""Factory that builds state machines

BuildMainStateMachine: build the main state machine
BuildTrainStateMachine: build the state machine that trains autogluon models
BuildDeployStateMachine: build the state machine that deploys autogluon models
"""
from abc import ABC, abstractmethod
import os
import json

from aws_cdk import (
    aws_iam as iam,
    aws_stepfunctions as sfn,
    core as cdk,
    aws_stepfunctions_tasks as tasks,
    aws_ec2 as ec2,
    aws_lambda as lambda_,
)

SLEEP_SECONDS = 60
WORKDIR = "stepfunctions_automl_workflow/"
LAMBDAS_DIR = f"{WORKDIR}lambdas/"
VALIDATE_MODEL_LAMBDA_ASSET = os.path.join(LAMBDAS_DIR, "validation")
STATUS_ENDPOINT_LAMBDA_ASSET = os.path.join(LAMBDAS_DIR, "deploy")

# pylint: disable=too-few-public-methods
class BuildStateMachine(ABC):
    """Abstract class that defines an interfaces to create state machines

    Attributes
    ----------
        _stack: cdf.Stack
            CDK stack
        _config: str
            path to the stack configuration file
    """

    def __init__(self, stack: cdk.Stack, cfg_file: str) -> None:
        super().__init__()
        self._stack = stack
        self._config = cfg_file

    def _create_role_from_config(self, stage: str) -> iam.Role:
        """Create role from configuration file by loading the definition from configuration file"""
        with open(self._config, "r") as config_file:
            role_config = json.load(config_file)["Roles"][stage]

        # Add principals that can assume the role
        composite_principal = None
        for principal in role_config["Principals"]:
            if composite_principal:
                composite_principal.add_principals(iam.ServicePrincipal(principal))
            else:
                composite_principal = iam.CompositePrincipal(
                    iam.ServicePrincipal(principal)
                )

        # Create the role
        role = iam.Role(self._stack, f"Role{stage}", assumed_by=composite_principal)

        if "Custom" in role_config["Policies"]:
            for custom_policy in role_config["Policies"]["Custom"].values():

                effect = (
                    iam.Effect("ALLOW")
                    if "Effect" not in custom_policy
                    else iam.Effect(custom_policy["Effect"].upper())
                )

                role.add_to_policy(
                    iam.PolicyStatement(
                        resources=custom_policy["Resources"],
                        actions=custom_policy["Actions"],
                        effect=effect,
                    )
                )

        if "AWS" in role_config["Policies"]:
            for managed_policy in role_config["Policies"]["AWS"]:
                role.add_managed_policy(
                    iam.ManagedPolicy.from_aws_managed_policy_name(managed_policy)
                )

        return role

    @abstractmethod
    def build(self) -> sfn.StateMachine:
        """Build the state machine

        The state machine has to be defined in concrete classes that derive from `BuildStateMachine`

        Returns
        -------
            sfn.StateMachine : state machine
        """
        ...


class BuildTrainStateMachine(BuildStateMachine):
    """Build the state machine that trains AG models

    Attributes
    ----------
    _stack: cdf.Stack
        see parent `BuildStateMachine`
    _config: str
        see parent `BuildStateMachine`
    _autogluon_script_uri : str
        S3 URI of the autogluon script
    _autogluon_training_image : str
        Autogluon ECR training image
    _autogluon_inference_image : str
        Autogluon ECR deployment image
    """

    def __init__(
        self,
        stack: cdk.Stack,
        cfg_file: str,
        scripts_uri: str,
        ag_train_uri: str,
        ag_inference_uri: str,
    ) -> None:
        super().__init__(stack, cfg_file)
        self._autogluon_script_uri = scripts_uri
        self._autogluon_training_image = ag_train_uri
        self._autogluon_inference_image = ag_inference_uri

    def build(self) -> sfn.StateMachine:
        """See parent `BuildStateMachine.build`"""
        train_state_machine_role = self._create_role_from_config("TrainStateMachine")
        training_role = self._create_role_from_config("Train")

        training_step = self._create_training_step(
            role=training_role, scripts_uri=self._autogluon_script_uri
        )
        create_model_step = self._create_model_step(role=training_role)
        training_step.next(create_model_step)

        return sfn.StateMachine(
            scope=self._stack,
            id="TrainStateMachine",
            definition=training_step,
            role=train_state_machine_role,
            timeout=cdk.Duration.minutes(60),
        )

    def _create_training_step(self, role, scripts_uri):
        """Implementation of the training step"""
        training_image = tasks.DockerImage.from_registry(
            image_uri=self._autogluon_training_image
        )

        s3_input_train_location = tasks.S3Location.from_json_expression(
            expression="$.Parameters.Train.TrainDataPath",
        )

        s3_input_test_location = tasks.S3Location.from_json_expression(
            expression="$.Parameters.Train.TestDataPath",
        )

        s3_output_location = tasks.S3Location.from_json_expression(
            expression="$.Parameters.Train.TrainingOutput",
        )

        input_train_data_source = tasks.S3DataSource(
            s3_location=s3_input_train_location,
            s3_data_distribution_type=tasks.S3DataDistributionType.FULLY_REPLICATED,
        )

        input_train_channel = tasks.Channel(
            channel_name="train",
            data_source=tasks.DataSource(s3_data_source=input_train_data_source),
            content_type="text/csv",
        )

        input_test_data_source = tasks.S3DataSource(
            s3_location=s3_input_test_location,
            s3_data_distribution_type=tasks.S3DataDistributionType.FULLY_REPLICATED,
        )

        input_test_channel = tasks.Channel(
            channel_name="test",
            data_source=tasks.DataSource(s3_data_source=input_test_data_source),
            content_type="text/csv",
        )

        output_data_config = tasks.OutputDataConfig(
            s3_output_location=s3_output_location
        )

        resource_config = tasks.ResourceConfig(
            instance_count=sfn.JsonPath.number_at("$.Parameters.Train.InstanceCount"),
            instance_type=ec2.InstanceType(
                sfn.JsonPath.string_at("$.Parameters.Train.InstanceType")
            ),
            volume_size=cdk.Size.gibibytes(30),
        )

        hyperparameters = {
            "fit_args": sfn.JsonPath.string_at("$.Parameters.Train.FitArgs"),
            "init_args": sfn.JsonPath.string_at("$.Parameters.Train.InitArgs"),
            "sagemaker_submit_directory": scripts_uri,
            "sagemaker_program": "tabular_train.py",
            "sagemaker_region": "eu-west-1",
            "sagemaker_container_log_level": "20",
            "sagemaker_job_name": sfn.JsonPath.string_at("$$.Execution.Name"),
        }

        algorithm_specification = tasks.AlgorithmSpecification(
            training_image=training_image
        )

        training_step = tasks.SageMakerCreateTrainingJob(
            self._stack,
            "TrainingStep",
            training_job_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            algorithm_specification=algorithm_specification,
            input_data_config=[input_train_channel, input_test_channel],
            output_data_config=output_data_config,
            resource_config=resource_config,
            hyperparameters=hyperparameters,
            role=role,
            result_path="$.TrainingOutput",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )

        return training_step

    def _create_model_step(self, role=None):
        """Implementation of the step that creates the model"""
        inference_image = tasks.DockerImage.from_registry(
            image_uri=self._autogluon_inference_image
        )

        container_definition = tasks.ContainerDefinition(
            image=inference_image,
            model_s3_location=tasks.S3Location.from_json_expression(
                "$.TrainingOutput.ModelArtifacts.S3ModelArtifacts"
            ),
            environment_variables=sfn.TaskInput.from_object(
                {
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_PROGRAM": "tabular_serve.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": sfn.JsonPath.string_at(
                        "$.TrainingOutput.ModelArtifacts.S3ModelArtifacts"
                    ),
                }
            ),
        )

        return tasks.SageMakerCreateModel(
            scope=self._stack,
            id="CreateModel",
            model_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            primary_container=container_definition,
            result_path="$.CreateModelOutput",
            role=role,
        )


class BuildDeployStateMachine(BuildStateMachine):
    """Build the state machine that deploys autogluon models

    Attributes
    ----------
    _stack: cdf.Stack
        see parent `BuildStateMachine`
    _config: str
        see parent `BuildStateMachine`
    _nb_instances : int
        Number of istances to use at inference time (default to 1)
    """

    def __init__(self, stack: cdk.Stack, cfg_file: str, nb_instances: int = 1) -> None:
        super().__init__(stack, cfg_file)
        self._nb_instances = nb_instances

    def build(self) -> sfn.StateMachine:
        """See parent `BuildStateMachine.build`"""
        deploy_state_machine_role = self._create_role_from_config("DeployStateMachine")

        create_endpoint_config_step = self._create_endpoint_config_step()
        create_endpoint_step = self._create_endpoint_step()
        create_endpoint_config_step.next(create_endpoint_step)

        wait_for_endpoint_step = self._create_wait_step(step_id="WaitForEndpoint")
        create_endpoint_step.next(wait_for_endpoint_step)

        endpoint_status_lambda_step = self._create_endpoint_status_step()
        wait_for_endpoint_step.next(endpoint_status_lambda_step)

        success_step = sfn.Succeed(scope=self._stack, id="Deployment Succeded")
        failure_step = sfn.Fail(scope=self._stack, id="Deployment Failed")

        choice_endpoint_status_step = self._create_choice_status_step(
            step_id="ChoiceEndpointStatus",
            variable="$.EndpointStatusLambdaOutput.Payload.EndpointStatus",
            wait_step=wait_for_endpoint_step,
            next_step=success_step,
            fail_step=failure_step,
            next_value="InService",
            wait_value="Creating",
        )
        endpoint_status_lambda_step.next(choice_endpoint_status_step)

        batch_inference_step = self._create_batch_inference_step()
        batch_inference_step.next(success_step)

        choice_deployment_mode_step = self._create_choice_deployment_step(
            skip_deployment_step=sfn.Fail(scope=self._stack, id="Unexpected Mode"),
            endpoint_step=create_endpoint_config_step,
            batch_inference_step=batch_inference_step,
        )

        return sfn.StateMachine(
            scope=self._stack,
            id="DeployStateMachine",
            definition=choice_deployment_mode_step,
            role=deploy_state_machine_role,
            timeout=cdk.Duration.minutes(60),
        )

    def _create_endpoint_config_step(self):
        """Implementation of the step that creates the endpoint configuration"""
        production_variant = tasks.ProductionVariant(
            instance_type=ec2.InstanceType(
                sfn.JsonPath.string_at("$.Parameters.Deploy.InstanceType")
            ),
            model_name=sfn.JsonPath.string_at("$.Parameters.PretrainedModel.Name"),
            variant_name="AllTraffic",
            # TODO instance count can't be parametric. CDK BUG HERE
            initial_instance_count=self._nb_instances,
        )

        return tasks.SageMakerCreateEndpointConfig(
            scope=self._stack,
            id="EndpointConfig",
            endpoint_config_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            production_variants=[production_variant],
            result_path="$.CreateEndpointConfigOutput",
        )

    def _create_endpoint_step(self):
        """Implementation of the step that creates the endpoint"""
        return tasks.SageMakerCreateEndpoint(
            self._stack,
            "CreateEndpoint",
            endpoint_config_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            endpoint_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            result_path="$.CreateEndpointOutput",
        )

    def _create_wait_step(self, step_id, duration=SLEEP_SECONDS):
        """Implementation of the step that creates the endpoint"""
        return sfn.Wait(
            self._stack,
            step_id,
            time=sfn.WaitTime.duration(cdk.Duration.seconds(duration)),
        )

    def _create_endpoint_status_step(self):
        """Implementation of the step that checks on the status of the endpoint"""
        endpoint_status_lambda = self._create_endpoint_status_lambda()
        endpoint_lambda_payload = {
            "EndpointName": sfn.JsonPath.string_at("$$.Execution.Name")
        }

        endpoint_lambda_step = tasks.LambdaInvoke(
            self._stack,
            "EndpointStatusLambdaStep",
            lambda_function=endpoint_status_lambda,
            payload=sfn.TaskInput.from_object(endpoint_lambda_payload),
            result_path="$.EndpointStatusLambdaOutput",
        )

        return endpoint_lambda_step

    def _create_endpoint_status_lambda(self):
        """Implementation of the step that checks on the status of the endpoint"""
        lambda_execution_role = self._create_role_from_config(
            stage="EndpointStatusLambda"
        )

        return lambda_.Function(
            self._stack,
            "GetEndpointStatus",
            code=lambda_.Code.from_asset(STATUS_ENDPOINT_LAMBDA_ASSET),
            handler="get_endpoint_status.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_6,
            role=lambda_execution_role,
            timeout=cdk.Duration.seconds(60),
        )

    def _create_choice_status_step(
        self,
        step_id,
        variable,
        wait_step,
        next_step,
        fail_step,
        wait_value="InProgress",
        next_value="Completed",
    ):
        """Implement step that decides on the action to take according to the status"""
        choice_status_step = sfn.Choice(self._stack, step_id)
        choice_status_step.when(
            condition=sfn.Condition.string_equals(variable, next_value), next=next_step,
        )
        choice_status_step.when(
            condition=sfn.Condition.string_equals(variable, wait_value), next=wait_step,
        )
        choice_status_step.otherwise(fail_step)
        return choice_status_step

    def _create_batch_inference_step(self):
        """Implementation of the step that creates and run a SM batch transform job"""
        input_eval_data_source = tasks.TransformDataSource(
            s3_data_source=tasks.TransformS3DataSource(
                s3_uri=sfn.JsonPath.string_at("$.Parameters.Deploy.BatchInputDataPath"),
                s3_data_type=tasks.S3DataType.S3_PREFIX,
            )
        )

        transform_input = tasks.TransformInput(
            transform_data_source=input_eval_data_source,
            content_type="text/csv",
            split_type=tasks.SplitType.LINE,
        )

        transform_resources = tasks.TransformResources(
            instance_count=sfn.JsonPath.number_at("$.Parameters.Deploy.InstanceCount"),
            instance_type=ec2.InstanceType(
                sfn.JsonPath.string_at("$.Parameters.Deploy.InstanceType")
            ),
        )

        return tasks.SageMakerCreateTransformJob(
            scope=self._stack,
            id="BatchInference",
            transform_job_name=sfn.JsonPath.string_at("$$.Execution.Name"),
            model_name=sfn.JsonPath.string_at("$.Parameters.PretrainedModel.Name"),
            transform_input=transform_input,
            transform_output=tasks.TransformOutput(
                s3_output_path=sfn.JsonPath.string_at(
                    "$.Parameters.Deploy.BatchOutputDataPath"
                ),
                assemble_with=tasks.AssembleWith.LINE,
            ),
            transform_resources=transform_resources,
            batch_strategy=tasks.BatchStrategy.MULTI_RECORD,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )

    def _create_choice_deployment_step(
        self, skip_deployment_step, endpoint_step, batch_inference_step
    ):
        """Implementation of the step that decides on the type of deployment"""
        choice_deployment_step = sfn.Choice(self._stack, "ChoiceDeploymentMode")

        choice_deployment_step.when(
            condition=sfn.Condition.string_equals(
                variable="$.Parameters.Deploy.Mode", value="endpoint"
            ),
            next=endpoint_step,
        )
        choice_deployment_step.when(
            condition=sfn.Condition.string_equals(
                variable="$.Parameters.Deploy.Mode", value="batch"
            ),
            next=batch_inference_step,
        )
        choice_deployment_step.otherwise(skip_deployment_step)

        return choice_deployment_step


class BuildMainStateMachine(BuildStateMachine):
    """Build the main state machine that combines training and deployment

    Attributes
    ----------
    _stack: cdf.Stack
        see parent `BuildStateMachine`
    _config: str
        see parent `BuildStateMachine`
    _train_sm : sfn.StateMachine
        state machine for training steps
    _deploy_sm : sfn.StateMachine
        state machine for deployment steps
    """

    def __init__(
        self,
        stack: cdk.Stack,
        cfg_file,
        train_sm: sfn.StateMachine,
        deploy_sm: sfn.StateMachine,
    ) -> None:
        super().__init__(stack, cfg_file)
        self._train_sm = train_sm
        self._deploy_sm = deploy_sm

    def build(self) -> sfn.StateMachine:
        """See parent `BuildStateMachine.build`"""

        train_steps = tasks.StepFunctionsStartExecution(
            scope=self._stack,
            id="TrainSteps",
            state_machine=self._train_sm,
            result_selector={"Output.$": "$.Output"},
            result_path="$.TrainStepsOutput",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )
        pass_model_name_step = self._create_pass_model_name_step()
        train_steps.next(pass_model_name_step)

        model_validation_step = self._create_model_validation_step()

        deploy_steps = tasks.StepFunctionsStartExecution(
            scope=self._stack,
            id="DeploySteps",
            state_machine=self._deploy_sm,
            result_selector={"Output.$": "$.Output"},
            result_path="$.DeployStepsOutput",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )
        skip_deployment_step = sfn.Succeed(scope=self._stack, id="Skipped Deployment")

        choice_deploy_flow_step = self._create_choice_flow_step(
            step_id="IsDeploy",
            variable="$.Parameters.Flow.Deploy",
            next_step=deploy_steps,
            skip_step=skip_deployment_step,
        )

        choice_model_validation_step = self._create_choice_flow_step(
            step_id="EvaluationResults",
            variable="$.ModelValidationLambdaOutput.Payload.IsValid",
            next_step=choice_deploy_flow_step,
            skip_step=skip_deployment_step,
        )

        model_validation_step.next(choice_model_validation_step)

        choice_model_validation_flow_step = self._create_choice_flow_step(
            step_id="IsEvaluation",
            variable="$.Parameters.Flow.Evaluate",
            next_step=model_validation_step,
            skip_step=choice_deploy_flow_step,
        )

        pass_model_name_step.next(choice_model_validation_flow_step)

        main_state_machine_role = self._create_role_from_config("MainStateMachine")

        choice_training_flow_step = self._create_choice_flow_step(
            step_id="IsTraining",
            variable="$.Parameters.Flow.Train",
            next_step=train_steps,
            skip_step=choice_deploy_flow_step,
        )

        # Istantiate Main State Machine
        return sfn.StateMachine(
            scope=self._stack,
            id="MainStateMachine",
            definition=choice_training_flow_step,
            role=main_state_machine_role,
            timeout=cdk.Duration.minutes(60),
        )

    def _create_pass_model_name_step(self):
        """Implement step that passes the model name to other steps"""
        return sfn.Pass(
            scope=self._stack,
            id="PassModelName",
            parameters={
                "Name": sfn.JsonPath.string_at(
                    "$.TrainStepsOutput.Output.TrainingOutput.TrainingJobName"
                )
            },
            result_path="$.Parameters.PretrainedModel",
        )

    def _create_choice_flow_step(self, step_id, variable, next_step, skip_step):
        """Implement choice step"""
        choice_flow_step = sfn.Choice(self._stack, step_id)
        choice_flow_step.when(
            condition=sfn.Condition.boolean_equals(variable, True), next=next_step
        )
        choice_flow_step.when(
            condition=sfn.Condition.boolean_equals(variable, False), next=skip_step
        )
        choice_flow_step.otherwise(next_step)

        return choice_flow_step

    def _create_model_validation_step(self):
        """Implement the step that validates the model"""
        model_validation_lambda = self._create_model_validation_lambda()
        payload = {
            "ModelPath": sfn.JsonPath.string_at(
                "$.TrainStepsOutput.Output.TrainingOutput.ModelArtifacts.S3ModelArtifacts"
            ),
            "EvalThreshold": sfn.JsonPath.number_at(
                "$.Parameters.Evaluation.Threshold"
            ),
            "EvalMetric": sfn.JsonPath.string_at("$.Parameters.Evaluation.Metric"),
        }

        model_validation_step = tasks.LambdaInvoke(
            self._stack,
            "ModelValidation",
            lambda_function=model_validation_lambda,
            payload=sfn.TaskInput.from_object(payload),
            result_path="$.ModelValidationLambdaOutput",
        )

        return model_validation_step

    def _create_model_validation_lambda(self):
        """Implement the step that uses a lambda function to validate model performance"""

        lambda_execution_role = self._create_role_from_config(
            stage="ModelValidationLambda"
        )

        return lambda_.Function(
            self._stack,
            "ValidateModelPerformances",
            code=lambda_.Code.from_asset(VALIDATE_MODEL_LAMBDA_ASSET),
            handler="validate_model.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_6,
            role=lambda_execution_role,
            timeout=cdk.Duration.seconds(60),
        )
