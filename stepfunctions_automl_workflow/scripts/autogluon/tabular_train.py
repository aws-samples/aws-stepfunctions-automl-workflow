"""SageMaker Training Job entrypoint"""

import argparse
import os
import json
from shutil import copyfile
from autogluon.tabular import TabularDataset, TabularPredictor


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {path} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR")
    )
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument(
        "--training_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        default=get_env_if_present("SM_CHANNEL_TEST"),
    )
    parser.add_argument(
        "--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG")
    )

    # Hyperparameters
    parser.add_argument(
        "--fit_args", type=str, default="{'label': 'y', problem_type':'classification'}"
    )
    parser.add_argument("--init_args", type=str, default="{'presets': 'best_quality'}")
    parser.add_argument("--leaderboard", type=bool, default=False)
    parser.add_argument("--feature_importance", type=bool, default=False)

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    ##
    # See SageMaker-specific environment variables:
    #   https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    ##
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)
    os.makedirs(f"{args.model_dir}/code", exist_ok=True)

    ag_fit_args = json.loads(args.fit_args)
    ag_predictor_args = json.loads(args.init_args)

    copyfile("tabular_serve.py", f"{args.model_dir}/code/tabular_serve.py")

    # ---------------------------------------------------------------- Training

    train_file = get_input_path(args.training_dir)
    train_data = TabularDataset(train_file)

    # Saving raw columns for inference consistency
    columns = list(train_data.columns)
    columns.remove(ag_predictor_args["label"])
    with open(f"{args.model_dir}/headers.csv", "w") as header_file:
        header_file.write(",".join(columns))

    ag_predictor_args["path"] = args.model_dir

    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # --------------------------------------------------------------- Inference

    test_file = get_input_path(args.test_dir)
    test_data = TabularDataset(test_file)

    # Predictions
    y_pred_proba = predictor.predict_proba(test_data)
    y_pred_proba.to_csv(f"{args.output_data_dir}/predictions.csv")

    # Leaderboard
    lb = predictor.leaderboard(test_data, silent=False)
    lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    # Evaluate
    eval_scores = predictor.evaluate(test_data)
    with open(f"{args.output_data_dir}/scores.json", "w") as eval_file:
        json.dump(eval_scores, eval_file)

    # Feature importance
    feature_importance = predictor.feature_importance(test_data)
    feature_importance.to_csv(f"{args.output_data_dir}/feature_importance.csv")
