"""SageMaker Serving entrypoint"""
import json
from io import StringIO
import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)

    globals()["raw_columns"] = pd.read_csv(f"{model_dir}/headers.csv").columns
    return model


def transform_fn(
    model, request_body, input_content_type, output_content_type="application/json"
):

    if input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf, header=None)
        # ! raw_columns is defined in global scope (see `model_fn` function)
        data.columns = raw_columns  # pylint: disable=undefined-variable

    else:
        raise Exception(f"{input_content_type} content type not supported")

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1).values

    return json.dumps(prediction.tolist()), output_content_type
