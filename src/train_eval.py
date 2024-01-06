import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from src.utils import read_params, logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn


def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = median_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return rmse, mae, r2


def start_training(cfg_path: str):
    PARAMS = read_params(cfg_path)

    df = pd.read_csv(PARAMS['load_data']['path'])

    x = df.drop(columns=[PARAMS['base']['target_col']])
    y = df[PARAMS['base']['target_col']]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(PARAMS['load_data']['test_size']), random_state=PARAMS['base']['random_state'])
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=float(
        PARAMS['load_data']['test_size'])/2, random_state=PARAMS['base']['random_state'])

    logger.info(f'x_train shape: {x_train.shape}')
    logger.info(f'x_val shape: {x_val.shape}')
    logger.info(f'x_test shape: {x_test.shape}')
    with mlflow.start_run():
        logger.info(f"Loading model")
        _model = LinearRegression()
        _model.fit(x_train, y_train)
        logger.info(f"Model trained")

        score = _model.score(x_val, y_val)
        logger.info(f"Model score on validation set: {score}")

        score = _model.score(x_test, y_test)
        predicted = _model.predict(x_test)
        signature = infer_signature(x_train, predicted)

        rmse, mae, r2 = eval_metrics(y_test, predicted)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        logger.info(f"Model score on test set: {score}")

        logger.info(f"Model performance on test set:")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"MAE: {mae}")
        logger.info(f"R2: {r2}")

         # reporting the informations
        scores_file = PARAMS["reports"]["scores"]
        with open(scores_file, 'w') as f:
            matric = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            json.dump(matric, f, indent=4)

        remote_server_uri = "https://dagshub.com/sadhiin/mlflow-tutorial.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        logger.info(f"Saving model")
        _path = os.path.join(PARAMS['model']['path'],PARAMS['model']['filename'])

        with open(_path, 'wb') as file:
            pickle.dump(_model, file)
        logger.info(f"Model saved at {_path}")

        # for remote server

        remote_server_uri = "https://github.com/sadhiin/stock-prediction.git"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                _model, "model", registered_model_name="LinearRegression", signature=signature)
        else:
            mlflow.sklearn.log_model(_model, "model")

        signature = infer_signature(x_train, y_train)
        print("Signature: ", signature)
        mlflow.sklearn.log_model(_model, "model", signature=signature)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger.info(f"Training started with config: {parsed_args.config}")
    start_training(cfg_path=parsed_args.config)
