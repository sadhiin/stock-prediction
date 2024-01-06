import os
import yaml
import pickle
import json
import numpy as np
from src import logger

# from src.utils import read_params
# param_path = 'params.yaml'


def prediction(data):
    try:
        with open(os.path.join("models", "model.pkl"), 'rb') as m:
            model = pickle.load(m)
        pred = model.predict([data]).tolist()[0]
        return pred
    except Exception as e:
        print(e)
        logger.error(f"Something went wrong: {e}")
        return f"Something went wrong {e}"
