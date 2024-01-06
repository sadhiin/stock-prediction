import yaml
import pathlib
import pandas as pd
from src import logger

def read_params(config_path):
    try:
        logger.info(f"Reading config from {config_path}")
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        logger.error(f"Error reading config file: {e}")

def get_data(config_path):
    config = read_params(config_path)
    logger.info(f"Reading data from source {config['load_data']['path']}")
    data_path = pathlib.Path(config['load_data']['path'])
    df = pd.read_csv(data_path, encoding='utf-8')
    return df