import yaml
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
    # print(config)
    data_path = config['data_source']['raw_source']
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    # print(df.head())
    return df