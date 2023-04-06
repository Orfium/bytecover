from dataclasses import dataclass
from logging import config as logging_config
from typing import Dict

from yaml import FullLoader, load, safe_load


@dataclass
class bcolors:
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"


def load_config(config_path: str) -> Dict:
    with open(config_path) as file:
        config = safe_load(file)

    if config["device"] == "gpu":
        config["device"] = "cuda:0"

    return config


def initialize_logging(config_path: str, debug: bool) -> None:
    """
    Setup logging according to the configuration in the given file.
    :param str config_path: The path to the file containing the logging configuration
    :return:
    """
    with open(config_path) as yaml_fh:
        config_description = load(yaml_fh, Loader=FullLoader)
        if debug:
            config_description["root"]["level"] = "DEBUG"
        logging_config.dictConfig(config_description)
