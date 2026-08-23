import tomllib
import random
import numpy as np
import torch


def load_config(config_path: str = "tinyvit/config.toml") -> dict:
    """
    Load config data from TOML file

    Args:
        config_path: path of the config file to load

    Returns:
        dictionary of config data loaded from the file
    """
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    
    return data


def set_seed(seed: int) -> None:
    """
    Set a seed for all RNG libraries used in training

    Args:
        seed: seed to set globally
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)