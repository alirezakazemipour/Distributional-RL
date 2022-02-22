import os
import numpy as np
import random
import torch


def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
