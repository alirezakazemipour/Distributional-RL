import os
import numpy as np
import random
import torch


def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def huber_loss(x, kappa):
    return torch.where(torch.abs(x) <= kappa,
                       0.5 * x.pow(2),
                       kappa * (torch.abs(x) - 0.5 * kappa)
                       )
