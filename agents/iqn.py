import torch
from torch import from_numpy
import numpy as np
from models import IQNModel
import random


class IQN:
    def __init__(self, **configs):
        self.configs = configs
        self.exp_eps = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.online_model = IQNModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"]
                                     ).to(self.device)

    def choose_action(self, state):
        if random.random() < self.exp_eps:
            return random.randint(0, self.configs["n_actions"] - 1)
        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).byte().to(self.device)
            taus = torch.rand((1, self.configs["K"]), device=self.device)
            with torch.no_grad():
                q_values = self.online_model.get_qvalues(state, taus).cpu()
            return torch.argmax(q_values, -1).item()

    @staticmethod
    def get_configs():
        configs = {"adam_eps": 0.01 / 32,
                   "n_embedding": 64,
                   "min_exp_eps": 0.01,
                   "target_update_freq": 8000,
                   "huber_k": 1,
                   "N": 64,  # 8 or 32 are acceptable too
                   "N_prime": 64,  # 8 or 32 are acceptable too
                   "K": 32
                   }
        return configs
