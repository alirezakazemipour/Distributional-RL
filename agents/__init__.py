from .iqn import IQN
from.c51 import C51

AGENTS = dict(IQN=IQN,
              C51=C51
              )


def get_agent_configs(**configs):
    agent_configs = AGENTS[configs["agent_name"]].get_configs()
    return {**configs, **agent_configs}


def get_agent(**kwargs):
    return AGENTS[kwargs["agent_name"]](**kwargs)
