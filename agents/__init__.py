from .iqn import IQN

AGENTS = dict(IQN=IQN)


def get_agent_configs(**configs):
    agent_configs = AGENTS[configs["agent_name"]].get_configs()
    return {**configs, **agent_configs}


def get_agent(**kwargs):
    return AGENTS[kwargs["agent_name"]](**kwargs)
