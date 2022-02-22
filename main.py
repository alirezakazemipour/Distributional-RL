from common import set_random_seeds, get_common_configs
from common import make_atari
from agents import get_agent_configs, get_agent

if __name__ == "__main__":
    configs = get_common_configs()
    set_random_seeds(configs["seed"])

    configs = get_agent_configs(**configs)
    print("params:", configs)

    test_env = make_atari(configs["env_name"], configs["seed"])
    configs.update({"n_actions": test_env.action_space.n})
    del test_env
    print(f"Environment: {configs['env_name']}\n"
          f"Number of actions: {configs['n_actions']}")

    env = make_atari(configs["env_name"], configs["seed"])
    agent = get_agent(**configs)
    # TODO: logger = Logger()

    if not configs["do_test"]:
        for episode in range(configs["max_episodes"]):
            episode_reward = 0
            episode_len = 0
            state = env.reset()
            for step in range(env.spec.max_episode_steps):
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state

            agent.exp_eps = agent.exp_eps - 0.01 if agent.exp_eps > configs["min_exp_eps"] + 0.01 else configs[
                "min_exp_eps"]
