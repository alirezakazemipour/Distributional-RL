# Reference: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch

from common import set_random_seeds, get_common_configs, make_atari
from common import Logger, Evaluator
from agents import get_agent_configs, get_agent
import os

if __name__ == "__main__":
    configs = get_common_configs()
    set_random_seeds(configs["seed"])

    configs = get_agent_configs(**configs)
    print("params:", configs)

    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not configs["online_wandb"]:
                os.environ["WANDB_MODE"] = "offline"

    test_env = make_atari(configs["env_name"], configs["seed"])
    configs.update({"n_actions": test_env.action_space.n})
    del test_env
    print(f"Environment: {configs['env_name']}\n"
          f"Number of actions: {configs['n_actions']}")

    env = make_atari(configs["env_name"], configs["seed"])
    agent = get_agent(**configs)
    logger = Logger(agent=agent, **configs)

    if not configs["do_test"]:
        total_steps = 0
        for episode in range(1, 1 + configs["max_episodes"]):
            logger.on()
            episode_reward = 0
            episode_loss = 0
            state = env.reset()
            for step in range(1, 1 + env.spec.max_episode_steps):
                total_steps += 4  # 4: MaxAndSkip env!
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store(state, reward, done, action, next_state)
                episode_reward += reward
                if total_steps % configs["train_interval"] == 0:
                    loss = agent.train()
                    episode_loss += loss
                if total_steps % configs["target_update_freq"] == 0:
                    agent.hard_target_update()
                if done:
                    break
                state = next_state

            agent.exp_eps = agent.exp_eps - 0.005 if agent.exp_eps > configs["min_exp_eps"] + 0.005 else configs[
                "min_exp_eps"]

            logger.off()
            logger.log(episode,
                       episode_reward,
                       episode_loss / step * configs["train_interval"],
                       total_steps,
                       step
                       )
    else:
        checkpoint = logger.load_weights()
        agent.online_model.load_state_dict(checkpoint["online_model_state_dict"])
        agent.exp_eps = 0
        evaluator = Evaluator(agent, **configs)
        evaluator.evaluate()
