import gym
import numpy as np
from .atari_wrappers import make_atari
import time


class Evaluator:
    def __init__(self, agent, max_episode=1, **config):
        self.config = config
        self.env = make_atari(self.config["env_name"], episodic_life=False, clip_reward=False, seed=int(time.time()))
        self.env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.prepare_to_play()

    def evaluate(self):
        total_reward = 0
        info = {}
        print("--------Play mode--------")
        for _ in range(self.max_episode):
            state = self.env.reset()
            episode_reward = 0
            info["ale.lives"] = np.inf

            while info["ale.lives"] > 0:
                action = self.agent.choose_action(state)
                nex_state, r, done, info = self.env.step(action)
                self.env.render()
                time.sleep(0.01)
                episode_reward += r
                state = nex_state
                if done:
                    state = self.env.reset()
            total_reward += episode_reward

        print("Total episode reward:", total_reward / self.max_episode)
        self.env.close()
