from abc import ABC
import cv2
import gym
import numpy as np


def make_atari(env_name: str,
               episodic_life: bool = True,
               clip_reward: bool = True,
               seed: int = 123
               ):
    env = gym.make(env_name)
    if "NoFrameskip" not in env.spec.id:  # noqa
        raise ValueError(f"env should be from `NoFrameskip` type got: {env_name}")  # noqa
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ResizedAndGrayscaleEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = StackFrameEnv(env)

    env.seed(seed)
    env.observation_space.np_random.seed(seed)
    env.action_space.np_random.seed(seed)
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()

        noops = np.random.randint(1, self.noop_max + 1)  # noqa
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)

        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        reward = 0
        done = None
        info = None
        for i in range(self.skip):
            obs, r, done, info = self.env.step(action)

            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            reward += r
            if done:
                break

        max_frame = self.obs_buffer.max(axis=0)  # noqa

        return max_frame, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done

        lives = info["ale.lives"]
        if self.lives > lives > 0:
            done = True

        self.lives = lives
        return obs, reward, done, info

    def reset(self):

        if self.real_done:
            obs = self.env.reset()
        else:
            obs, *_ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class ResizedAndGrayscaleEnv(gym.ObservationWrapper, ABC):
    def __init__(self, env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.channels = 1
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, self.channels),
                                                dtype=np.uint8
                                                )

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class ClipRewardEnv(gym.RewardWrapper, ABC):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return bool(reward > 0) - bool(reward < 0)


class StackFrameEnv(gym.Wrapper):
    def __init__(self, env, stack_size: int = 4):
        gym.Wrapper.__init__(self, env)
        self.stack_size = stack_size
        w, h, c = env.observation_space.shape
        self.frames = np.zeros((w, h, c), dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.frames = np.stack([obs for _ in range(self.stack_size)], axis=0)  # PyTorch's channel axis is 0!
        return self.frames

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        old_frames = self.frames[1:, ...]
        self.frames = np.concatenate([old_frames, np.expand_dims(obs, axis=0)], axis=0)
        return self.frames, reward, done, info
