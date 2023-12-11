import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

from buffer import SMIRLBuffer
from gymnasium.spaces import Box


def encoded_space(obs_space, buffer):
    shape = obs_space.shape
    if len(shape) == 1:
        shape = (shape[0] + buffer.get_params().shape[0] + 1,)
    else:
        shape = (shape[0] + buffer.get_params().shape[0] + 1, shape[1], shape[2])
    low = np.min(obs_space.low)
    high = np.max(obs_space.high)

    return Box(low, high, shape=shape)


class SMIRLWrapper(ParallelEnv):
    def __init__(self, env, buffer=SMIRLBuffer, use_reward=False, max_timestep=500, smirl_coeff=1):
        self.env = env
        old_observation_spaces = self.observation_spaces.copy()
        self.augmented_spaces = {}
        for agent in self.possible_agents:
            self.augmented_spaces[agent] = encoded_space(old_observation_spaces[agent],
                                                         buffer(old_observation_spaces[agent]))
        self.buffers = {agent: buffer(old_observation_spaces[agent], smirl_coeff=smirl_coeff)
                        for agent in self.possible_agents}
        self._max_timestep = max_timestep
        self._time = 0
        if type(use_reward) is list:
            self.use_reward = {agent: use_reward[i] for i, agent in enumerate(self.possible_agents)}
        else:
            self.use_reward = use_reward

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def step(self, action):
        self._time += 1
        next_obs, reward, terminations, truncations, info = self.env.step(action)
        if self._max_timestep is not None:
            if False not in terminations.values():
                next_obs, _ = self.env.reset()
                terminations = {agent: False for agent in terminations.keys()}
        for agent in self.possible_agents:
            info[agent] = {"entropy": -self.buffers[agent].smirl_reward(next_obs[agent])}
            if type(self.use_reward) is dict:
                use_reward = self.use_reward[agent]
            else:
                use_reward = self.use_reward
            if use_reward != "only":
                if use_reward:
                    reward[agent] += self.buffers[agent].smirl_reward(next_obs[agent])
                else:
                    reward[agent] = self.buffers[agent].smirl_reward(next_obs[agent])
                self.buffers[agent].insert(next_obs[agent])
        if self._time == self._max_timestep:
            truncations = {agent: True for agent in truncations.keys()}
        return self.encode_obs(next_obs), reward, terminations, truncations, info

    def reset(self, seed=None, options=None):
        self._time = 0
        obs, info = self.env.reset(seed, options)
        for agent in self.possible_agents:
            self.buffers[agent].reset()
            if type(obs) is tuple:
                self.buffers[agent].insert(obs[agent][0])
            else:
                self.buffers[agent].insert(obs[agent])
        return self.encode_obs(obs), info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def encode_obs(self, obs):
        params = {agent: self.buffers[agent].get_params() for agent in obs.keys()}
        obs_space = self.buffers[self.possible_agents[0]].obs_space.shape
        if len(obs_space) == 1:
            time_obs = self._time
        else:
            time_obs = np.ones(shape=(1, obs_space[1], obs_space[2])) * self._time

        obs = {agent: np.concatenate([obs[agent], params[agent], time_obs]).astype(np.float32) for agent in obs.keys()}
        return obs

    def observation_space(self, agent):
        return self.augmented_spaces[agent]