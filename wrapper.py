from pettingzoo import ParallelEnv


class SoftReset(ParallelEnv):
    def __init__(self, env, max_timestep=1000):
        self.env = env
        self._max_timestep = max_timestep
        self._time = max_timestep

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def step(self, action):
        self._time -= 1
        next_obs, reward, terminations, truncations, info = self.env.step(action)
        if False not in terminations.values():
            next_obs = self.env.reset()
        terminations = {agent: False for agent in terminations.keys()}
        if self._time == 0:
            truncations = {agent: True for agent in truncations.keys()}
        return next_obs, reward, terminations, truncations, info

    def reset(self, seed=None, options=None):
        self._time = self._max_timestep
        return self.env.reset(seed, options)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
