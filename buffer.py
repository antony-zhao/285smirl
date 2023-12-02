import numpy as np


class ReplayBuffer:
    def __init__(self, obs_space, capacity=1_000_000):
        self.capacity = capacity
        self.current_size = 0
        self.obs = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        self.action = np.empty((capacity))
        self.reward = np.empty((capacity))
        self.next_obs = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        self.done = np.empty((capacity))

    def sample(self, batch_size):
        ind = np.random.randint(0, self.current_size, size=batch_size) % self.capacity
        return self.obs[ind], self.action[ind], self.reward[ind], self.next_obs[ind], self.done[ind]

    def insert(self, obs, action, reward, next_obs, done):
        ind = self.current_size % self.capacity
        self.obs[ind] = obs
        self.action[ind] = action
        self.reward[ind] = reward
        self.next_obs[ind] = next_obs
        self.done[ind] = done

        self.current_size += 1


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        raise NotImplementedError


class SMIRLReplayBuffer(ReplayBuffer):
    def __init__(self, obs_space, capacity=1_000_000, use_reward=False):
        super().__init__(obs_space, capacity)
        self.use_reward = use_reward

    def sample(self, batch_size):
        ind = np.random.randint(0, self.current_size, size=batch_size) % self.capacity
        rewards = self.reward[ind] * self.use_reward + self.log_probs(self.next_obs[ind])
        return self.obs[ind], self.action[ind], rewards, self.next_obs[ind], self.done[ind]

    def log_probs(self, obs):
        raise NotImplementedError
