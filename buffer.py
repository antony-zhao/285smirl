import numpy as np


class ReplayBuffer:
    def __init__(self, obs_space, capacity=1_000_000):
        self.capacity = capacity
        self.current_size = 0
        self.obs_dim = obs_space.shape
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


class BernoulliBuffer(SMIRLReplayBuffer):
    def __init__(self, obs_space, capacity=1_000_000, use_reward=False):
        super().__init__(obs_space, capacity, use_reward)
        self.threshold = 1e-4
        self.obs_cum = np.zeros(*obs_space.shape)

    def get_mean(self):
        mean = self.obs_cum / self.current_size
        return np.clip(mean, self.threshold, 1-self.threshold)

    def log_probs(self, obs):
        theta = self.get_mean()
        prob = theta * obs - (1 - theta) * obs

        log_prob = np.sum(np.log(prob), axis=0)
        return log_prob

    def insert(self, obs, action, reward, next_obs, done):
        self.obs_cum += obs
        super().insert(obs, action, reward, next_obs, done)

