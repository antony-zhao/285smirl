import numpy as np


class ReplayBuffer:
    def __init__(self, obs_space, capacity=1_000_000, normalize_rewards=False):
        self.capacity = capacity
        self.current_size = 0
        self.obs_dim = obs_space.shape
        self.obs = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        self.action = np.empty((capacity))
        self.reward = np.empty((capacity))
        self.next_obs = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        self.done = np.empty((capacity))
        self.normalize_rewards = normalize_rewards
        self.eps = 1e-5

    def sample(self, batch_size):
        ind = np.random.randint(0, self.current_size, size=batch_size) % self.capacity
        reward = self.reward[ind]
        if self.normalize_rewards:
            mean = self.reward.mean()
            std = self.reward.std() + self.eps
            reward = (reward - mean) / std
        return self.obs[ind], self.action[ind], reward, self.next_obs[ind], self.done[ind]

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


class SMIRLBuffer:
    def __init__(self, obs_space, capacity=1_000_000, reward_thresh=300, smirl_coeff=0.5):
        self.capacity = capacity
        self.obs_space = obs_space
        self.reward_thresh = reward_thresh
        self.reward_multiplier = smirl_coeff

    def insert(self, obs):
        raise NotImplementedError

    def log_probs(self, obs):
        raise NotImplementedError

    def smirl_reward(self, obs):
        lob_prob = self.log_probs(obs)
        log_prob = np.clip(lob_prob, -self.reward_thresh, self.reward_thresh)
        return log_prob * self.reward_multiplier

    def reset(self):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError


class BernoulliBuffer(SMIRLBuffer):
    def __init__(self, obs_space, capacity=1_000_000, reward_thresh=300, smirl_coeff=0.5):
        super().__init__(obs_space, capacity, reward_thresh, smirl_coeff)
        self.threshold = 1e-4
        self.obs_cum = np.zeros(obs_space.shape)
        self.current_size = 0

    def get_mean(self):
        mean = self.obs_cum / self.current_size
        return np.clip(mean, self.threshold, 1 - self.threshold)

    def get_params(self):
        return self.get_mean()

    def log_probs(self, obs):
        theta = self.get_mean()
        prob = theta * obs + (1 - theta) * (1 - obs)

        log_prob = np.sum(np.log(prob))
        return log_prob

    def insert(self, obs):
        self.obs_cum += obs
        self.current_size += 1

    def reset(self):
        self.obs_cum = np.zeros(self.obs_space.shape)
        self.current_size = 0
