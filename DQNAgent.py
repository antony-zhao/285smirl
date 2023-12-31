import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from model import Critic, cuda_available
from buffer import ReplayBuffer


# TODO Better handling of hyperparameters (preferably some args dict that can be used for both DQN and SMIRL)

class DQNAgent:
    def __init__(self, obs_space, num_actions, lr=1e-4, soft_update=None, gamma=0.99,
                 eps_decay=0.99, buffer=ReplayBuffer, capacity=None, normalize_rewards=False, batch_size=256, update_freq=4,
                 start_after=5000, eps_min=0.1, target_update_freq=10000, eps_decay_per=1000,
                 use_gpu_if_available=True, filters=None):
        self.update_freq = update_freq
        self.obs_shape = obs_space.shape
        self.num_actions = num_actions.n
        self.Q = Critic(obs_space, num_actions, use_gpu_if_available, filters)
        self.target_Q = Critic(obs_space, num_actions, use_gpu_if_available, filters)

        self.device = cuda_available if use_gpu_if_available else "cpu"
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.eps_decay = lambda step: max(eps_decay ** (step // eps_decay_per), eps_min)
        self.eps = 1

        self.gamma = gamma
        self.tau = soft_update
        self.batch_size = batch_size
        self.step = 0
        if capacity is None:
            self.buffer = buffer(obs_space, normalize_rewards=normalize_rewards)
        else:
            self.buffer = buffer(obs_space, capacity, normalize_rewards=normalize_rewards)
        self.optim = optim.Adam(self.Q.parameters(), lr=lr)
        self.start_after = start_after
        self.target_update_freq = target_update_freq
        self.loss = nn.MSELoss()

    def choose_action(self, obs, explore=True):
        self.eps = self.eps_decay(self.step)
        if np.random.rand() > self.eps or explore is False:
            qa_values = self.Q(obs)
            qa_values = qa_values.detach().cpu().numpy()
            return np.argmax(qa_values)
        else:
            return np.random.randint(self.num_actions)

    def update_Q(self):
        obs, action, reward, next_obs, done = self.buffer.sample(self.batch_size)
        obs = torch.tensor(obs).float().to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        done = torch.tensor(done).to(self.device)
        with torch.no_grad():
            next_qa_values = self.target_Q(next_obs)

            next_action = torch.argmax(self.Q(next_obs), dim=1)

            next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            target_values = reward + self.gamma * (1 - done.long()) * next_q_values

        qa_values = self.Q(obs)
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss(q_values, target_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach().cpu().numpy()

    def soft_update(self):
        for target_param, param in zip(
                self.target_Q.parameters(), self.Q.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.insert(obs, action, reward, next_obs, done)
        self.step += 1
        if self.target_update_freq is None:
            self.soft_update()
        elif self.step % self.target_update_freq == 0:
            self.hard_update()

        if self.step % self.update_freq == 0 and self.step > self.start_after:
            loss = self.update_Q()
            return loss
