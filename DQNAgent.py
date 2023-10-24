import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from model import Model
from buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    def __init__(self, obs_space, num_actions, lr=1e-4, soft_update=None, gamma=0.99,
                 eps_decay=0.99, buffer=ReplayBuffer, capacity=None, batch_size=256, update_freq=4,
                 start_after=5000, eps_min=0.1, target_update_freq=10000):
        self.update_freq = update_freq
        self.obs_shape = obs_space.shape
        self.num_actions = num_actions.n
        self.Q = Model(obs_space, num_actions)
        self.target_Q = Model(obs_space, num_actions)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.eps_decay = eps_decay
        self.eps = 1
        self.eps_min = eps_min
        self.gamma = gamma
        self.tau = soft_update
        self.batch_size = batch_size
        self.step = 0
        if capacity is None:
            self.buffer = buffer(self.obs_shape)
        else:
            self.buffer = buffer(self.obs_shape, capacity)
        self.optim = optim.Adam(self.Q.parameters(), lr=lr)
        self.start_after = start_after
        self.target_update_freq = target_update_freq
        self.loss = nn.MSELoss()

    def choose_action(self, obs, eps=0.1):
        if np.random.rand() > eps:
            qa_values = self.Q(obs)
            qa_values = qa_values.detach().cpu().numpy()
            return np.argmax(qa_values)
        else:
            return np.random.randint(self.num_actions)

    def update_Q(self):
        obs, action, reward, next_obs, done = self.buffer.sample(self.batch_size)
        obs = torch.tensor(obs).float()
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward).float()
        next_obs = torch.tensor(next_obs).float()
        done = torch.tensor(done)
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
