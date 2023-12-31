from DQNAgent import DQNAgent
from buffer import ReplayBuffer
from model import VAE, loss_vae
import numpy as np
import torch


class SMIRL_VAEAgent(DQNAgent):
    def __init__(self, obs_space, num_actions, lr=1e-4, soft_update=None, gamma=0.99, eps_decay=0.99,
                 buffer=ReplayBuffer, capacity=None, batch_size=256, update_freq=4, start_after=5000, eps_min=0.1,
                 target_update_freq=10000, eps_decay_per=1000, latent_dim=100):
        super().__init__(obs_space, num_actions, lr, soft_update, gamma, eps_decay, buffer, capacity, batch_size,
                         update_freq, start_after, eps_min, target_update_freq, eps_decay_per)
        self.vae = VAE(obs_space, latent_dim=latent_dim)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=1e-5)

    def update(self, obs, action, reward, next_obs, done):
        reward_vae = self.vae.log_prob(next_obs).mean()
        super().update(obs, action, reward_vae, next_obs, done)
        self.update_vae()

    def update_vae(self):
        obs, _, _, _, _ = self.buffer.sample(self.batch_size)
        vae_out, mean, var = self.vae(obs)
        # _ = 1 / (1 + np.exp(-obs))
        loss = loss_vae(obs, vae_out, mean, var, self.vae)
        self.vae_optim.zero_grad()
        loss.backward()
        self.vae_optim.step()
