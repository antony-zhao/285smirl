import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces


class Model(nn.Module):
    def __init__(self, obs_space, num_actions):
        super(Model, self).__init__()
        if len(obs_space.shape) == 3:
            # is image
            self.is_image = True
            self.preprocessor = nn.Sequential(
                nn.Conv2d(obs_space.shape[0], 16, (8, 8), stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, (4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), stride=2),
                nn.ReLU(),
                nn.Flatten()
            )
            test_tensor = torch.zeros(size=obs_space.shape)
            out_features = self.preprocessor(test_tensor)
        else:
            # is some flattened state
            self.is_image = False
            self.preprocessor = nn.Sequential(
                nn.Linear(obs_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            out_features = 64

        self.action = nn.Linear(out_features, num_actions.n)

    def forward(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs).float()
        x = self.preprocessor(obs)
        return self.action(x)
