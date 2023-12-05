import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces

cuda_available = "cuda" if torch.cuda.is_available() else "cpu"


def compute_output_dim(conv, input_shape, device):
    test_tensor = torch.zeros(size=input_shape).to(device)
    test_output = conv(test_tensor).cpu()
    return test_output.flatten().shape[0], test_output.shape


def get_preprocessor(input_shape, device, filters=None):
    if len(input_shape) == 3:
        # is image
        if filters is None:
            preprocessor = nn.Sequential(
                nn.Conv2d(input_shape[0], 16, (8, 8), stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, (4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), stride=2),
                nn.ReLU()
            ).to(device)
        else:
            filter_list = [nn.Conv2d(input_shape[0], out_channels=filters[0][0],
                                     kernel_size=(filters[0][1], filters[0][1]), stride=filters[0][2]), nn.ReLU()]
            for i in range(1, len(filters)):
                filter_list += [nn.Conv2d(filters[i - 1][0], out_channels=filters[i][0],
                                          kernel_size=(filters[i][1], filters[i][1]), stride=filters[i][2]), nn.ReLU()]

            preprocessor = nn.Sequential(*filter_list).to(device)
        out_features, output_shape = compute_output_dim(preprocessor, input_shape, device)
        preprocessor.append(nn.Flatten(-3, -1))
    else:
        # is some flattened state
        preprocessor = nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ).to(device)
        out_features = 64
        output_shape = 64
    return preprocessor, out_features, output_shape


def get_decoder(input_shape, device):
    if len(input_shape) == 3:
        # is image
        decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_shape[0], (8, 8), stride=2),
            nn.Sigmoid()
        ).to(device)
    else:
        # is some flattened state
        # Will need to change if passing args or doing some other custom models
        decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_shape[0]),
            nn.Sigmoid()
        ).to(device)
    return decoder


class Critic(nn.Module):
    def __init__(self, obs_space, num_actions, use_gpu_if_available=True, filters=None):
        super(Critic, self).__init__()
        self.use_gpu_if_available = use_gpu_if_available
        self.device = cuda_available if self.use_gpu_if_available else "cpu"

        self.preprocessor, out_features, _ = get_preprocessor(obs_space.shape, self.device, filters)

        self.action = nn.Linear(out_features, num_actions.n).to(self.device)

    def forward(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs).float().to(self.device)

        x = self.preprocessor(obs)
        return self.action(x)


class Encoder(nn.Module):
    def __init__(self, obs_space, latent_dim=100, use_gpu_if_available=True):
        super(Encoder, self).__init__()
        self.use_gpu_if_available = use_gpu_if_available
        self.device = cuda_available if self.use_gpu_if_available else "cpu"

        self.preprocessor, self.hidden_dim, _ = get_preprocessor(obs_space.shape, self.device)
        self.mean = nn.Linear(self.hidden_dim, latent_dim).to(self.device)
        self.log_var = nn.Linear(self.hidden_dim, latent_dim).to(self.device)
        self.log_var.weight.data.fill_(0)

    def forward(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs).float().to(self.device)

        if len(obs.shape) > 3:
            obs /= 256
        x = self.preprocessor(obs)
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, obs_space, latent_dim=100, use_gpu_if_available=True):
        super(Decoder, self).__init__()
        self.use_gpu_if_available = use_gpu_if_available
        self.device = cuda_available if self.use_gpu_if_available else "cpu"

        _, hidden_dim, self.output_dim = get_preprocessor(obs_space.shape, "cpu")
        self.preprocessor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        self.decoder = get_decoder(obs_space.shape, self.device)

    def forward(self, features):
        x = self.preprocessor(features)
        if type(self.output_dim) is int:
            x = x.view(-1, self.output_dim)
        else:
            x = x.view(-1, *self.output_dim)
        return self.decoder(x)


class VAE(nn.Module):
    def __init__(self, obs_space, latent_dim=100, use_gpu_if_available=True):
        super(VAE, self).__init__()
        self.encoder = Encoder(obs_space, latent_dim, use_gpu_if_available)
        self.decoder = Decoder(obs_space, latent_dim, use_gpu_if_available)
        self.dist = torch.distributions.Normal
        self.standard_normal = torch.distributions.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))

    def forward(self, obs):
        mean, log_var = self.encoder(obs)
        var = torch.exp(log_var)
        dist = self.dist(mean, var)
        x = dist.rsample()
        x_hat = self.decoder(x)
        return x_hat, mean, var

    def log_prob(self, obs):
        mean, log_var = self.encoder(obs)
        return self.standard_normal.log_prob(self.dist(mean, torch.exp(log_var)).sample()).detach().cpu().numpy()


def loss_vae(x, x_hat, mean, var, vae):
    # TODO: Check if this is better https://www.tensorflow.org/tutorials/generative/cvae
    dist = vae.dist(mean, var)
    z = dist.rsample()
    d_kl = 0.5 * torch.sum(1 + torch.log(var) - mean.pow(2) - var.pow(2))
    x = torch.sigmoid(torch.tensor(x).float())
    log_likehood = nn.functional.binary_cross_entropy(x, x_hat, reduction='sum')
    mse_loss = nn.functional.mse_loss(x, x_hat, reduction='sum')
    pz = vae.standard_normal.log_prob(z).mean()
    pz_x = dist.log_prob(z).mean()
    return mse_loss - d_kl  # log_likehood - pz - pz_x
