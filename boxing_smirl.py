import torch
def save_model(agent, filename):
    checkpoint = {
        'Q_state_dict': agent.Q.state_dict(),
        'target_Q_state_dict': agent.target_Q.state_dict(),
        'optimizer': agent.optim.state_dict(),
        'step': agent.step
    }
    torch.save(checkpoint, filename)

def load_model(agent, filename):
    checkpoint = torch.load(filename)
    agent.Q.load_state_dict(checkpoint['Q_state_dict'])
    agent.target_Q.load_state_dict(checkpoint['target_Q_state_dict'])
    agent.optim.load_state_dict(checkpoint['optimizer'])
    agent.step = checkpoint['step']
    agent.eps = agent.eps_decay(agent.step)

import os

model_save_dir = '/content/drive/My Drive/smirl_vs_dqn/'
os.makedirs(model_save_dir, exist_ok=True)  # Creates the directory if it doesn't exist
from copy import deepcopy

import gymnasium as gym
import numpy as np
import os
from pettingzoo.atari import boxing_v2
from pettingzoo.atari import pong_v3
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from wrapper import SMIRLWrapper
from buffer import BernoulliBuffer, ReplayBuffer, GaussianBuffer
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
from PIL import Image
import gc
from tqdm import tqdm

boxing_train = boxing_v2.env(obs_type='grayscale_image', full_action_space=False)
boxing_eval = boxing_v2.env(obs_type='grayscale_image', full_action_space=False, render_mode='rgb_array')
def boxing_wrappers(boxing):
    boxing = ss.dtype_v0(boxing, np.float32)
    boxing = ss.normalize_obs_v0(boxing)
    boxing = ss.max_observation_v0(boxing, 2)
    boxing = ss.frame_skip_v0(boxing, 4)
    boxing = ss.resize_v1(boxing, 84, 84)
    boxing = ss.frame_stack_v1(boxing, 4)
    boxing = ss.reshape_v0(boxing, (4, 84, 84))
    boxing = aec_to_parallel(boxing)
    return boxing

env = SMIRLWrapper(boxing_wrappers(boxing_train), GaussianBuffer,
                   use_reward=[True, "only"], smirl_coeff=0.1, max_timestep=1000)
eval_env = SMIRLWrapper(boxing_wrappers(boxing_eval), GaussianBuffer,
                        use_reward="only", max_timestep=None)


obs_space = env.observation_space(env.possible_agents[0])
num_actions = env.action_space(env.possible_agents[0])
agent_1 = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=1000, buffer=ReplayBuffer,
                 filters=[[16, 5, 2], [32, 3, 2], [64, 2, 2]], normalize_rewards=False, capacity=int(5e4))
agent_2 = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=1000, buffer=ReplayBuffer,
                 filters=[[16, 5, 2], [32, 3, 2], [64, 2, 2]], normalize_rewards=False, capacity=int(5e4))



rewards = {agent: [] for agent in env.possible_agents}
losses = {agent: [] for agent in env.possible_agents}
train_rewards = {agent: [] for agent in env.possible_agents}
reward_temp = {agent: [] for agent in env.possible_agents}
timestep = 0

for ep in tqdm(range(5000)):
    info = generate_trajectory_pz(env, [agent_1, agent_2])
    timestep += info["timestep"]
    for agent in env.possible_agents:
        losses[agent].append(info["loss"][agent])
        reward_temp[agent].append(info["total_reward"][agent])
    if ep % 50 == 0:
        print("Train Reward: ", {agent: np.mean(reward_temp[agent]) for agent in env.possible_agents})

        obs = eval_env.reset()
        total_reward = evaluate_trajectory_pz(eval_env, [agent_1, agent_2], ep, render=True)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {agent_1.eps}")
        for agent in env.possible_agents:
            train_rewards[agent].append(np.mean(reward_temp[agent]))
            rewards[agent].append(total_reward[agent])
        reward_temp = {agent: [] for agent in env.possible_agents}

        model_filename = os.path.join(model_save_dir, f'agent_1_checkpoint_epoch{ep}.pth')
        save_model(agent_1, model_filename)
        model_filename = os.path.join(model_save_dir, f'agent_2_checkpoint_epoch{ep}.pth')
        save_model(agent_2, model_filename)

        for agent in env.possible_agents:
            np.savetxt(f'/content/drive/My Drive/smirl_vs_dqn/boxing_reward_{agent}.csv', np.array(rewards[agent]))

            np.savetxt(f'/content/drive/My Drive/smirl_vs_dqn/boxing_loss_{agent}.csv', np.array(losses[agent]))

            np.savetxt(f'/content/drive/My Drive/smirl_vs_dqn/boxing_train_{agent}.csv', np.array(train_rewards[agent]))

        print(timestep)
