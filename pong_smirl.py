from copy import deepcopy

import gymnasium as gym
import numpy as np

from pettingzoo.atari import pong_v3
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from tetris import TetrisEnv
from wrapper import SMIRLWrapper
from buffer import BernoulliBuffer, ReplayBuffer, GaussianBuffer
from pettingzoo.utils import aec_to_parallel
import supersuit as ss

pong_train = pong_v3.env(num_players=2, obs_type='grayscale_image', full_action_space=False, max_cycles=2000)
pong_eval = pong_v3.env(num_players=2, obs_type='grayscale_image', full_action_space=False, render_mode='human',
                        max_cycles=2000)


def pong_wrappers(pong):
    pong = ss.dtype_v0(pong, np.float32)
    pong = ss.normalize_obs_v0(pong)
    pong = ss.max_observation_v0(pong, 2)
    pong = ss.frame_skip_v0(pong, 4)
    pong = ss.resize_v1(pong, 84, 84)
    pong = ss.frame_stack_v1(pong, 4)
    pong = ss.reshape_v0(pong, (4, 84, 84))
    pong = aec_to_parallel(pong)
    return pong


env = SMIRLWrapper(pong_wrappers(pong_train), GaussianBuffer,
                   use_reward=[True, "only"], smirl_coeff=0.1, max_timestep=1000)
eval_env = SMIRLWrapper(pong_wrappers(pong_eval), GaussianBuffer,
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
for ep in range(5000):
    info = generate_trajectory_pz(env, [agent_1, agent_2])
    timestep += info["timestep"]
    for agent in env.possible_agents:
        losses[agent].append(info["loss"][agent])
        reward_temp[agent].append(info["total_reward"][agent])
    if ep % 50 == 0:
        print("Train Reward: ", {agent: np.mean(reward_temp[agent]) for agent in env.possible_agents})
        total_reward = evaluate_trajectory_pz(eval_env, [agent_1, agent_2], render=True)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {agent_1.eps}")
        for agent in env.possible_agents:
            train_rewards[agent].append(np.mean(reward_temp[agent]))
            rewards[agent].append(total_reward[agent])
        reward_temp = {agent: [] for agent in env.possible_agents}

print(timestep)
for agent in env.possible_agents:
    plt.plot(rewards[agent])
    np.savetxt(f'pong_reward_{agent}.csv', np.array(rewards[agent]))
    plt.show()

    plt.plot(losses[agent])
    np.savetxt(f'pong_losses_{agent}.csv', np.array(rewards[agent]))
    plt.show()

    plt.plot(train_rewards[agent])
    np.savetxt(f'pong_train_{agent}.csv', np.array(rewards[agent]))
    plt.show()
