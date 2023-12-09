import gymnasium as gym
import numpy as np

from SMIRL_VAEAgent import SMIRL_VAEAgent
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from tetris import TetrisEnv
from wrapper import SMIRLWrapper
from buffer import BernoulliBuffer, ReplayBuffer

env = SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                   use_reward=[False, "only"], smirl_coeff=0.1)
eval_env = SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                        use_reward="only", max_timestep=None)
obs_space = env.observation_space(env.possible_agents[0])
num_actions = env.action_space(env.possible_agents[0])
agent_1 = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=8000, buffer=ReplayBuffer,
                 filters=[[16, 5, 1], [32, 3, 1], [64, 2, 1]], normalize_rewards=False)
agent_2 = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=8000, buffer=ReplayBuffer,
                 filters=[[16, 5, 1], [32, 3, 1], [64, 2, 1]], normalize_rewards=False)

rewards = {agent: [] for agent in env.possible_agents}
losses = {agent: [] for agent in env.possible_agents}
train_rewards = {agent: [] for agent in env.possible_agents}
reward_temp = {agent: [] for agent in env.possible_agents}
timestep = 0
for ep in range(10000):
    info = generate_trajectory_pz(env, [agent_1, agent_2])
    timestep += info["timestep"]
    for agent in env.possible_agents:
        losses[agent].append(info["loss"][agent])
        reward_temp[agent].append(info["total_reward"][agent])
    if ep % 100 == 0:
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
    plt.show()

    plt.plot(losses[agent])
    plt.show()

    plt.plot(train_rewards[agent])
    plt.show()
