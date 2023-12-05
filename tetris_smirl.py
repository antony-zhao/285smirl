import gymnasium as gym
from SMIRL_VAEAgent import SMIRL_VAEAgent
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from tetris import TetrisEnv
from buffer import BernoulliBuffer

env = TetrisEnv()
obs_space = env.observation_space
num_actions = env.action_space
agent = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=1000,
                 filters=[[3, 3], [4, 4]])

rewards = []
losses = []
timestep = 0
for ep in range(100000):
    info = generate_trajectory_pz(env, [agent])
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep % 1000 == 0:
        total_reward = evaluate_trajectory_pz(env, [agent], render=True)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {agent.eps}")
        rewards.append(total_reward[env.possible_agents[0]])

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
