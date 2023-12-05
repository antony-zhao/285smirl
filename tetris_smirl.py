import gymnasium as gym
from SMIRL_VAEAgent import SMIRL_VAEAgent
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from tetris import TetrisEnv
from wrapper import SoftReset
from buffer import BernoulliBuffer, ReplayBuffer

env = SoftReset(TetrisEnv())
eval_env = TetrisEnv()
obs_space = env.observation_space(env.possible_agents[0])
num_actions = env.action_space(env.possible_agents[0])
agent = DQNAgent(obs_space, num_actions, lr=1e-4, update_freq=1, start_after=10000,
                 batch_size=256, target_update_freq=20000, eps_decay_per=5000, buffer=BernoulliBuffer,
                 filters=[[16, 5, 2], [32, 3, 1], [64, 2, 1]], normalize_rewards=False)

rewards = []
losses = []
timestep = 0
for ep in range(10000):
    info = generate_trajectory_pz(env, [agent])
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep > 0 and ep % 100 == 0:
        total_reward = evaluate_trajectory_pz(eval_env, [agent], render=True)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {agent.eps}")
        rewards.append(total_reward[env.possible_agents[0]])

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
