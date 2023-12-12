import gymnasium as gym
from SMIRL_VAEAgent import SMIRL_VAEAgent
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory
from gymnasium.wrappers import NormalizeReward, TransformReward

env = NormalizeReward(TransformReward(gym.make("ALE/Tetris-ram-v5"), lambda r: r + 0.01))
render_env = gym.make("ALE/Tetris-ram-v5")
obs_space = env.observation_space
num_actions = env.action_space
agent = DQNAgent(obs_space, num_actions, capacity=1000000, eps_decay=0.99, soft_update=None, lr=1e-4, update_freq=1,
                 start_after=10000, batch_size=128, target_update_freq=10000, eps_decay_per=5000)

rewards = []
losses = []
timestep = 0
eps = 1
for ep in range(1000):
    info = generate_trajectory(env, agent)
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep % 100 == 0:
        total_reward = evaluate_trajectory(render_env, agent, render)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {agent.eps}")
        rewards.append(total_reward)

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
