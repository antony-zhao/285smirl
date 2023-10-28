import gymnasium as gym
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt

from trajectory_utils import generate_trajectory, evaluate_trajectory

env = gym.make("CartPole-v1")
obs_space = env.observation_space
num_actions = env.action_space
dqn = DQNAgent(obs_space, num_actions, capacity=100000, eps_decay=0.99, soft_update=None, lr=1e-3, update_freq=1,
               start_after=30000, batch_size=128, target_update_freq=1000)

rewards = []
losses = []
timestep = 0
for ep in range(30000):
    info = generate_trajectory(env, dqn)
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep % 100 == 0:
        total_reward = evaluate_trajectory(env, dqn)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep}")
        rewards.append(total_reward)

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
