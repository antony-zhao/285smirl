import gymnasium as gym
import numpy as np

from SMIRLAgent import SMIRLAgent
from gymnasium.wrappers import TransformObservation
from matplotlib import pyplot as plt

from trajectory_utils import generate_trajectory, evaluate_trajectory

env = gym.make("CarRacing-v2", continuous=False)
env = TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
env.observation_space = gym.spaces.Box(0, 255, (3, 96, 96), dtype=np.uint8)

obs_space = env.observation_space
num_actions = env.action_space
smirl = SMIRLAgent(obs_space, num_actions, capacity=1000000, eps_decay=0.99, soft_update=None, lr=1e-3, update_freq=1,
                   start_after=10000, batch_size=128, target_update_freq=1000)

rewards = []
losses = []
timestep = 0

for ep in range(30000):
    info = generate_trajectory(env, smirl)
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep % 100 == 0:
        total_reward = evaluate_trajectory(env, smirl)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep}")
        rewards.append(total_reward)

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
