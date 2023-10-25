import gymnasium as gym
import numpy as np

from SMIRLAgent import SMIRLAgent
from gymnasium.wrappers import TransformObservation
from matplotlib import pyplot as plt

env = gym.make("CarRacing-v2", continuous=False)
env = TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
env.observation_space = gym.spaces.Box(0, 255, (3, 96, 96), dtype=np.uint8)
1
obs_space = env.observation_space
num_actions = env.action_space
smirl = SMIRLAgent(obs_space, num_actions, capacity=1000000, eps_decay=0.99, soft_update=None, lr=1e-3, update_freq=1,
                   start_after=10000, batch_size=128, target_update_freq=1000)

rewards = []
losses = []
timestep = 0
eps = 1
for ep in range(30000):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        if smirl.step >= smirl.start_after and smirl.step % 1000 == 0:
            eps = max(eps * 0.99, 0.05)
        timestep += 1
        action = smirl.choose_action(obs, eps)
        next_obs, reward, truncated, _, done = env.step(action)
        done = done or truncated
        loss = smirl.update(obs, action, reward, next_obs, done)
        if loss is not None and timestep % 10 == 0:
            losses.append(loss)
        obs = next_obs
    if ep % 100 == 0:
        obs, _ = env.reset()
        done = False
        while not done:
            action = smirl.choose_action(obs, 0)
            next_obs, reward, truncated, _, done = env.step(action)
            total_reward += reward
            done = done or truncated
            obs = next_obs
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {eps}")
    rewards.append(total_reward)

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
