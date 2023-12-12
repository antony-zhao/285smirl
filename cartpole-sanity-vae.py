import gymnasium as gym
from SMIRL_VAEAgent import SMIRL_VAEAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory

env = gym.make("CartPole-v1")
obs_space = env.observation_space
num_actions = env.action_space
smirl = SMIRL_VAEAgent(obs_space, num_actions, capacity=1000000, eps_decay=0.99, soft_update=None, lr=1e-3, update_freq=1,
                       start_after=10000, batch_size=128, target_update_freq=1000)

rewards = []
losses = []
timestep = 0
eps = 1
for ep in range(30000):
    info = generate_trajectory(env, smirl)
    timestep += info["timestep"]
    losses.append(info["loss"])
    if ep % 100 == 0:
        total_reward = evaluate_trajectory(env, smirl)
        print(f"Episode {ep}: Reward {total_reward} Timestep {timestep} Eps {eps}")
        rewards.append(total_reward)

print(timestep)
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()
