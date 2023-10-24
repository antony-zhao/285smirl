import gymnasium as gym
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt

env = gym.make("CartPole-v1")
obs_space = env.observation_space
num_actions = env.action_space
dqn = DQNAgent(obs_space, num_actions, capacity=100000, eps_decay=0.99, soft_update=None, lr=1e-3, update_freq=1,
               start_after=30000, batch_size=128, target_update_freq=1000)

rewards = []
losses = []
timestep = 0
eps = 1
for ep in range(30000):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        if dqn.step >= dqn.start_after and dqn.step % 1000 == 0:
            eps = max(eps * 0.99, 0.05)
        timestep += 1
        action = dqn.choose_action(obs, eps)
        next_obs, reward, truncated, _, done = env.step(action)
        done = done or truncated
        loss = dqn.update(obs, action, reward, next_obs, done)
        if loss is not None and timestep % 10 == 0:
            losses.append(loss)
        # total_reward += reward
        obs = next_obs
    if ep % 100 == 0:
        obs, _ = env.reset()
        done = False
        while not done:
            action = dqn.choose_action(obs, 0)
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
