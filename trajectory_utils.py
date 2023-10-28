import numpy as np


def generate_trajectory(env, agent):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    losses = []
    timestep = 0
    while not done:
        timestep += 1
        action = agent.choose_action(obs)
        next_obs, reward, truncated, info, done = env.step(action)
        if type(info) is bool:
            done, info = info, done
        total_reward += reward
        done = done or truncated
        loss = agent.update(obs, action, reward, next_obs, done)
        if loss is not None:
            losses.append(loss)

        obs = next_obs

    return {"total_reward": total_reward, "loss": np.mean(losses), "timestep": timestep}


def evaluate_trajectory(env, agent):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(obs, explore=False)
        next_obs, reward, truncated, info, done = env.step(action)
        if type(info) is bool:
            done, info = info, done
        total_reward += reward
        done = done or truncated
        obs = next_obs

    return total_reward
