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


def generate_trajectory_pz(env, agents):
    obs = env.reset()
    terminations = {agent: False for agent in env.possible_agents}
    truncations = {agent: False for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    losses = {agent: [] for agent in env.possible_agents}
    timestep = 0
    while False in terminations.values() and False in truncations.values():
        action = {}
        timestep += 1
        for i, agent in enumerate(env.possible_agents):
            if truncations[agent] or terminations[agent]:
                action[agent] = None
            else:
                action[agent] = agents[i].choose_action(obs[agent])

        next_obs, reward, terminations, truncations, info = env.step(action)

        for i, agent in enumerate(env.possible_agents):
            total_rewards[agent] += reward[agent]
            if not truncations[agent] or terminations[agent]:
                loss = agents[i].update(obs[agent], action[agent], reward[agent], next_obs[agent],
                                        terminations[agent] or truncations[agent])
                if loss is not None:
                    losses[agent].append(loss)

        obs = next_obs

    return {"total_reward": total_rewards, "loss": {agent: np.mean(losses[agent]) for agent in env.possible_agents},
            "timestep": timestep}


def evaluate_trajectory_pz(env, agents, render=False):
    obs = env.reset()
    terminations = {agent: False for agent in env.possible_agents}
    truncations = {agent: False for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}

    timestep = 0
    while False in terminations.values() and False in truncations.values():
        if render:
            env.render()
        action = {}
        timestep += 1
        for i, agent in enumerate(env.possible_agents):
            if truncations[agent] or terminations[agent]:
                action[agent] = None
            else:
                action[agent] = agents[i].choose_action(obs[agent])

        next_obs, reward, terminations, truncations, info = env.step(action)

        for i, agent in enumerate(env.possible_agents):
            total_rewards[agent] += reward[agent]

        obs = next_obs
    if render:
        env.close()

    return total_rewards
