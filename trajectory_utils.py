import numpy as np
import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


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
    obs, _ = env.reset()
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
    obs, _ = env.reset()
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
                action[agent] = agents[i].choose_action(obs[agent], explore=False)

        next_obs, reward, terminations, truncations, info = env.step(action)

        for i, agent in enumerate(env.possible_agents):
            total_rewards[agent] += reward[agent]

        obs = next_obs
    if render:
        env.close()

    return total_rewards


def tianshou_eval(train_env, network, test_env=None):
    def get_env(env):
        return lambda: PettingZooEnv(env)

    def get_agents():
        env = train_env
        observation_space = env.observation_space(env.agents[0])
        agents = []
        for _ in env.possible_agents:
            # model
            net = deepcopy(network)

            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
            agent_learn = DQNPolicy(
                net,
                optim,
                discount_factor=0.99,
                estimation_step=4,
                target_update_freq=10000
            )
            agents.append(agent_learn)

        policy = MultiAgentPolicyManager(agents, PettingZooEnv(train_env))
        return policy, agents

    def train_agent(
    ) -> Tuple[dict, BasePolicy]:

        # ======== environment setup =========
        train_envs = DummyVectorEnv([get_env(train_env) for _ in range(8)])
        test_envs = DummyVectorEnv([get_env(test_env) if test_env is not None else get_env(train_env)
                                    for _ in range(1)])

        # ======== agent setup =========
        policy, agents = get_agents()

        # ======== collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(int(1e6), len(train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)

        # ======== tensorboard logging setup =========
        log_path = os.path.join('logdir', 'tic_tac_toe', 'dqn')
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        # ======== callback functions used during training =========
        def save_best_fn(policy):
            model_save_path = os.path.join(
                'logdir', 'tic_tac_toe', 'dqn', 'policy.pth'
            )
            torch.save(
               agents[0].state_dict(), model_save_path
            )

        def stop_fn(mean_rewards):
            return mean_rewards >= 500

        def train_fn(epoch, env_step):
            train_eps = max(0.99 ** (env_step / 8), 0.05)
            for i in range(len(agents)):
                agents[i].set_eps(train_eps)
            agents[0].set_eps(1)

        def test_fn(epoch, env_step):
            for i in range(len(agents)):
                agents[i].set_eps(0)
            agents[0].set_eps(1)

        def reward_metric(rews):
            return rews[:, 1]

        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=int(1e4),
            step_per_epoch=1000 * len(train_envs),
            step_per_collect=4,
            batch_size=256,
            episode_per_test=len(test_envs),
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=1,
            logger=logger,
            test_in_train=False,
            reward_metric=reward_metric
        )

        return result, agents[0]

    return train_agent()
