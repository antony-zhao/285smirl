import numpy as np
from ray import tune

from model import Critic
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz
from tetris import TetrisEnv
from wrapper import SMIRLWrapper
from buffer import BernoulliBuffer, ReplayBuffer

import ray
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv, MultiAgentEnv, PettingZooEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.algorithms.ppo import PPOConfig


train_env = lambda: SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                   use_reward=[False, "only"], smirl_coeff=0.1)
eval_env = SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                        use_reward="only", max_timestep=None)

ray.init(local_mode=False)

# ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
register_env("tetris", lambda config: ParallelPettingZooEnv(train_env()))
env_name = "tetris"
env = train_env()
rollout_workers = 16
rollout_length = 100
num_envs_per = 1

batch_size = rollout_workers * rollout_length * num_envs_per
mini_batch = 8

config = (
    PPOConfig()  # Version 2.5.0
    .environment(env="tetris", disable_env_checking=True, render_env=False)  # , env_task_fn=curriculum_fn
    .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
              num_envs_per_worker=num_envs_per)
    .training(
        train_batch_size=batch_size,
        lr=5e-4,
        kl_coeff=0.2,
        kl_target=1e-3,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        clip_param=0.3,
        grad_clip=20,
        entropy_coeff=1e-2,
        vf_loss_coeff=0.05,  # 0.05
        vf_clip_param=10,  # 10 (2 vehicle)
        sgd_minibatch_size=512,
        num_sgd_iter=20,
        model={"dim": 140, "use_lstm": False, "framestack": True,  # "post_fcnet_hiddens": [512, 512],
               "vf_share_layers": True, "free_log_std": False}
    )
    .debugging(log_level="INFO")
    .framework(framework="torch")
    .resources(num_gpus=1)
    .multi_agent(
        policies=env.possible_agents,  # {"shared_policy"},
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id)  # "shared_policy")
    )
    .evaluation(

    )
)

results = tune.run(
    "PPO",
    name=f"PPO-{2}",
    verbose=0,
    metric="episode_reward_mean",
    mode="max",
    stop={"episode_reward_mean": 500},
    checkpoint_freq=10,
    # local_dir="ray_results/" + env_name,
    config=config.to_dict(),
)
