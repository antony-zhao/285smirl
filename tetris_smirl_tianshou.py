import numpy as np
from model import Critic
from DQNAgent import DQNAgent
from matplotlib import pyplot as plt
from trajectory_utils import generate_trajectory, evaluate_trajectory_pz, generate_trajectory_pz, tianshou_eval
from tetris import TetrisEnv
from wrapper import SMIRLWrapper
from buffer import BernoulliBuffer, ReplayBuffer
from pettingzoo.utils import parallel_to_aec


train_env = parallel_to_aec(SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                   use_reward=[False, "only"], smirl_coeff=0.1))
eval_env = parallel_to_aec(SMIRLWrapper(TetrisEnv(shape=(20, 4), num_players=2, full_obs=True), BernoulliBuffer,
                        use_reward="only", max_timestep=None))
obs_space = train_env.observation_space(train_env.possible_agents[0])
num_actions = train_env.action_space(train_env.possible_agents[0])

tianshou_eval(train_env, Critic(obs_space, num_actions, filters=[[16, 5, 2], [32, 3, 1], [64, 2, 1]]), eval_env)