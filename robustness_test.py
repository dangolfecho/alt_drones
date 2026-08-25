import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np
import os
import gc
import csv
from datetime import datetime
import pandas as pd

from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy

from PyFlyt.gym_envs.quadx_envs.quadx_pole_waypoints_env import QuadXPoleWaypointsEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
from PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

from tqdm import tqdm

DEFAULT_ENV = 0
DEFAULT_ALGO = 0
DEFAULT_SUFFIX = 0
DEFAULT_CONTINUE = 0

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']
def get_model_saved(algo_str, env_name, env_test):
    save_path = f'results/{env_name}/{algo_str}.zip'
    if(algo_str == 'a2c'):
        return A2C.load(save_path, env_test)
    elif(algo_str == 'ddpg'):
        return DDPG.load(save_path, env_test)
    elif(algo_str == 'dqn'):
        return DQN.load(save_path, env_test)
    elif(algo_str == 'sac'):
        return SAC.load(save_path, env_test)
    elif(algo_str == 'td3'):
        return TD3.load(save_path, env_test)
    elif(algo_str == 'ppo'):
        return PPO.load(save_path, env_test)

def read_schedule(path):
    sch_list = []
    with open(path, "r") as fp:
        for line in fp:
            parts = line.split()
            for i in range(8):
                parts[i] = float(parts[i])
            sch_list.append(parts)
    return sch_list

def give_str(num):
    if num < 0:
        return 'm'+str(-num)
    return str(num)

def save_data(algo_str, env_str, data, schedule_suffix=0):
    df = pd.DataFrame(data)
    p_name, env_name = env_str.split('/')
    f_name = f"cfiles/{env_name}_{algo_str}_schedule{schedule_suffix}.csv"
    df.to_csv(f_name, index=True, header=['x', 'y', 'z', 'roll', 'pitch', 'yaw',
        'Mean episode rewards', 'Std episode rewards'], index_label='Config_No')

def test(algo_str, env_str, val):
    #start_pos = np.array([[config[0], config[1], config[2]]])
    #start_orn = np.array([[config[3], config[4], config[5]]])
    start_pos = np.array([[0.0, 0.0, 2.0]])
    start_orn = np.array([[0.0, val, 0.0]])
    print(val)
    pack_name, env_name= env_str.split('/')
    pack_name, env_name= env_str.split('/')
    '''
    env_test = make_vec_env(env_str, n_envs=24, vec_env_cls=SubprocVecEnv,
            env_kwargs={'render_mode': 'rgb_array', 'start_pos':start_pos,
                'start_orn':start_orn},
            vec_env_kwargs=dict(start_method='fork'),)
    '''

    env_test = gym.make(env_str, render_mode='human',
            start_pos=start_pos,
            start_orn=start_orn)
    if(str(type(env_test.observation_space)) == "<class 'gymnasium.spaces.dict.Dict'>"):
        env_test = gym.make(env_str, render_mode='rgb_array')
        context_length = 4
        env_test = FlattenWaypointEnv(env_test, context_length)
    model = get_model_saved(algo_str, env_name, env_test)
    #vec_env = model.get_env()
    mean_ep_rewards, std_ep_rewards = evaluate_policy(model, env_test, n_eval_episodes=1, deterministic=True,
            render=False,
            return_episode_rewards = False,)
    env_test.close()
    return (mean_ep_rewards, std_ep_rewards)
    '''
    obs = vec_env.reset()
    for i in range(200):
        action, states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
    '''

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO,
        file_suffix=DEFAULT_SUFFIX):
    schedule = read_schedule(f'schedule{file_suffix}.txt')
    count = 0
    data = []
    for i in range(5, 21):
        test(algos[algo_num], envs[env_num], (i*(3.14/20)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='robustness.py',
            description='checks robustness of trained drones to changes in\
                    starting position')
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which environment to train')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            trained model to use')
    parser.add_argument('file_suffix', type=int, default=DEFAULT_SUFFIX, help='1 if\
            you want to train, 0 if you want to test')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify gym_envs/quadx_envs/quadx_hover_env.py to take start position and
orientation as arguments


'''
