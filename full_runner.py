import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import pandas as pd
import numpy as np
import os
import gc
import csv
from datetime import datetime

from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from PyFlyt.gym_envs.quadx_envs.quadx_pole_waypoints_env import QuadXPoleWaypointsEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
from PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

DEFAULT_ENV = 0
DEFAULT_ALGO = 0
DEFAULT_TRAIN = 0
DEFAULT_CONTINUE = 0
DEFAULT_REWARD_FLAG = 0#0 for dense rewards 1 for sparse

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]
#quadpolewaypoints as well as quadwaypoints bug
#fixedwingwaypoints also - 3, 4, 5

env_config = {
        'render_mode': 'rgb_array',
}

def get_date():
    return datetime.today().strftime('%Y-%m-%d %H-%M-%S')


def reg_env_creator(config, env_str):
    if(env_str == envs[3]):
        def create_env():
            env = QuadXPoleWaypointsEnv(**config)
            context_length = config.get('context_length', 4)
            env = FlattenWaypointEnv(env, context_length)
            return env
    elif(env_str == envs[4]):
        def create_env():
            env = QuadXWaypointsEnv(**config)
            context_length = config.get('context_length', 4)
            env = FlattenWaypointEnv(env, context_length)
            return env
    elif(env_str == envs[5]):
        def create_env():
            env = FixedwingWaypointsEnv(**config)
            context_length = config.get('context_length', 4)
            env = FlattenWaypointEnv(env, context_length)
            return env
    return create_env

def get_model_fresh(algo_str, env_train, n_actions):
    policy_type = 'MlpPolicy'
    if(algo_str == 'a2c'):
        return A2C(policy_type, env_train, verbose=1)
    elif(algo_str == 'ddpg'):
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return DDPG(policy_type, env_train, action_noise=action_noise, verbose=1)
    elif(algo_str == 'dqn'):
        return DQN(policy_type, env_train, verbose=1)
    elif(algo_str == 'sac'):
        return SAC(policy_type, env_train, verbose=1)
    elif(algo_str == 'td3'):
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3(policy_type, env_train, action_noise=action_noise, verbose=1)
    elif(algo_str == 'ppo'):
        return PPO(policy_type, env_train, verbose=1)

def get_model_saved(algo_str, env_name, env_test, reward_flag):
    if(reward_flag):
        save_path = f'results/sparse/{env_name}/{algo_str}.zip'
    else:
        save_path = f'results/dense/{env_name}/{algo_str}.zip'
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

def model_exists(algo_str, env_name, reward_flag):
    if(reward_flag):
        save_path = f'results/sparse/{env_name}/{algo_str}.zip'
    else:
        save_path = f'results/dense/{env_name}/{algo_str}.zip'
    if (os.path.isfile(save_path)):
        return 1
    else:
        return 0

def get_run_num(log_path):
    dirs = os.listdir(log_path)
    if(len(dirs) == 0):
        return 0
    run_nums = []
    for filename in dirs:
        if('run' in filename):
            _, num = filename.split('_')#run_i gets split into run, i
            run_nums.append(int(num))
    num = max(run_nums)
    return num

def run(algo_str, env_str, timesteps=1e4, to_train=True, continue_training=1,
        reward_flag=0):
    if(to_train):
        train(algo_str, env_str, timesteps, continue_training, reward_flag)
    else:
        test(algo_str, env_str, reward_flag)

def train(algo_str, env_str, timesteps=1e4, continue_training=1,
        reward_flag=True):
    Z = 10.0
    start_state = np.array([[0.0, 0.0, Z]])
    goal_state = np.array([0.0, 0.0, Z])
    pack_name, env_name = env_str.split('/')
    n_envs = 16
    env_train = make_vec_env(env_str, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
            env_kwargs={'render_mode': 'rgb_array',
                'start_pos': start_state,
                'goal_state': goal_state,
                'sparse_reward': reward_flag,
                'flight_dome_size': 50.0,},
            vec_env_kwargs=dict(start_method='fork'),)
    if (str(type(env_train.observation_space)) == "<class 'gymnasium.spaces.dict.Dict'>"):
        env_train = make_vec_env(reg_env_creator(env_config, env_str), n_envs=16, seed=0,
                vec_env_cls=SubprocVecEnv,)
    if(reward_flag):
        log_path = f'results/sparse/{env_name}/{algo_str}/'
        if(not(os.path.isdir(f'results/sparse/{env_name}/'))):
            os.mkdir(f'results/sparse/{env_name}/') 
        if(not(os.path.isdir(f'results/sparse/{env_name}/{algo_str}/'))):
            os.mkdir(f'results/sparse/{env_name}/{algo_str}/') 
    else:
        log_path = f'results/dense/{env_name}/{algo_str}/'
        if(not(os.path.isdir(f'results/dense/{env_name}/'))):
            os.mkdir(f'results/dense/{env_name}/') 
        if(not(os.path.isdir(f'results/dense/{env_name}/{algo_str}/'))):
            os.mkdir(f'results/dense/{env_name}/{algo_str}/') 
    
    run_num = get_run_num(log_path)
    log_path += f'run_{run_num+1}/'
    new_logger = configure(log_path, ['csv'])
    n_actions = env_train.action_space.shape[-1]
    print(model_exists(algo_str, env_name, reward_flag))
    if(model_exists(algo_str, env_name, reward_flag)):
        if(continue_training):
            model = get_model_saved(algo_str, env_name, env_train, reward_flag)
            model.set_logger(new_logger)
        else:
            model = get_model_fresh(algo_str, env_train, n_actions)
            model.set_logger(new_logger)
    else:
        model = get_model_fresh(algo_str, env_train, n_actions)
        model.set_logger(new_logger)


    checkpoint_callback = CheckpointCallback(
            save_freq=int(2000),
            save_path=log_path+'models/',
            name_prefix="iter_",
    )
    if(algo_str == 'ppo'):
        model.learn(total_timesteps=timesteps, log_interval=1,
                progress_bar=True, callback=checkpoint_callback)
    else:
        model.learn(total_timesteps=timesteps, log_interval=10,
                progress_bar=True, callback=checkpoint_callback)
    if(reward_flag):
        model.save(f'results/sparse/{env_name}/{algo_str}')
    else:
        model.save(f'results/dense/{env_name}/{algo_str}')

    if(reward_flag):
        if(continue_training):
            if(os.path.isfile(f'results/sparse/{env_name}/{algo_str}/info.txt')):
                with open(f'results/sparse/{env_name}/{algo_str}/info.txt', 'a', newline='') as fp:
                    csv_writer = csv.writer(fp, delimiter=',')
                    csv_writer.writerow([timesteps, get_date()])
            else:
                with open(f'results/sparse/{env_name}/{algo_str}/info.txt', 'w', newline='') as fp:
                    csv_writer = csv.writer(fp, delimiter=',')
                    csv_writer.writerow(['No_of_iterations', 'Date'])
                    csv_writer.writerow([timesteps, get_date()])
        else:
            with open(f'results/sparse/{env_name}/{algo_str}/info.txt', 'w', newline='') as fp:
                csv_writer = csv.writer(fp, delimiter=',')
                csv_writer.writerow(['No_of_iterations', 'Date'])
                csv_writer.writerow([timesteps, get_date()])
    else:
        if(continue_training):
            if(os.path.isfile(f'results/dense/{env_name}/{algo_str}/info.txt')):
                with open(f'results/dense/{env_name}/{algo_str}/info.txt', 'a', newline='') as fp:
                    csv_writer = csv.writer(fp, delimiter=',')
                    csv_writer.writerow([timesteps, get_date()])
            else:
                with open(f'results/dense/{env_name}/{algo_str}/info.txt', 'w', newline='') as fp:
                    csv_writer = csv.writer(fp, delimiter=',')
                    csv_writer.writerow(['No_of_iterations', 'Date'])
                    csv_writer.writerow([timesteps, get_date()])
        else:
            with open(f'results/dense/{env_name}/{algo_str}/info.txt', 'w', newline='') as fp:
                csv_writer = csv.writer(fp, delimiter=',')
                csv_writer.writerow(['No_of_iterations', 'Date'])
                csv_writer.writerow([timesteps, get_date()])

    del env_train
    gc.collect()
    del model
    gc.collect()

def test(algo_str, env_str, reward_flag=True):
    pack_name, env_name= env_str.split('/')
    #env_test = gym.make(env_str, render_mode='human')
    env_test = gym.make(env_str, render_mode='rgb_array')
    if(str(type(env_test.observation_space)) == "<class 'gymnasium.spaces.dict.Dict'>"):
        env_test = gym.make(env_str, render_mode='human')
        context_length = 4
        env_test = FlattenWaypointEnv(env_test, context_length)
    model = get_model_saved(algo_str, env_name, env_test, reward_flag)
    vec_env = model.get_env()
    obs = vec_env.reset()
    reward_list = []
    for i in range(10):
    #for i in range(200):
        action, states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        reward_list.append(rewards)
    reward_df = pd.DataFrame(reward_list)
    reward_df.to_csv('rewards.csv')

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO, train_flag=DEFAULT_TRAIN,
        continue_training=DEFAULT_CONTINUE, reward_flag=DEFAULT_REWARD_FLAG):
    ts = 2e6
    #ts = 2e3
    algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']
    env = envs[env_num]
    print(env)
    run(algos[algo_num], env, ts, train_flag, continue_training, reward_flag)
    #run(algos[algo_num], env, ts, False)
    #run(algos[algo_num], env, ts, True)
    #run("dqn", env, ts, True) - since dqn only works for discrete environments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='full_runner.py',
            description='does training runs',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
        environment to use')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to train')
    parser.add_argument('train_flag', type=int, default=DEFAULT_TRAIN, help='1 if\
            you want to train, 0 if you want to test')
    parser.add_argument('continue_training', type=int, default=DEFAULT_CONTINUE,
            help='1 if use pre_existing model, 0 if fresh training')
    parser.add_argument('reward_flag', type=int, default=DEFAULT_REWARD_FLAG,
            help='sets dense or sparse rewards')
    ARGS = parser.parse_args()
    main(**vars(ARGS))
'''
for env in envs:
    print(env)
    run("a2c", env, ts, True)
    run("ddpg", env, ts, True)
    #run("dqn", env, ts, True) - since dqn only works for discrete environments
    run("sac", env, ts, True)
    run("td3", env, ts, True)
    run("ppo", env, 2048*16*(1), True)

for env in envs:
    #run("a2c", env, to_train=False)
    #run("ddpg", env, to_train=False)
    #run("dqn", env, to_train=False)
    #run("sac", env, to_train=False)
    #run("td3", env, to_train=False)
    run("ppo", env, to_train=False)
#vec_env = make_vec_env("CartPole-v1", n_envs=4)
vec_env = make_vec_env("PyFlyt/QuadX-Hover-v4", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_quadx_hover")

del model 

model = A2C.load("a2c_quadx_hover")

obs = vec_env.reset()

for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    vec_env.render("human")
'''
