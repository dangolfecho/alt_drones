import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np
import os
import gc

from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

from PyFlyt.gym_envs.quadx_envs.quadx_pole_waypoints_env import QuadXPoleWaypointsEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
from PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv
from PyFlyt.gym_envs import FlattenWaypointEnv


DEFAULT_ENV = 0
DEFAULT_ALGO = 0
DEFAULT_TRAIN = 0

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]
#quadpolewaypoints as well as quadwaypoints bug
#fixedwingwaypoints also - 3, 4, 5

env_config = {
        'render_mode': 'rgb_array',
}


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
            env = QuadXFixedwingWaypointsEnv(**config)
            context_length = config.get('context_length', 4)
            env = FlattenWaypointEnv(env, context_length)
            return env
    return create_env

def get_model(algo_str, env_train, n_actions, dict_flag):
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

def get_model_test(algo_str, env_name, env_test):
    save_path = f'backup/results/{env_name}/{algo_str}.zip'
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

def run(algo_str, env_str, timesteps=1e4, to_train=True):
    if(to_train):
        train(algo_str, env_str, timesteps)
    else:
        test(algo_str, env_str)

def train(algo_str, env_str, timesteps=1e4):
    pack_name, env_name = env_str.split('/')
    dict_flag = 0
    env_train = make_vec_env(env_str, n_envs=16, vec_env_cls=SubprocVecEnv,
            env_kwargs={'render_mode': 'rgb_array'},
            vec_env_kwargs=dict(start_method='fork'),)
    if (str(type(env_train.observation_space)) == "<class 'gymnasium.spaces.dict.Dict'>"):
        dict_flag = 1
        env_train = make_vec_env(reg_env_creator(env_config, env_str), n_envs=16, seed=0,
                vec_env_cls=SubprocVecEnv,)
    log_path = f'results/{env_name}/{algo_str}/'
    if(not(os.path.isdir(f'results/{env_name}/'))):
        os.mkdir(f'results/{env_name}/') 
    if(not(os.path.isdir(f'results/{env_name}/{algo_str}/'))):
        os.mkdir(f'results/{env_name}/{algo_str}/') 
    new_logger = configure(log_path, ['csv'])

    n_actions = env_train.action_space.shape[-1]
    model = get_model(algo_str, env_train, n_actions, dict_flag)
    model.set_logger(new_logger)
    if(algo_str == 'ppo'):
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True)
    else:
        model.learn(total_timesteps=timesteps, log_interval=10, progress_bar=True)
    model.save(f'results/{env_name}/{algo_str}')
    del env_train
    gc.collect()
    del model
    gc.collect()

def test(algo_str, env_str):
    pack_name, env_name= env_str.split('/')
    env_test = gym.make(env_str, render_mode='human')
    if(str(type(env_test.observation_space)) == "<class 'gymnasium.spaces.dict.Dict'>"):
        dict_flag = 1
        env_test = gym.make(env_str, render_mode='human')
        context_length = 4
        env_test = FlattenWaypointEnv(env_test, context_length)
    model = get_model_test(algo_str, env_name, env_test)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(200):
        action, states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO, train_flag=DEFAULT_TRAIN):
    #ts = 2e6
    ts = 1e5
    algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']
    env = envs[env_num]
    print(env)
    run(algos[algo_num], env, ts, train_flag)
    #run(algos[algo_num], env, ts, False)
    #run(algos[algo_num], env, ts, True)
    #run("dqn", env, ts, True) - since dqn only works for discrete environments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='full_runner.py',
            description='does training runs',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which environment to train')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to train')
    parser.add_argument('train_flag', type=int, default=DEFAULT_TRAIN, help='1 if\
            you want to train, 0 if you want to test')
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
