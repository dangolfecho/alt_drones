import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np

from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

envs = ["PyFlyt/QuadX-Hover-v4"]

def get_model(model_str, env_train, n_actions):
    if(model_str == 'a2c'):
        return A2C("MlpPolicy", env_train, verbose=1)
    elif(model_str == 'ddpg'):
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return DDPG("MlpPolicy", env_train, action_noise=action_noise, verbose=1)
    elif(model_str == 'dqn'):
        return DQN("MlpPolicy", env_train, verbose=1)
    elif(model_str == 'sac'):
        return SAC("MlpPolicy", env_train, verbose=1)
    elif(model_str == 'td3'):
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3("MlpPolicy", env_train, action_noise=action_noise, verbose=1)
    elif(model_str == 'ppo'):
        return PPO("MlpPolicy", env_train, verbose=1)

def get_model_test(model_str, env_name, env_test):
    if(model_str == 'a2c'):
        return A2C.load(model_str+env_name, env_train)
    elif(model_str == 'ddpg'):
        return DDPG.load(model_str+env_name, env_train)
    elif(model_str == 'dqn'):
        return DQN.load(model_str+env_name, env_train)
    elif(model_str == 'sac'):
        return SAC.load(model_str+env_name, env_train)
    elif(model_str == 'td3'):
        return TD3.load(model_str+env_name, env_train)
    elif(model_str == 'ppo'):
        return PPO.load(model_str+env_name, env_train)

def run(model_str, env_name, timesteps=1e4, to_train=True):
    if(to_train):
        train(model_str, env_name, timesteps)
    else:
        test(model_str, env_name)

def train(model_str, env_name, timesteps=1e4):
    pack_name, actual_name = env_name.split('/')
    env_train = make_vec_env(env_name, n_envs=16, vec_env_cls=SubprocVecEnv,
            env_kwargs={'render_mode': 'rgb_array'},
            vec_env_kwargs=dict(start_method='fork'))
    env_train.seed(0)
    env_test = gym.make(env_name, render_mode='human')

    log_path = 'results/' + model_str + '/'
    new_logger = configure(log_path, ['csv'])

    n_actions = env_train.action_space.shape[-1]
    model = get_model(model_str, env_train, n_actions)
    model.set_logger(new_logger)
    if(model_str == 'ppo'):
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True)
    else:
        model.learn(total_timesteps=timesteps, log_interval=10, progress_bar=True)
    model.save(model_str+actual_name)
    print("done")

def test(model_str, env_name):
    pack_name, actual_name = env_name.split('/')
    model = get_model_test(model_str, actual_name, env_test)
    vec_env = model.get_env()
    obs = vec_env.rest()
    for i in range(200):
        action, states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
    
ts = 2e6
for env in envs:
    run("a2c", env, ts, True)
    run("ddpg", env, ts, True)
    #run("dqn", env, ts, True) - since dqn only works for discrete environments
    run("sac", env, ts, True)
    run("td3", env, ts, True)
    run("ppo", env, 2048*16*(62), True)

'''
for env in envs:
    run("a2c", env, ts, False)
    run("ddpg", env, ts, False)
    run("dqn", env, ts, False)
    run("sac", env, ts, False)
    run("td3", env, ts, False)
    run("ppo", env, ts, False)
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
