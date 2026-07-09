import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

def train_td3(env_name, train=True):
    print("ENV_NAME ", env_name)
    lib_name, ac_env = env_name.split('/')
    env_train = make_vec_env(env_name, n_envs=8, vec_env_cls=SubprocVecEnv,
                             env_kwargs={'render_mode': 'rgb_array'},
                             vec_env_kwargs=dict(start_method='fork'))
    env_train.seed(0)
    env_display = gym.make(env_name, render_mode='human')

    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 *
                                    np.ones(n_actions))
    log_path = 'results/td3/'
    new_logger = configure(log_path, ["csv"])

    model = TD3("MlpPolicy", env_train, action_noise=action_noise, verbose=1)
    model.set_logger(new_logger)
    if train:
        model.learn(total_timesteps=1e4, log_interval=10)
        model.save("td3"+ac_env)

        model = TD3.load("td3"+ac_env, env=env_display)
        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in range(200):
            action, states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)

    if not train:
        model = TD3.load("td3"+ac_env, env=env_display)

        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in range(200):
            action, states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
