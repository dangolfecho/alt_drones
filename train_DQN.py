import gymnasium as gym
import PyFlyt.gym_envs

from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

def train_dqn(env_name, train=True):
    print("ENV_NAME ", env_name)
    lib_name, ac_env = env_name.split('/')
    env_train = make_vec_env(env_name, n_envs=16, vec_env_cls=SubprocVecEnv,
                             env_kwargs={'render_mode': 'rgb_array'},
                             vec_env_kwargs=dict(start_method='fork'))
    env_train.seed(0)
    env_display = gym.make(env_name, render_mode='human')

    log_path = 'results/dqn/'
    new_logger = configure(log_path, ["csv"])

    model = DQN("MlpPolicy", env_train, verbose=1)
    model.set_logger(new_logger)
    if train:
        model.learn(total_timesteps=1e5, log_interval=10)
        model.save("dqn"+ac_env)

        model = DQN.load("dqn"+ac_env, env=env_display)
        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in range(200):
            action, states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)

    if not train:
        model = DQN.load("dqn"+ac_env, env=env_display)

        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in range(200):
            action, states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
