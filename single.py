import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def train_model(timesteps=1e4, save_path="ppo_quad_pole_balance"):
    train_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v4", n_envs=16)
    model = A2C("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(save_path)
    train_env.close()


if __name__ == '__main__':
    train_model()
