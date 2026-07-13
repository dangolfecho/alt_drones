import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def train_model(timesteps=2e6, save_path="ppo_quad_pole_balance"):
    train_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v4", n_envs=4)
    model = PPO("MultiInputPolicy", train_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    train_env.close()


if __name__ == '__main__':
    train_model()
