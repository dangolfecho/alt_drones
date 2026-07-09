import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import A2C, DDPG, DQN, HerReplayBuffer, SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env

from train_TD3 import train_td3

train_td3("PyFlyt/QuadX-Hover-v4", True)
'''
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
