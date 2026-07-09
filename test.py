import gymnasium
import PyFlyt.gym_envs
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

#vec_env = make_vec_env("CartPole-v1", n_envs=4)
vec_env = make_vec_env("PyFlyt/QuadX-Hover-v4", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
#model.learn(total_timesteps=25000)
#model.save("a2c_quadx_hover")

#del model 

model = A2C.load("a2c_quadx_hover")

obs = vec_env.reset()

vec_env.render("human")

term, trunc = False, False

while not(term or trunc):
    action, _states = model.predict(obs)
    obs, rew, done, info = vec_env.step(action)
