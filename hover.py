import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Hover-v4", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()

while not (term or trunc):
    obs, rev, term, trunc, _ = env.step(env.action_space.sample())
