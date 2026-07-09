from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

env = MAFixedwingDogfightEnvV2(render_mode="human")
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
