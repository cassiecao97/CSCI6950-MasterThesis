from gym.envs.registration import register

register(
    id='gridworld-v2',
    entry_point='gridworld.envs:GridworldEnv',
    kwargs={'map_file': 'env1.txt'},
)
