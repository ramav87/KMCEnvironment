from gym.envs.registration import register

register(
    id='kmcenv-v0',
    entry_point='kmc_env.envs:KmcEnv',
)
