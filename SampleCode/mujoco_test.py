import gymnasium as gym

env = gym.make('Hopper-v4')
print(env.reset(seed = 0))