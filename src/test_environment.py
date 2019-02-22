import gym
import time

env = gym.make("MsPacman-v0")
env.reset()

done = False

while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

print(obs.shape)
