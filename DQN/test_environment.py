import gym
import time

env = gym.make("MsPacman-ram-v0")
env.reset()
print(env.action_space)

done = False

while not done:
    obs, reward, done, info = env.step(9)
    env.render()

print(obs)
