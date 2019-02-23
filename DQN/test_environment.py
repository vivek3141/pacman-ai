import gym
import time

env = gym.make("CartPole-v0")
env.reset()
print(env.action_space)

done = False

while not done:
    obs, reward, done, info = env.step(1)
    env.render()

print(obs)
