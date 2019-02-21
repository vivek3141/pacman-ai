from PIL import Image
import gym
import gym_pacman
import time

env = gym.make('BerkeleyPacman-v0')
env.seed(1)


done = False

while True:
    done = False
    env.reset()
    i = 0
    while i < 5:
        i += 1
        s_, r, done, info = env.step(env.action_space.sample())
        env.render()
        