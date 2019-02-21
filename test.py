import gym_pacman
import gym

env = gym.make("BerkeleyPacman-v0")
done = False

while not done:
    env.render()
    env.step(env.action_space.sample())
