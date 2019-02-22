import gym
from q_learning_agent import QLearningAgent

env = gym.make("Taxi-v2")

agent = QLearningAgent(env)
agent.train(to_render=False)

agent.test()
