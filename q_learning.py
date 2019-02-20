import numpy as np
import gym


class QLearningAgent:
    def __init__(self, env):
        # Initialize empty q table
        self.q = np.zeros([env.observation_space.n, env.action_space.n])
        self.env = env

    def train(self, num_episodes, learning_rate, gamma, to_print=True, interval=50, to_render=True):
        if to_print:
            print("Training...\n")

        for i in range(num_episodes):
            done = False
            total_reward = 0

            s = self.env.reset()

            while not done:
                action = np.argmax(self.q[s])
                state, reward, done, info = self.env.step(action)

                # Q[s_t, a_t] = a * (r + gamma * max(Q[s_t+1, a_t])) -> Q Learning formula
                # https://en.wikipedia.org/wiki/Q-learning
                self.q[s, action] = learning_rate * (reward + gamma * np.max(self.q[state]))

                s = state
                total_reward += reward

                if to_render:
                    self.env.render()

            if i % interval == 0 and to_print:
                print(f"Episode: {i}, Reward{total_reward}")

    def test(self, num_episodes, to_print=True, to_render=True):
        if to_print:
            print("Testing...\n")

        for i in range(num_episodes):
            done = False
            total_reward = 0
            state = self.env.reset()

            while not done:
                state, reward, done, info = self.env.step(self.q[np.argmax(state)])
                total_reward += reward

                if to_render:
                    self.env.render()

            if to_print:
                print(f"Episode {i}, Reward {total_reward}")
