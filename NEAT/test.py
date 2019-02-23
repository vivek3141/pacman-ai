import os
import neat
import gym
import pickle
import multiprocessing as mp
import visualize
import time

gym.logger.set_level(40)  # Disable gym warnings
os.chdir("./checkpoints")  # To store the checkpoints in this folder

# Learning Parameters
NUM_GENERATIONS = 1000
PARALLEL = 2  # Number of environments to run at once
ENV = "MsPacman-ram-v0"  # RAM means number of inputs 128
CONFIG_FILE = "../config"

env = gym.make(ENV)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     CONFIG_FILE)

genome = pickle.load(open("winner.pkl", "rb"))
fitness = 0
while fitness < 2000:
    try:
        state = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        done = False
        total_reward = 0

        while not done:
            # Pass input through neural network
            state = state.flatten()
            output = net.activate(state)
            action = output.index(max(output))

            observation, reward, done, info = env.step(action)
            state = observation
            total_reward += reward
            env.render()  # Uncomment this if you want the game to show when training

            time.sleep(0.1)

        fitness = total_reward
        print(fitness)

        # if index % 30 == 0:
        #    print(f"Genome {index}. Fitness {total_reward}")

        if fitness >= 500:
            pickle.dump(genome, open("finisher.pkl", "wb"))  # Save a good model just in case of a crash

        env.close()

    # To easily stop the training
    except KeyboardInterrupt:
        env.close()
        exit()
