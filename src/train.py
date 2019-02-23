import gym
import tensorflow as tf
import numpy as np
import argparse

ENV = "MsPacman-ram-v0"  # MsPacman-ram gives the ram as input - 128 inputs

# Learning Arguments
NUM_EPISODES = 2000
GAMMA = 0.95
EPSILON = 0.1
MEMORY_SIZE = 1000
BATCH_SIZE = 64
INPUTS = 128
RANDOM_THRESHOLD = 1000  # Threshold until which it will select a random action with probability epsilon


class Network:
    def __init__(self):
        pass

    def create_layer(self, inputs, outputs, activation=False):
        pass
