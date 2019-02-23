import os
import neat
import gym
import pickle
import multiprocessing as mp
import visualize

gym.logger.set_level(40)  # Disable gym warnings
os.chdir("./checkpoints")  # To store the checkpoints in this folder

# Learning Parameters
NUM_GENERATIONS = 1000
PARALLEL = 2  # Number of environments to run at once
ENV = "MsPacman-ram-v0"  # RAM means number of inputs 128


class Train:
    def __init__(self, generations, parallel=2):
        self.generations = generations
        self.par = parallel

    @staticmethod
    def _fitness_func(index, genome, config):
        env = gym.make(ENV)

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
                # env.render()  # Uncomment this if you want the game to show when training

            fitness = total_reward
            genome.fitness = fitness

            if index % 30 == 0:
                print(f"Genome {index}. Fitness {total_reward}")

            if fitness >= 500:
                pickle.dump(genome, open("finisher.pkl", "wb"))  # Save a good model just in case of a crash

            env.close()

        # To easily stop the training
        except KeyboardInterrupt:
            env.close()
            exit()

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.par):
            processes = [mp.Process(target=self._fitness_func, args=(i, genome, config)) for genome in
                         genomes[i:i + self.par]]  # Define all the processes

            # Run the processes
            [p.start() for p in processes]
            [p.join() for p in processes]

    def _run(self, config_file, generations):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self._eval_genomes, generations)

        pickle.dump(winner, open('winner.pkl', 'wb'))

        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)


if __name__ == "__main__":
    t = Train(NUM_GENERATIONS)
    t.main()
