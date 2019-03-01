# Pacman AI
This project builds a program that can play the original 1980 Atari Pacman. 
The approaches used are Deep-Q-Learning and Neuroevolution of Augmenting Topologies

## Running The Program
Use the Makefile to run various parts of this project.
* NEAT
    * Train - `make neat-train`
    * Test - `make neat-test`
* DQN
    * Train - `make rl-train`
    * Test - `make rl-test`
* Q Learning Demo - `make q-learning`

#### Alternative
`main.py` usage
```bash
python3 main.py [algorithm] [train/test]
```

## Requirements
Install the requirements with
```bash
pip install -r requirements.txt
```
#### Alternative
```bash
sudo make install
```

## NEAT
All files using NEAT are stored under `NEAT/`

Train the NEAT model:
```bash
make neat-train
```

Test the NEAT model:
```bash
make neat-test
```

### Explanation
For an explanation on a project using the same algorithm watch [this video](https://www.youtube.com/watch?v=UdJ4titVY7I).
#### Short Explanation
This program uses a mathematical model, called a neural network, which simulates the brain of a human being. 
A neural network works by taking inputs and outputting probabilities for each of the outputs. This can be accomplished
by using a sigmoid function. <br><br>
![Sigmoid](https://qph.fs.quoracdn.net/main-qimg-07066668c05a556f1ff25040414a32b7)
<br><br>
`Neuroevolution of Augmenting Topologies`, or `NEAT` is what this project uses. The way standard
`neuroevolution` works is by randomly initializing a population of neural networks and
using survival of the fittest to get the best model. The best networks in each generations
are bred and some mutations are introduced. `NEAT` introduces features like speciation to
make a much more effective neuroevolution model. Neuroevolution is known to do better than standard
reinforcement learning models.<br>

## DQN
All files using NEAT are stored under `DQN/`

Train the NEAT model:
```bash
make neat-train
```

Test the NEAT model:
```bash
make neat-test
```

### Explanation