install:
	@sudo pip3 install -r requirements.txt

rl-train:
	@python3 DQN/train.py

rl-test:
	@python3 DQN/test.py

neat-train:
	@python3 NEAT/train.py

neat-test:
	@python3 NEAT/test.py
