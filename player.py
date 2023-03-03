import os
import numpy as np
from copy import copy
from shutil import rmtree, copytree

import environment
import config
import files
from nn import NeuralNetwork, MainNeuralNetwork, TargetNeuralNetwork
from funcs import string_to_tuple, format_card

class User():
	def __init__(self, name="You"):
		self.name = name

		self.trainable = False

	def get_name(self):
		return self.name

	def get_action(self, state, *args):
		print(f"Drawn card is: {format_card(state.drawn_card)}")

		legal_moves = state.legal_moves
		if len(environment.GAME_DIMENSIONS) == 2:
			moves = [(legal_move % environment.GAME_DIMENSIONS[1] + 1, environment.GAME_DIMENSIONS[1] - legal_move // environment.GAME_DIMENSIONS[1]) if legal_move != 0 else 0 for legal_move in legal_moves]
			func = string_to_tuple
		else:
			moves = legal_moves
			func = int

		user_move = None
		while user_move not in moves:
			print(f"Legal moves for you are: {moves}")
			try:
				user_move = func(input("Make your move: "))
			except ValueError:
				print("Please enter a valid move.")

		action = legal_moves[moves.index(user_move)]

		return None, action

class Agent():
	def __init__(self, load=False, name=None, trainable=False, to_weights=False):
		self.name = name
		self.trainable = trainable
		self.to_weights = to_weights

		if trainable:
			self.target_nn = TargetNeuralNetwork(load)
			self.main_nn = MainNeuralNetwork(load)
		else:
			self.main_nn = NeuralNetwork(load)

	def get_name(self):
		return self.name or f"Version {self.main_nn.version}"

	def get_action(self, state, epsilon):
		probs = self.main_nn.get_preds(state)

		action = self.choose_action(state, probs, epsilon)

		return probs, action

	def choose_action(self, state, pi, epsilon):
		if epsilon is None:
			epsilon = config.epsilon(self.main_nn.version)

		action = np.random.choice(state.legal_moves) if np.random.rand() <= epsilon else np.argmax(pi)

		return action

	def calculate_target(self, data, t):
		next_state = data[t + 1]["state"]
		return data[t]["reward"] + config.GAMMA * np.max(self.target_nn.get_preds(next_state))

	def copy_network(self):
		# self.target_nn.load_dir("main_nn")
		# self.target_nn.save_model("target_nn", self.to_weights)

		self.target_nn = copy(self.main_nn)

		files.edit_key("save.json", ["target_nn_version"], [self.main_nn.version])

	def change_version(self):
		self.main_nn.version += 1
		self.main_nn.model.optimizer.learning_rate.assign(config.learning_rate(self.main_nn.version))

		if not self.main_nn.version % config.SAVING_FREQUENCY:
			self.main_nn.save_model("main_nn", self.to_weights)
			self.main_nn.save_metrics()
			self.main_nn.plot_agent()

			if os.path.exists(files.get_path("training/target_nn")):
				rmtree(files.get_path("training/target_nn"))
			copytree(files.get_path("training/main_nn"), files.get_path("training/target_nn"))

		if not (self.main_nn.version - 1) % config.VERSION_OFFSET:
			self.copy_network()
