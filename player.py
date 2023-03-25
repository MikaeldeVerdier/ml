import numpy as np
from copy import copy

import config
import files
from nn import NeuralNetwork, MainNeuralNetwork, TargetNeuralNetwork
from funcs import order_moves, format_move, get_move

class User():
	def __init__(self, name="You"):
		self.name = name

		self.trainable = False

	def get_name(self):
		return self.name

	def get_action(self, state, *_):
		legal_moves = order_moves(state.legal_moves)
		moves = [format_move(move) for move in legal_moves]
		user_move = get_move(moves)

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
			self.main_nn = NeuralNetwork(load, name)

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

	def copy_network(self):
		# self.target_nn.load_dir(self.main_nn.name)
		# self.target_nn.save_model(self.to_weights)

		self.target_nn.model = copy(self.main_nn.model)
		self.target_nn.version = self.main_nn.version

		# self.target_nn = copy(self.main_nn)  # target_nn becomes a main_nn object

		files.edit_keys("save.json", ["target_nn_version"], [self.main_nn.version])

	def change_version(self):
		self.main_nn.version += 1
		self.main_nn.model.optimizer.learning_rate.assign(config.learning_rate(self.main_nn.version))

		if not self.main_nn.version % config.SAVING_FREQUENCY:
			self.main_nn.save_model(self.to_weights)
			self.target_nn.save_model(self.to_weights)
			self.main_nn.save_metrics()
			self.main_nn.plot_agent()

			# files.copy_dir(f"training/{self.main_nn.name}", f"training/{self.target_nn.name}")

		if not self.main_nn.version % config.VERSION_OFFSET:
			self.copy_network()

		if not self.main_nn.version % config.MODEL_CHECKPOINT_FREQUENCY:
			self.main_nn.save_model(self.to_weights, f"{self.name} v.{self.main_nn.version}")
