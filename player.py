import numpy as np

import config
import files
from nn import NeuralNetwork, MainNeuralNetwork, TargetNeuralNetwork
from funcs import order_moves, format_move, get_move

class User():
	def __init__(self, name="You"):
		self.name = name

		self.trainable = False

	@property
	def full_name(self):
		return self.name

	def get_action(self, state, *_):
		formatted_moves = [format_move(move) for move in state.legal_moves]
		moves = order_moves(formatted_moves)
		user_move = get_move(moves)

		action = state.legal_moves[formatted_moves.index(user_move)]

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

	@property
	def full_name(self):
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
	
	def save_metrics(self):
		files.edit_keys("save.json", ["main_nn_version", "target_nn_version", "metrics"], [self.main_nn.version, self.target_nn.version, self.main_nn.metrics])
	
	def save_progress(self):
		self.main_nn.save_model(self.to_weights)
		self.target_nn.save_model(self.to_weights)
		self.save_metrics()
		self.main_nn.plot_agent()

	def copy_network(self):
		# self.target_nn.load_dir(self.main_nn.name)
		# self.target_nn.save_model(self.to_weights)

		self.target_nn.copy_model(self.main_nn.model)

		# self.target_nn = copy(self.main_nn)  # target_nn becomes a main_nn object

	def save_checkpoint(self):
		self.main_nn.save_model(self.to_weights, f"{self.name} v.{self.main_nn.version}")

	def change_version(self):
		self.main_nn.version += 1
		self.main_nn.model.optimizer.learning_rate.assign(config.learning_rate(self.main_nn.version))

		if not self.main_nn.version % config.SAVING_FREQUENCY:
			self.save_progress()

		if not self.main_nn.version % config.VERSION_OFFSET:
			self.copy_network()

		if not self.main_nn.version % config.MODEL_CHECKPOINT_FREQUENCY:
			self.save_checkpoint()
