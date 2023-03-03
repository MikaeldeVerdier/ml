import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

import environment
import config
import files

try:
	matplotlib.use("Agg")
	matplotlib.rcParams["agg.path.chunksize"] = 10000
	from functools import cache
	raise ImportError  # functools cache is really slow for some reason
except ImportError:
	def cache(f):
		cache = {}

		def caching(*args):
			cache_ref = hash(args)
			if cache_ref in cache:
				return cache[cache_ref]
			v = f(*args)
			cache[cache_ref] = v
			return v

		caching.cache_clear = cache.clear

		return caching

class NeuralNetwork:
	def __init__(self, load):
		self.version = load

		if load:
			if self.load_dir(load):
				print(f"NN loaded with version called: {load}")
				return

		position_input = Input(shape=environment.NN_INPUT_DIMENSIONS[0], name="position_input")
		position = self.position_cnn(position_input)

		deck_input = Input(shape=environment.NN_INPUT_DIMENSIONS[1], name="deck_input")
		deck = self.deck_mlp(deck_input)

		drawn_card_input = Input(shape=environment.NN_INPUT_DIMENSIONS[2], name="drawn_card_input")
		drawn_card = self.drawn_card_mlp(drawn_card_input)

		x = Concatenate()([position, deck, drawn_card])

		ph = self.policy_head(x)

		self.model = Model(inputs=[position_input, deck_input, drawn_card_input], outputs=ph)
		self.model.compile(loss=self.mean_squared_error, optimizer=Adam(learning_rate=config.learning_rate(0)))
		
		if load:
			self.load_dir(load, from_weights=True)
			print(f"Weights loaded from version called: {load}")
		else:
			try:
				plot_model(self.model, to_file=files.get_path("model.png"), show_shapes=True, show_layer_names=True)
			except ImportError:
				print("You need to download pydot and graphviz to plot model.")

		print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
		self.model.summary()

	def __hash__(self):
		return hash(self.version)

	def load_dir(self, file, from_weights=False):
		path = files.get_path(f"training/{file}/checkpoint")
		if not from_weights:
			if not os.path.exists(path):
				self.model = load_model(files.get_path(f"training/{file}"), custom_objects={"mean_squared_error": self.mean_squared_error})
				return True
		else:
			self.model.load_weights(path).expect_partial()

	def save_model(self, name, to_weights):
		if not to_weights:
			self.model.save(files.get_path(f"training/{name}"))
		else:
			self.model.save_weights(files.get_path(f"training/{name}/checkpoint"))

	@staticmethod
	def mean_squared_error(y_true, y_pred):
		logits = tf.reshape(y_pred, (tf.shape(y_true)[0], -1))

		actions = tf.cast(tf.gather(y_true, tf.constant([0]), axis=1), tf.int32)
		targets = tf.gather(y_true, 1, axis=1)

		index_tensor = tf.stack([tf.range(tf.shape(actions)[0]), actions[:, 0]], axis=1)
		preds = tf.gather_nd(logits, index_tensor)
		loss = tf.reduce_mean(tf.math.square(targets - preds))

		return loss
	
	@staticmethod
	def convolutional_layer_2D(x, filters, kernel_size):
		x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", data_format="channels_last", use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		return x

	def position_cnn(self, x):
		for filter_amount in config.CONVOLUTIONAL_LAYERS_POSITION:
			x = self.convolutional_layer_2D(x, filter_amount, config.CONVOLUTIOANL_SHAPE_POSITION)

		x = Flatten()(x)
		for neuron_amount in config.DENSE_POSITION:
			x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)

		return x

	@staticmethod
	def deck_mlp(x):
		for neuron_amount in config.DENSE_DECK:
			x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)

		return x

	@staticmethod
	def drawn_card_mlp(x):
		for neuron_amount in config.DENSE_DRAWN_CARD:
			x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)

		return x

	@staticmethod
	def policy_head(x):
		for neuron_amount in config.DENSE_POLICY_HEAD:
			x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
		x = Dense(environment.MOVE_AMOUNT, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="policy_head")(x)

		return x

	@cache
	def get_preds(self, game_state):
		data = [np.expand_dims(dat, 0) for dat in game_state.generate_nn_pass()[0]]
		logits = self.model.predict_on_batch(data)[0]

		mask = np.full(logits.shape, True)
		mask[game_state.legal_moves] = False

		logits[mask] = -np.inf

		return logits


class MainNeuralNetwork(NeuralNetwork):
	def __init__(self, load):
		self.load = "main_nn" if load is True else load

		super().__init__(self.load)

		loaded = files.load_file("save.json")
		self.version = loaded["target_nn_version"]
		self.metrics = loaded["metrics"]

	def train(self, x, y):
		self.get_preds.cache_clear()

		fit = self.model.fit(x, y, batch_size=config.BATCH_SIZE[1], epochs=config.EPOCHS, verbose=1, validation_split=config.VALIDATION_SPLIT)
		for metric in fit.history:
			self.metrics[metric] += list(map(float, fit.history[metric]))

	def plot_agent(self):
		_, axs = plt.subplots(2, 2, figsize=(40, 15))

		for (i, metric), color in zip(enumerate(self.metrics), list(matplotlib.colors.BASE_COLORS.keys())):
			data = self.metrics[metric]
			if type(data) is list:
				data = dict(enumerate(data))
			if data:
				x, data = zip(*data.items())
				x = list(map(int, x))

				# n = int(np.ceil(len(data) / 50))
				# y = moving_average(data, n)

				ax = (i // 2, i % 2)
				# color = list(matplotlib.colors.BASE_COLORS.keys())[i]
				axs[ax].plot(x, data, color=color, label=f"{metric}\n(last point: {data[-1]:5f})")
				axs[ax].axhline(data[-1], color="black", linestyle=":")

				axs[ax].set_title(metric)
				xlabel_dict = {"loss": "Training iteration", "val_loss": "Training iteration", "outcomes": "Version", "average_q_value": "Game"}
				axs[ax].set_xlabel(xlabel_dict[metric])
				axs[ax].set_ylabel(metric)

				axs[ax].set_xscale("linear")
				box = axs[ax].get_position()
				axs[ax].set_position([box.x0, box.y0, box.width * 0.9, box.height])
				axs[ax].yaxis.set_tick_params(labelbottom=True)
				axs[ax].legend(loc="center left", bbox_to_anchor=(1, 0.5))

				# deriv = np.diff(data)
				# axs[ax[0] + 1, ax[1]].plot(x[1:], deriv, color=color, label=f"Derivative of {metric}")

		plt.ioff()
		plt.savefig(files.get_path(f"agent.png"), dpi=300)
		plt.close()

	def save_metrics(self):
		files.edit_key("save.json", ["main_nn_version", "metrics"], [self.version, self.metrics])


class TargetNeuralNetwork(NeuralNetwork):
	def __init__(self, load):
		self.load = "target_nn" if load is True else load

		super().__init__(self.load)

		loaded = files.load_file("save.json")
		self.version = loaded["target_nn_version"]
