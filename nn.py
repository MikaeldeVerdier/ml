import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import config
import game
import files
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv3D, Conv1D, Flatten, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

try:
    matplotlib.use("Agg")
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
    def __init__(self, load, kind, to_weights):
        self.to_weights = to_weights

        if kind is not None:
            if load:
                load = kind
            loaded = files.load_file("save.json")
            self.version = loaded[f"{kind}_version"]
            self.version_outcomes = loaded["version_outcomes"]
            self.iterations = loaded["iterations"]
            self.metrics = loaded["metrics"]
        else:
            self.version = load
            self.version_outcomes = {}

        if load:
            if self.load_dir(load):
                print(f"NN loaded with version called: {load}")
                return

        position_input = Input(shape=game.NN_INPUT_DIMENSIONS[0], name="position_input")
        position = self.position_cnn(position_input)

        deck_input = Input(shape=game.NN_INPUT_DIMENSIONS[1], name="deck_input")
        deck = self.deck_cnn(deck_input)

        drawn_card_input = Input(shape=game.NN_INPUT_DIMENSIONS[2], name="drawn_card_input")
        drawn_card = self.drawn_card_cnn(drawn_card_input)

        x = Concatenate()([position, deck, drawn_card])

        x = self.shared_mlp(x)

        ph = self.policy_head(x)

        self.model = Model(inputs=[position_input, deck_input, drawn_card_input], outputs=ph)
        self.model.compile(loss=self.mean_squared_error, optimizer=Adam(learning_rate=config.LEARNING_RATE))
        
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
        path = f"{config.SAVE_PATH}training/{file}/checkpoint"
        if not from_weights:
            if not os.path.exists(path):
                self.model = load_model(f"{config.SAVE_PATH}training/{file}", custom_objects={"loss": self.mean_squared_error})
                return True
        else:
            self.model.load_weights(path).expect_partial()

    def mean_squared_error(self, y_true, y_pred):
        logits = tf.reshape(y_pred, (tf.shape(y_true)[0], -1))

        actions = tf.cast(tf.gather(y_true, tf.constant([0]), axis=1), tf.int32)
        targets = tf.gather(y_true, 1, axis=1)

        index_tensor = tf.stack([tf.range(tf.shape(actions)[0]), actions[:, 0]], axis=1)
        preds = tf.gather_nd(logits, index_tensor)
        loss = tf.math.square(targets - preds)

        return loss

    def ph_mae(self, y_true, y_pred):
        logits = tf.reshape(y_pred, (tf.shape(y_true)[0], -1))

        action = tf.cast(tf.gather(y_true, tf.constant([0]), axis=1), tf.int32)
        pi_action = tf.gather(y_true, 1, axis=1)
        legal_moves = tf.gather(y_true, tf.range(3, 3 + game.MOVE_AMOUNT), axis=1)
        
        pi_new = self.softmax(logits, legal_moves)
        pi_theta = tf.gather_nd(pi_new, tf.stack([tf.range(tf.shape(action)[0]), action[:, 0]], axis=1))

        return tf.math.abs(pi_action - pi_theta)

    @staticmethod
    def softmax_cross_entropy_with_logits(y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(shape=tf.shape(pi), dtype=np.float32)
        where = tf.equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0)
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

        return loss

    def position_cnn(self, x):
        for filter_amount, kernel_size in config.CONVOLUTIONAL_LAYERS_POSITION: x = self.convolutional_layer_3D(x, filter_amount, kernel_size)  # , config.POOLING_SIZE_POSITION)
        x = Flatten()(x)
        for neuron_amount in config.DENSE_POSITION: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        
        return x

    def deck_cnn(self, x):
        for filter_amount, kernel_size in config.CONVOLUTIONAL_LAYERS_DECK: x = self.convolutional_layer_1D(x, filter_amount, kernel_size)
        x = Flatten()(x)
        for neuron_amount in config.DENSE_DECK: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        
        return x

    def drawn_card_cnn(self, x):
        for filter_amount, kernel_size in config.CONVOLUTIONAL_LAYERS_DRAWN_CARD: x = self.convolutional_layer_1D(x, filter_amount, kernel_size)
        x = Flatten()(x)
        for neuron_amount in config.DENSE_DRAWN_CARD: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        
        return x

    @staticmethod
    def convolutional_layer_3D(x, filters, kernel_size):
        x = Conv3D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        return x

    @staticmethod
    def convolutional_layer_1D(x, filters, kernel_size):
        x = Conv1D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        return x

    @staticmethod
    def shared_mlp(x):
        for neuron_amount in config.DENSE_SHARED: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        
        return x

    @staticmethod
    def policy_head(x):
        for neuron_amount in config.DENSE_POLICY_HEAD: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = Dense(game.MOVE_AMOUNT, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="policy_head")(x)
        
        return x

    @cache
    def get_preds(self, history):
        data = [np.expand_dims(dat, 0) for dat in game.generate_nn_pass(history)[0]]
        logits = self.model.predict_on_batch(data)[0]

        mask = np.full(logits.shape, True)
        mask[history[-1].legal_moves] = False

        logits[mask] = 0

        return logits

    def train(self, x, y):
        self.get_preds.cache_clear()

        fit = self.model.fit(x, y, batch_size=config.BATCH_SIZE[1], epochs=config.EPOCHS, verbose=1, validation_split=config.VALIDATION_SPLIT)
        for metric in fit.history:
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.EPOCHS)]

    def save_model(self, kind):
        if not self.to_weights:
            self.model.save(f"{config.SAVE_PATH}training/{kind}")
        else:
            self.model.save_weights(f"{config.SAVE_PATH}training/{kind}/checkpoint")

    def plot_agent(self, iteration_lines=False):
        _, axs = plt.subplots(4, figsize=(20, 20))
        for metric in self.metrics:
            data = self.metrics[metric]
            if data:
                axs[0].plot(data, label=f"{metric}\n(last point: {data[-1]:5f})")
                axs[0].axhline(data[-1], color="black", linestyle=":")

                x = list(range(1, len(data)))
                deriv = np.diff(data)
                axs[1].plot(x, deriv, label=f"Derivative of {metric}")
        
        data = self.version_outcomes
        if data:
            x = list(map(int, data.keys()))
            data = [value["average"] for value in data.values()]
            axs[2].plot(x, data, label=f"Outcome\n(last point: {data[-1]:5f})")

            deriv = np.diff(data)
            axs[3].plot(x[1:], deriv, label=f"Derivative of outcome")

        for ax_index, axis in enumerate(["Loss", "Loss derivatives", "Outcome", "Outcome derivative"]):
            ax = axs[ax_index]
            ax.set_title(axis)
            ax.set_xlabel("Training iteration" if "Loss" in axis else "Version")
            ax.set_ylabel(axis)
            ax.set_xscale("linear")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            if iteration_lines:
                iterations = self.iterations
                [ax.axvline(np.sum(iterations[:i2 + 1]) - 1, color="black", linestyle=":") for i2 in range(len(iterations))]

        plt.ioff()
        plt.savefig(f"{config.SAVE_PATH}agent.png", dpi=300)
        plt.close()

    def register_result(self, result):
        if not self.version in self.version_outcomes:
            self.version_outcomes[self.version] = {"length": 1, "average": result}
            return
        
        self.version_outcomes[self.version]["length"] += 1
        self.version_outcomes[self.version]["average"] = (self.version_outcomes[self.version]["average"] * (self.version_outcomes[self.version]["length"] - 1) + result) / self.version_outcomes[self.version]["length"]

    def save_outcomes(self):
        loaded = files.load_file("save.json")
        loaded["version_outcomes"] = self.version_outcomes

        files.write("save.json", json.dumps(loaded))

    def save_metrics(self):
        loaded = files.load_file("save.json")
        loaded[f"main_nn_version"] = self.version
        loaded["iterations"] = self.iterations
        loaded["metrics"] = self.metrics

        files.write("save.json", json.dumps(loaded))
