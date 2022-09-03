import numpy as np
import json
import matplotlib.pyplot as plt
import config
import game
import files
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
# from keras.utils.vis_utils import plot_model

try:
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
    def __init__(self, load, version):
        self.version = version

        if load:
            self.load_version(version)
            print(f"NN loaded with version: {version}")
            return

        main_input = Input(shape=game.NN_INPUT_DIMENSIONS, name="main_input")

        x = self.convolutional_layer(main_input, config.CONVOLUTIONAL_LAYER["filter_amount"], config.CONVOLUTIONAL_LAYER["kernel_size"])
        for _ in range(config.RESIDUAL_LAYER["amount"]): x = self.residual_layer(x, config.RESIDUAL_LAYER["filter_amount"], config.RESIDUAL_LAYER["kernel_size"])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        self.model = Model(inputs=[main_input], outputs=[vh, ph])
        self.model.compile(loss={"value_head": "mean_absolute_error", "policy_head": self.softmax_cross_entropy_with_logits}, optimizer=SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM), loss_weights={"value_head": 0.5, "policy_head": 0.5}, metrics="accuracy")
        
        try:
            pass
            # plot_model(self.model, to_file=f"{config.SAVE_PATH}model.png", show_shapes=True, show_layer_names=True)
        except ImportError:
            print("You need to download pydot and graphviz to plot model.")

        # self.model.summary()

    def load_version(self, version):
        self.model = load_model(f"{config.SAVE_PATH}/training/v.{version}", custom_objects={"softmax_cross_entropy_with_logits": self.softmax_cross_entropy_with_logits})

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

    @staticmethod
    def convolutional_layer(x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        return (x)

    def residual_layer(self, input_block, filters, kernel_size):
        x = self.convolutional_layer(input_block, filters, kernel_size)
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return (x)

    @staticmethod
    def value_head(x):
        x = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(config.DENSE_VALUE_HEAD, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="value_head")(x)
        return (x)

    @staticmethod
    def policy_head(x):
        x = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(game.MOVE_AMOUNT, use_bias=config.USE_BIAS, activation="sigmoid", kernel_regularizer=regularizers.l2(config.REG_CONST), name="policy_head")(x)
        return (x)

    @cache
    def get_preds(self, node):
        data = np.expand_dims(game.generate_tutorial_game_state(node), axis=0)
        (v, p) = self.model.predict(data)

        logits = p[0]

        mask = np.full(logits.shape, True)
        legal_moves = game.get_legal_moves(node)
        mask[legal_moves] = False

        if max(logits) > 85: logits *= 85 / max(logits)
        logits[mask] = -100

        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return (v[0][0], probs)


class CurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, load, version):
        loaded = files.load_file("save.json")["current_agent"]
        self.metrics = loaded["metrics"]
        self.iterations = loaded["iterations"]
        if version is None: version = loaded["version"]

        super().__init__(load, version)
        self.model.save(f"{config.SAVE_PATH}/training/v.{version}")

    def train(self, x, y):
        self.get_preds.cache_clear()

        fit = self.model.fit(x, y, batch_size=32, epochs=config.EPOCHS, verbose=1, validation_split=config.VALIDATION_SPLIT)
        for metric in fit.history:
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.EPOCHS)]

    def plot_metrics(self, iteration_lines, derivative_lines):
        _, axs = plt.subplots(4, sharey="row", figsize=(20, 15))
        plt.xlabel("Training Iteration")

        for metric in self.metrics:
            data = self.metrics[metric]
            if data:
                ax_index = (2, 3) if "val_" in metric else (0, 1)
                ax_index = ax_index[0 if "loss" in metric else 1]
                ax = axs[ax_index]

                ax.plot(data, label=metric)
                ax.axhline(data[-1], color="black", linestyle=":")

                if derivative_lines:
                    deriv = (data[-1] - data[0]) / len(data)
                    y = [deriv * x + data[0] for x in range(len(data))]
                    ax.plot(y, color="black", linestyle="-.")

        for ax_index, metric in enumerate(["Loss", "Accuracy", "Validation Loss", "Validation Accuracy"]):
            ax = axs[ax_index]
            ax.set_title(metric)
            ax.set_ylabel(metric)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            if iteration_lines:
                iterations = self.iterations
                [ax.axvline(np.sum(iterations[:i2 + 1]) - 1, color="black", linestyle=":") for i2 in range(len(iterations))]

        plt.savefig(f"{config.SAVE_PATH}metrics.png", dpi=300)
        plt.pause(0.1)
        plt.close("all")

    def save_to_file(self, agent_kind):
        loaded = files.load_file("save.json")
        loaded[agent_kind]["version"] = self.version
        loaded[agent_kind]["iterations"] = self.iterations
        loaded[agent_kind]["metrics"] = self.metrics

        files.write("save.json", json.dumps(loaded))


class BestNeuralNetwork(NeuralNetwork):
    def __init__(self, load, version):
        print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        if version is None:
            loaded = files.load_file("save.json")["best_agent" if load else "current_agent"]
            version = loaded["version"]

        super().__init__(True, version)

    def __hash__(self):
        return hash(self.version)

    def copy_weights(self, agent_nn):
        self.get_preds.cache_clear()
        
        self.version = agent_nn.version
        self.load_version(self.version)
