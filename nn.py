import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import config
import game
import files
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

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

        position_input = Input(shape=game.NN_INPUT_DIMENSIONS[0], name="position_input")
        position = self.position_cnn(position_input)

        deck_input = Input(shape=game.NN_INPUT_DIMENSIONS[1], name="deck_input")
        deck = self.deck_cnn(deck_input)

        drawn_card_input = Input(shape=game.NN_INPUT_DIMENSIONS[2], name="drawn_card_input")
        drawn_card = self.drawn_card_cnn(drawn_card_input)

        x = Concatenate()([position, deck, drawn_card])

        x = self.shared_mlp(x)

        vh = self.value_head(x)
        ph = self.policy_head(x)

        self.model = Model(inputs=[position_input, deck_input, drawn_card_input], outputs=[vh, ph])
        self.model.compile(loss={"value_head": self.J_vf, "policy_head": self.J_clip}, optimizer=SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM), loss_weights = {"value_head": 0.5, "policy_head": 0.5}, metrics="accuracy")
        
        try:
            # pass
            plot_model(self.model, to_file=f"{config.SAVE_PATH}model.png", show_shapes=True, show_layer_names=True)
        except ImportError:
            print("You need to download pydot and graphviz to plot model.")

        # self.model.summary()

    def __hash__(self):
        return hash(self.version)

    def load_version(self, version):
        self.model = load_model(f"{config.SAVE_PATH}training/v.{version}", custom_objects={"J_vf": self.J_vf, "J_clip": self.J_clip})

    def J_vf(self, y_true, y_pred):
        r = y_true[0][0]
        index = tf.cast(y_true[0][1], tf.int32)

        posses = np.load(files.get_path("positions.npy"), allow_pickle=True)

        bs = [[], [], []]
        for pos in posses:
            for i, b in enumerate(bs):
                bs[i].append(pos[0][i])

        for i, b in enumerate(bs):
            bs[i] = tf.convert_to_tensor(np.array(b))

        data = [tf.expand_dims(b[index], 0) for b in bs]
        v = self.model(data)[0][0][0]

        V_targ = r + config.GAMMA * v
        J_vf = (y_pred - V_targ) ** 2

        return J_vf

    @staticmethod
    def J_clip(y_true, y_pred):
        y_true = y_true[0]
        a = tf.cast(y_true[0], tf.int32)
        pi_a = y_true[1]
        Â = y_true[2]

        pi_theta = y_pred[0][a]
        pi_theta_old = pi_a
        r_theta = pi_theta / pi_theta_old

        L_cpi = r_theta * Â
        # L_clip = min(r_theta, 1 + config.EPSILON if Â > 0 else 1 - config.EPSILON)
        # J_clip = min(L_cpi, L_clip)
        J_clip = L_cpi

        return -J_clip

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
        for filter_amount, kernel_size in config.CONVOLUTIONAL_LAYERS_POSITION: x = self.convolutional_layer_2D(x, filter_amount, kernel_size)
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
    def convolutional_layer_2D(x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)
        return x

    @staticmethod
    def convolutional_layer_1D(x, filters, kernel_size):
        x = Conv1D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=2)(x)
        x = ReLU()(x)
        return x

    @staticmethod
    def shared_mlp(x):
        for neuron_amount in config.DENSE_SHARED: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        return x

    @staticmethod
    def value_head(x):
        for neuron_amount in config.DENSE_VALUE_HEAD: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = Dense(1, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="value_head")(x)
        return x

    @staticmethod
    def policy_head(x):
        for neuron_amount in config.DENSE_POLICY_HEAD: x = Dense(neuron_amount, use_bias=config.USE_BIAS, activation="relu", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = Dense(game.MOVE_AMOUNT, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="policy_head")(x)
        return x

    @cache
    def get_preds(self, history):
        data = [np.expand_dims(dat, 0) for dat in game.generate_nn_pass(history)[0]]
        (v, p) = self.model.predict(data)

        value = v[0][0]
        logits = p[0]

        mask = np.full(logits.shape, True)
        legal_moves = game.get_legal_moves(history[-1])
        mask[legal_moves] = False

        if max(logits) > 85: logits *= 85 / max(logits)
        logits[mask] = -100

        odds = np.exp(logits).astype(np.float64)
        probs = odds / np.sum(odds)

        return (value, probs, logits)


class CurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, load, version):
        loaded = files.load_file("save.json")["current_agent"]
        self.metrics = loaded["metrics"]
        self.iterations = loaded["iterations"]
        if version is None: version = loaded["version"]

        super().__init__(load, version)
        self.model.save(f"{config.SAVE_PATH}training/v.{version}")

    def train(self, x, y):
        self.get_preds.cache_clear()

        fit = self.model.fit(x, y, batch_size=32, epochs=config.EPOCHS, verbose=1, validation_split=config.VALIDATION_SPLIT)
        for metric in fit.history:
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.EPOCHS)]

    def plot_metrics(self, iteration_lines=False, derivative_lines=False):
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
                    deriv = (data[-1] - data[0]) / (len(data) - 1)  # np.mean(np.diff(data)) 
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
    
    def plot_outcomes(self, derivative_line=False):
        loaded = files.load_file("save.json")
        data = loaded["current_agent"]["version_outcomes"] 

        x = list(map(int, data.keys()))
        data = list(data.values())
        plt.plot(x, data)
        plt.xlabel("Version")
        plt.ylabel("Average outcome")
        plt.title("Average outcome for versions")

        if derivative_line:
            deriv = (data[-1] - data[0]) / x[-1]
            y = [deriv * x + data[0] for x in range(x[-1])]
            plt.plot(y, color="black", linestyle="-.")
        
        plt.savefig(f"{config.SAVE_PATH}outcomes.png", dpi=300)
        plt.pause(0.1)
        plt.close("all")

    def save_metrics(self, agent_kind):
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
