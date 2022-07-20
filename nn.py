import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import config
import game
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

class NeuralNetwork:
    def __init__(self, load, name):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.load = load
        self.name = name
        
        self.main_input = Input(shape=config.game_dimensions + (config.depth * 2 + 1,), name="main_input")

        # x = BatchNormalization(axis=3)(main_input)
        x = self.convolutional_layer(self.main_input, config.convolutional_layer["filter_amount"], config.convolutional_layer["kernel_size"])
        for _ in range(config.residual_layer["amount"]): x = self.residual_layer(x, config.residual_layer["filter_amount"], config.residual_layer["kernel_size"])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        checkpoint_path = f"{config.save_folder}training_{name}/cp.ckpt"
        self.cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        self.model = Model(inputs=[self.main_input], outputs=[vh, ph])
        self.model.compile(loss={"value_head": "mean_squared_error", "policy_head": self.softmax_cross_entropy_with_logits}, optimizer=SGD(learning_rate=config.lr, momentum=config.momentum), loss_weights={"value_head": 0.5, "policy_head": 0.5}, metrics="accuracy")
        
        if load:
            self.model.load_weights(checkpoint_path)
        else:
            try:
                plot_model(self.model, to_file=f"{config.save_folder}model.png", show_shapes=True, show_layer_names=True)
            except ImportError:
                print("You need to download pydot and graphviz to plot model.")

        self.metrics = {}

    def softmax_cross_entropy_with_logits(self, y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(shape = tf.shape(pi), dtype = np.float32)
        where = tf.equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0) 
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

        return loss

    def convolutional_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        return (x)

    def residual_layer(self, input_block, filters, kernel_size):
        x = self.convolutional_layer(input_block, filters, kernel_size)
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding="same", use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const))(x)
        x = BatchNormalization(axis=3)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return (x)

    def value_head(self, x):
        x = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(config.dense_value_head, use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=True, activation="tanh", kernel_regularizer=regularizers.l2(config.reg_const), name="value_head")(x)
        return (x)

    def policy_head(self, x):
        x = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(np.prod(config.game_dimensions), use_bias=True, activation="linear", kernel_regularizer=regularizers.l2(config.reg_const), name="policy_head")(x)
        return (x)

    def train(self, x, y):
        fit = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.epochs, verbose=1, validation_split=config.validation_split, callbacks=[self.cp_callback])
        # print(fit.history)
        for metric in fit.history:
            if metric not in self.metrics: self.metrics[metric] = []
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.epochs)]

    def save_progress(self, best_agent = None):
        loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())
        
        if best_agent is not None: loaded["best_agent"] = best_agent
        else:
            if not self.load: loaded[f"agent_{self.name}"] = json.loads(open(f"{config.save_folder}empty_save.json", "r").read())[f"agent_{self.name}"]
            loaded[f"agent_{self.name}"]["iterations"].append(config.training_iterations * config.epochs)
            for metric in self.metrics: loaded[f"agent_{self.name}"]["metrics"][metric] += self.metrics[metric]
            self.metrics = {}
            self.load = True
        open(f"{config.save_folder}save.json", "w").write(json.dumps(loaded))


    def plot_losses(self, show_plot):
        loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())[f"agent_{self.name}"]

        fig, axs = plt.subplots(3, sharex=True, figsize=(10, 5))
        plt.xlabel("Training Iteration")

        for metric in loaded["metrics"]:
            ax = 0 if metric.find("loss") != -1 else 1
            ax = 2 if metric.find("val_") != -1 else ax
            axs[ax].plot(loaded["metrics"][metric], label=metric)
        for ax, metric in zip(axs, ["Loss", "Accuracy", "Val_"]):
            ax.set_title(f"{metric} {self.name}")
            ax.set_ylabel(metric)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor = (1, .5))
            [ax.axvline(np.sum(loaded["iterations"][:i + 1]) - 1, color="black") for i in range(len(loaded["iterations"]))]

        plt.savefig(f"{config.save_folder}plot{self.name}.png", dpi=300)
        if not show_plot: plt.close(fig)
        else: print("PLOTTED")

    def get_preds(self, node):
        data = np.expand_dims(game.generate_game_state(node), axis=0)
        (v, p) = self.model.predict(data)

        logits = p[0]

        mask = np.full(logits.shape, True)
        allowed_actions = [move for move in game.get_legal_moves(node.s) if move != -1]
        mask[allowed_actions] = False

        logits[mask] = -100

        odds = np.exp(logits, dtype="float64") # Does it still overflow?
        probs = odds / np.sum(odds)

        return (v[0][0], probs)
