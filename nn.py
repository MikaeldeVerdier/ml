import json
import numpy as np
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

import time

class NeuralNetwork:
    def __init__(self, load, name):
        self.load = load
        self.name = name
        
        self.main_input = Input(shape=game.game_dimensions + (config.depth * 2 + 1,), name="main_input")

        # x = BatchNormalization(axis=3)(main_input)
        x = self.convolutional_layer(self.main_input, config.convolutional_layer["filter_amount"], config.convolutional_layer["kernel_size"])
        for _ in range(config.residual_layer["amount"]): x = self.residual_layer(x, config.residual_layer["filter_amount"], config.residual_layer["kernel_size"])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        checkpoint_path = f"training_{name}/cp.ckpt"
        self.cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        self.model = Model(inputs=[self.main_input], outputs=[vh, ph])
        self.model.compile(loss={"value_head": "mean_squared_error", "policy_head": self.softmax_cross_entropy_with_logits}, optimizer=SGD(learning_rate=config.lr, momentum=config.momentum), loss_weights={"value_head": 0.5, "policy_head": 0.5}, metrics="accuracy")
        
        if load:
            self.model.load_weights(checkpoint_path)
        else:
            try:
                plot_model(self.model, to_file="model.png", show_shapes=True, show_layer_names=True)
            except ImportError:
                print("You need to download pydot and graphviz to plot model.")

        self.metrics = {}

    def softmax_cross_entropy_with_logits(self, y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
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
        # x = x.reshape((6, 7))
        return (x)

    def train(self, x, y):
        fit = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.epochs, verbose=1, validation_split=config.validation_split, callbacks=[self.cp_callback])
        print(fit.history)
        for metric in fit.history:
            if metric not in self.metrics: self.metrics[metric] = []
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.epochs)]

    def save_progress(self, best_agent):
        fi = "save" if self.load else "empty_save"
        file = open(f"{fi}.json", "r")
        loaded = json.loads(file.read())
        loaded["best_agent"] = best_agent
        loaded[f"agent_{self.name}"]["iterations"].append(config.training_iterations * config.epochs)
        for i in self.metrics: loaded[f"agent_{self.name}"]["metrics"][i] += (self.metrics[i])
        file.close()
        file = open("save.json", "w")
        file.write(json.dumps(loaded))
        file.close()

    def plot_losses(self, plot):
        file = open("save.json", "r")
        loaded = json.loads(file.read())[f"agent_{self.name}"]
        self.metrics = loaded["metrics"]
        saves = loaded["iterations"]

        fig, axs = plt.subplots(3, sharex=True)
        plt.xlabel("Training Iteration")

        for metric in self.metrics:
            ax = 0 if metric.find("loss") != -1 else 1
            ax = 2 if metric.find("val_") != -1 else ax
            axs[ax].plot(self.metrics[metric], label=metric)
        for ax, metric in zip(axs, ["Loss", "Accuracy", "Val_"]):
            ax.set_title(f"{metric} {self.name}")
            ax.set_ylabel(metric)
            ax.legend(loc="best")
            [ax.axvline(np.sum(saves[:i + 1]) - 1, color="black") for i in range(len(saves))]

        plt.savefig("first_iteration_retraining.png", dpi=300)
        if not plot: plt.close(fig)
        else: print("PLOTTED")

    def test(self, data):
        data = np.expand_dims(data, axis=0)
        (v, p) = self.model.predict(data)

        return (v[0][0], p[0])
