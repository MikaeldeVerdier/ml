import numpy as np
import json
import config
import game
from files import *
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

try:
    from functools import cache
    raise ImportError
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
    def __init__(self, load, name, version):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.load = load
        self.name = name
        self.version = version

        self.metrics = {}
        
        self.main_input = Input(shape=game.GAME_DIMENSIONS + (config.DEPTH * 2,), name="main_input")

        x = self.convolutional_layer(self.main_input, config.CONVOLUTIONAL_LAYER["filter_amount"], config.CONVOLUTIONAL_LAYER["kernel_size"])
        for _ in range(config.RESIDUAL_LAYER["amount"]): x = self.residual_layer(x, config.RESIDUAL_LAYER["filter_amount"], config.RESIDUAL_LAYER["kernel_size"])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        self.model = Model(inputs=[self.main_input], outputs=[vh, ph])
        self.model.compile(loss={"value_head": "mean_squared_error", "policy_head": self.softmax_cross_entropy_with_logits}, optimizer=SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM), loss_weights={"value_head": 0.5, "policy_head": 0.5}, metrics="accuracy")
        
        if load:
            if version is None: self.version = load_file("save.json")[f"agent_{self.name}"]["version"]
            else: self.version = version + 1
            checkpoint_path = f"{config.SAVE_PATH}training_{self.name}/v.{self.version - 1}/cp.cpkt"
            self.model.load_weights(checkpoint_path).expect_partial()
            print(f"NN with name: {name} now loaded version: {self.version - 1}")
        else:
            if version is None: self.version = 1
            try:
                plot_model(self.model, to_file=f"{config.SAVE_PATH}model.png", show_shapes=True, show_layer_names=True)
            except ImportError:
                print("You need to download pydot and graphviz to plot model.")

        # self.model.summary()

    def __hash__(self):
        return hash((self.name, self.version - 1))

    def softmax_cross_entropy_with_logits(self, y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(shape=tf.shape(pi), dtype=np.float32)
        where = tf.equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0) 
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

        return loss

    def convolutional_layer(self, x, filters, kernel_size):
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

    def value_head(self, x):
        x = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(config.DENSE_VALUE_HEAD, use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=config.USE_BIAS, activation="tanh", kernel_regularizer=regularizers.l2(config.REG_CONST), name="value_head")(x)
        return (x)

    def policy_head(self, x):
        x = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_last", padding="same", use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(np.prod(game.GAME_DIMENSIONS), use_bias=config.USE_BIAS, activation="linear", kernel_regularizer=regularizers.l2(config.REG_CONST), name="policy_head")(x)
        return (x)

    def train(self, x, y):
        self.get_preds.cache_clear()

        checkpoint_path = f"{config.SAVE_PATH}training_{self.name}/v.{self.version}/cp.cpkt"
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        fit = self.model.fit(x, y, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1, validation_split=config.VALIDATION_SPLIT, callbacks=[cp_callback])
        for metric in fit.history:
            if metric not in self.metrics: self.metrics[metric] = []
            [self.metrics[metric].append(fit.history[metric][i]) for i in range(config.EPOCHS)]

    def save_progress(self, best_agent=None):
        loaded = load_file("save.json")
        
        if best_agent: loaded["best_agent"] = best_agent
        else:
            self.version += 1
            loaded[f"agent_{self.name}"]["version"] = self.version
            loaded[f"agent_{self.name}"]["iterations"].append(config.TRAINING_ITERATIONS * config.EPOCHS)
            for metric in self.metrics: loaded[f"agent_{self.name}"]["metrics"][metric] += self.metrics[metric]
            self.metrics = {}
            self.load = True

        write("save.json", json.dumps(loaded))

    @cache
    def get_preds(self, nodes):
        data = np.expand_dims(game.generate_tutorial_game_state(nodes, False), axis=0)
        (v, p) = self.model.predict(data)

        logits = p[0]

        # v = [[0.3]] #
        # logits = np.array(list(range(42))) #

        mask = np.full(logits.shape, True)
        node = nodes[-1]
        legal_moves = game.get_legal_moves(node.s)
        mask[legal_moves] = False

        if max(logits) > 85: logits *= 85/max(logits)
        logits[mask] = -100

        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return (v[0][0], probs)
