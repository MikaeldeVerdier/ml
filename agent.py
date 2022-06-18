import numpy as np
import game
from nn import NeuralNetwork
from node import Node

class Agent():
    def __init__(self, load, name):
        self.nn = NeuralNetwork(load, name)

        self.reset_mcts()

    def reset_mcts(self):
        empty_position = np.zeros(np.prod(game.game_dimensions))
        self.mcts = Node(empty_position[::], None, None, None, self.nn, None)
