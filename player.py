import numpy as np
import config
import game
from nn import NeuralNetwork
from node import Node

class User():
    def __init__(self):
        self.start_node = Node(np.zeros(np.prod(config.game_dimensions))[::], None, None, None, None)
        self.root = self.start_node
        
    def play_turn(self, root, tau):
        while len(root.children) != len(game.get_legal_moves(root.s)):
            root.expand(None)
        root = root.children[int(input("Make your move: "))]

        self.print_move(root)

        return root, None

    def print_move(self, root):
        print(f"It's {[None, 'O', 'X'][root.player]}'s turn")
        print(f"Move to make is: {root.parent_action}")
        print(f"Position is now: \n {game.print_board(root.s)}")

class Agent():
    def __init__(self, load, name):
        self.nn = NeuralNetwork(load, name)
        self.start_node = Node(np.zeros(np.prod(config.game_dimensions))[::], None, None, None, None)
        self.root = self.start_node

    def play_turn(self, root, tau):
        for _ in range(config.MCTSSims):
            root.selection(self.nn)

        pi, values = self.getAV(root, tau)
        
        action, value = self.choose_action(pi, values, tau)
        root = [child for child in root.children if child.parent_action == action][0]
        nn_value = self.nn.test(game.generate_game_state(self.root))[0]

        self.print_move(root, pi, value, nn_value)

        return root, pi

    def choose_action(self, pi, values, tau):
        action = np.argmax(pi) if tau == 0 else np.where(np.random.multinomial(1, pi)==1)[0][0]
        
        value = values[action]

        return action, value

    def getAV(self, root, tau):
        pi = np.zeros(np.prod(config.game_dimensions))
        values = np.zeros(np.prod(config.game_dimensions))

        for child in root.children:
            if child.parent_action != -1:
                pi[child.parent_action] = child.n
                values[child.parent_action] = child.q

        pi /= np.sum(pi)

        return pi, values
    
    def print_move(self, root, pi, mcts_value, nn_value):
        print(f"It's {[None, 'O', 'X'][root.player]}'s turn")
        print(f"Action values are: \n {game.print_values(np.round(pi, 3))}")
        print(f"Move to make is: {root.parent_action}")
        print(f"Position is now: \n {game.print_board(root.s)}")
        print(f"MCTS percieved value is: {np.round(mcts_value, 3)}")
        print(f"NN percieved value is: {np.round(nn_value * 1000)/1000} \n")