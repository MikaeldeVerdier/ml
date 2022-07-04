import numpy as np
import config
import game
from nn import NeuralNetwork
from node import Node

class User():
    def __init__(self):
        pass

    def play_turn(self, action, tau):
        if action is not None: self.mcts = self.mcts.update_root(action)
        action = int(input("Make your move: "))
        self.mcts = Node(game.move(self.mcts.s.copy(), action, self.mcts.player), self, action, -self.mcts.player, 0)

        self.print_move(self.mcts)

        return action, None

    def print_move(self, root):
        player_dict = {1: "X", -1: "O"}
        print(f"It's {player_dict[root.player]}'s turn")
        print(f"Move to make is: {root.parent_action}")
        print(f"Position is now: \n {game.print_board(root.s)}")

class Agent():
    def __init__(self, load, name):
        self.nn = NeuralNetwork(load, name)

    def play_turn(self, action, tau):
        if action is not None: self.mcts = self.mcts.update_root(action)
        
        for _ in range(config.MCTSSims):
            self.mcts.simulate(self.nn)

        pi, values = self.getAV(self.mcts, tau)
        
        action, value = self.choose_action(pi, values, tau)
        
        self.mcts = [child for child in self.mcts.children if child.parent_action == action][0]
        nn_value = self.nn.test(game.generate_game_state(self.mcts))[0]

        self.print_move(self.mcts, pi, value, nn_value)

        return action, pi

    def choose_action(self, pi, values, tau):
        action = np.argmax(pi) if tau == 1e-2 else np.where(np.random.multinomial(1, pi) == 1)[0][0]
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
        player_dict = {1: "X", -1: "O"}
        print(f"It's {player_dict[root.player]}'s turn")
        print(f"Action values are: \n {game.print_values(np.round(pi, 3))}")
        print(f"Move to make is: {root.parent_action}")
        print(f"Position is now: \n {game.print_board(root.s)}")
        print(f"MCTS percieved value is: {np.round(mcts_value, 3)}")
        print(f"NN percieved value is: {np.round(nn_value * 1000)/1000} \n")