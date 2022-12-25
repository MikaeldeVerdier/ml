import json
import numpy as np
import game
import config
import files
from nn import NeuralNetwork

class User():
    def __init__(self):
        pass

    def get_name(self):
        return ("You", "are")

    def play_turn(self, action, tau):
        moves = game.get_legal_moves(self.mcts)
        legal_moves = [move % game.MOVE_AMOUNT + 1 for move in moves]
        user_move = None
        while user_move not in legal_moves:
            print(f"Legal moves for you are: {legal_moves}")
            user_move = int(input("Make your move: "))
        action = [move for move in moves if move % 7 + 1 == user_move][0]
        self.mcts = self.mcts.update_root(action)

        self.print_move(self.mcts, action)

        return action, None

    @staticmethod
    def print_move(root, action):
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.print_board(root.s)}\n")
        print(f"Drawn card is: {game.format_card(root.drawn_card)}")
        print(f"Amount of cards left is now: {len(root.deck)}")


class Agent():
    def __init__(self, load, version=None, name=None, to_weights=False):
        self.main_nn = NeuralNetwork(load, version, to_weights=to_weights)
        self.target_nn = NeuralNetwork(load, version, to_weights=to_weights)
        self.name = name
        
        self.outcomes = {"average": 0, "length": 0}

    def get_name(self):
        return f"Version {self.main_nn.version}" if not self.name else self.name

    def play_turn(self, history, epsilon):
        game_states = game.generate_game_states(history, len(history) - 1)
        probs = self.main_nn.get_preds(game_states)

        action = self.choose_action(probs, epsilon)

        self.mcts = self.mcts.update_root(action)

        return action

    def choose_action(self, pi, epsilon):
        if epsilon is None:
            epsilon = config.EPSILON[0] - config.EPSILON_STEP_SIZE * self.main_nn.version if self.main_nn.version < config.EPSILON[2] else config.EPSILON[1]
        action = np.random.randint(len(pi)) if np.random.rand() <= epsilon else np.argmax(pi)

        return action

    def calculate_target(self, data, t):
        game_states = game.generate_game_states(data, t, "next_state")
        return data[t]["reward"] + config.GAMMA * np.max(self.target_nn.get_preds(game_states))

    def change_version(self):
        self.main_nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
        self.main_nn.version += 1
        self.main_nn.save_model()
        self.main_nn.save_metrics()
        self.outcomes = {"average": 0, "length": 0}

    @staticmethod
    def print_move(root, pi, action, nn_value):
        game.print_values(pi)
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.print_board(root.s)}")
        print(f"NN percieved value is: {nn_value:.3f} ({(nn_value * 50):.3f})")
        print(f"Drawn card is: {game.format_card(root.drawn_card)}")
        print(f"Amount of cards left is now: {len(root.deck)}")
