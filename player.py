import numpy as np
import config
import files
from nn import NeuralNetwork
from environment import Environment
from funcs import string_to_tuple, format_card

class User():
    def __init__(self):
        self.env = Environment()

    def get_name(self):
        return "You"

    def get_action(self, *args):
        print(f"Drawn card is: {format_card(self.game_state.drawn_card)}")

        legal_moves = self.game_state.legal_moves
        moves = [(legal_move % self.env.GAME_DIMENSIONS[1] + 1, self.env.GAME_DIMENSIONS[1] - legal_move // self.env.GAME_DIMENSIONS[1]) for legal_move in legal_moves]
        user_move = None
        while user_move not in moves:
            print(f"Legal moves for you are: {moves}")
            try:
                user_move = string_to_tuple(input("Make your move: "))
            except ValueError:
                print("Please enter a valid move.")
        
        action = legal_moves[moves.index(user_move)]

        if self.env.verbose:
            self.print_action(action)

        return action

    def print_action(self, action):
        print(f"Action taken is: {action}")


class Agent():
    def __init__(self, verbose=False, load=False, name=None, trainable=False, to_weights=False):
        self.env = Environment(verbose=verbose)

        main_kind = "main_nn" if trainable else None
        if trainable:
            main_kind = "main_nn"
            self.target_nn = NeuralNetwork(self.env, load, "target_nn", to_weights)
        else:
            main_kind = None
        self.main_nn = NeuralNetwork(self.env, load, main_kind, to_weights)
        self.name = name

        self.outcomes = {"average": 0, "length": 0}

    def get_name(self):
        return f"Version {self.main_nn.version}" if not self.name else self.name

    def get_action(self, epsilon):
        probs = self.main_nn.get_preds(self.env.game_state)

        action = self.choose_action(probs, epsilon)

        if self.env.verbose:
            self.print_action(probs, action)

        return action

    def choose_action(self, pi, epsilon):
        masked = np.full_like(pi, -np.inf)
        if epsilon is None:
            epsilon = config.EPSILON[0] - config.EPSILON_STEP_SIZE * self.main_nn.version if self.main_nn.version < config.EPSILON[2] else config.EPSILON[1]
        masked[self.env.game_state.legal_moves] = pi[self.env.game_state.legal_moves]
        action = np.random.choice(self.env.game_state.legal_moves) if np.random.rand() <= epsilon else np.argmax(masked)

        return action

    def calculate_target(self, data, t):
        next_state = data[t + 1]["state"]
        return data[t]["reward"] + config.GAMMA * np.max(self.target_nn.get_preds(next_state))

    def copy_network(self):
        self.target_nn.load_dir("main_nn")
        self.target_nn.save_model("target_nn")

        files.edit_key("save.json", ["target_nn_version"], [self.main_nn.version])

    def change_version(self):
        self.main_nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
        self.main_nn.version += 1
        self.main_nn.save_model("main_nn")
        self.main_nn.save_metrics()
        self.outcomes = {"average": 0, "length": 0}

    def print_action(self, values, action):
        print(f"Action values are: {[values[-1]]}\n{np.round(values[:-1], 8).reshape(self.env.GAME_DIMENSIONS)}")
        print(f"Action taken is: {action}")
