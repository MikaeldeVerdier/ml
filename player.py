import numpy as np
import game
import config
from nn import NeuralNetwork

class User():
    def __init__(self):
        pass

    def get_name(self):
        return "You"

    @staticmethod
    def string_to_tuple(s):
        a = s.replace(" ", "").replace("(", "").replace(")", "")
        b = a.split(',')
        res = tuple(int(el) for el in b)

        return res

    def play_turn(self, *args):
        print(f"Drawn card is: {game.format_card(self.game_state.drawn_card)}")

        legal_moves = self.game_state.legal_moves
        moves = [(legal_move % game.GAME_DIMENSIONS[1] + 1, game.GAME_DIMENSIONS[1] - legal_move // game.GAME_DIMENSIONS[1]) for legal_move in legal_moves]
        user_move = None
        while user_move not in moves:
            print(f"Legal moves for you are: {moves}")
            try:
                user_move = self.string_to_tuple(input("Make your move: "))
            except ValueError:
                print("Please enter a valid move.")
        
        action = legal_moves[moves.index(user_move)]

        self.game_state = self.game_state.update_root(action)

        self.print_move(self.game_state, action)

        return action

    @staticmethod
    def print_move(root, action):
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.shape_board(root.s)}\n")
        print(f"Amount of cards left is now: {len(root.deck)}")


class Agent():
    def __init__(self, load=False, name=None, trainable=False, to_weights=False):
        main_kind = "main_nn" if trainable else None
        if trainable:
            main_kind = "main_nn"
            self.target_nn = NeuralNetwork(load, kind="target_nn", to_weights=to_weights)
        else:
            main_kind = None
        self.main_nn = NeuralNetwork(load, main_kind, to_weights)
        self.name = name
        
        self.outcomes = {"average": 0, "length": 0}

    def get_name(self):
        return f"Version {self.main_nn.version}" if not self.name else self.name

    def play_turn(self, history, epsilon):
        game_states = game.generate_game_states(history, len(history) - 1)
        probs = self.main_nn.get_preds(game_states)

        action = self.choose_action(probs, epsilon)

        self.game_state = self.game_state.update_root(action)

        # self.print_move(self.game_state, probs, action)

        return action

    def choose_action(self, pi, epsilon):
        if epsilon is None:
            epsilon = config.EPSILON[0] - config.EPSILON_STEP_SIZE * self.main_nn.version if self.main_nn.version < config.EPSILON[2] else config.EPSILON[1]
        action = np.random.choice(self.game_state.legal_moves) if np.random.rand() <= epsilon else np.argmax(pi)

        return action

    def calculate_target(self, data, t):
        game_states = game.generate_game_states(data, t, "next_state")
        return data[t]["reward"] + config.GAMMA * np.max(self.target_nn.get_preds(game_states))

    def copy_network(self):
        self.target_nn.load_dir("main_nn")
        self.target_nn.save_model("target_nn")

    def change_version(self):
        self.main_nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
        self.main_nn.version += 1
        self.main_nn.save_model("main_nn")
        self.main_nn.save_metrics()
        self.outcomes = {"average": 0, "length": 0}

    @staticmethod
    def print_move(game_state, pi, action):
        game.print_values(pi)
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.shape_board(game_state.s)}")
        print(f"Drawn card is: {game.format_card(game_state.drawn_card)}")
        print(f"Amount of cards left is now: {len(game_state.deck)}")
