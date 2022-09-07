from tkinter import W
import numpy as np
import game
import config


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
        print(f"Deck length is now: {len(root.deck)}")


class Agent():
    def __init__(self, nn_class, load, version=None, name=None):
        self.nn = nn_class(load, version)
        self.name = name
        
        self.outcomes = {"average": 0, "length": 0}

    def get_name(self):
        return (f"Version {self.nn.version}" if not self.name else self.name, "is")

    def play_turn(self, tau):
        legal_moves = game.get_legal_moves(self.mcts)
        if len(legal_moves) == 1:
            action = legal_moves[0]
            pi = np.zeros(game.MOVE_AMOUNT)
            pi[action] = 1
        else: 
            for _ in range(config.MCTS_SIMS): self.mcts.simulate(self.nn)

            pi = self.getAV(self.mcts, tau)
            
            action = self.choose_action(pi, tau)
        self.mcts = self.mcts.update_root(action)

        nn_value = self.nn.get_preds(self.mcts)[0]
        self.print_move(self.mcts, pi, action, nn_value)

        return pi

    @staticmethod
    def getAV(root, tau):
        pi = np.zeros(game.MOVE_AMOUNT)

        for edge in root.edges:
            pi[edge.action] = edge.n  # ** 1/tau

        pi /= np.sum(pi)

        return pi

    @staticmethod
    def choose_action(pi, tau):
        if tau == 1e-2:
            actions = np.flatnonzero(pi == np.max(pi))
            action = np.random.choice(actions)
        else: action = np.where(np.random.multinomial(1, pi) == 1)[0][0]

        return action

    def copy_profile(self, agent):
        self.outcomes = agent.outcomes

        self.nn.copy_weights(agent.nn)
    
    @staticmethod
    def print_move(root, pi, action, nn_value):
        game.print_values(pi)
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.print_board(root.s)}")
        print(f"Deck length is now: {len(root.deck)}")
        print(f"NN percieved value is: {nn_value:.3f}")
