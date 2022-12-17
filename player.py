import json
import numpy as np
import game
import config
import files

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
    def __init__(self, nn_class, load, version=None, name=None, to_weights=False):
        self.nn = nn_class(load, version, to_weights=to_weights)
        self.name = name
        
        self.outcomes = {"average": 0, "length": 0}

    def __gt__(self, other):
        return type(other) == BestAgent or type(self) == CurrentAgent and other.nn.version > self.nn.version

    def get_name(self):
        return f"Version {self.nn.version}" if not self.name else self.name

    def play_turn(self, history, tau):
        legal_moves = game.get_legal_moves(self.mcts)
        if len(legal_moves) == 1:
            action = legal_moves[0]
            pi = np.zeros(game.MOVE_AMOUNT)
            pi[action] = 1
        else: 
            for _ in range(config.MCTS_SIMS): self.mcts.simulate(self.nn)

            pi = self.getAV(tau)

            action = self.choose_action(pi, tau)
        self.mcts = self.mcts.update_root(action)

        nodes = game.generate_nodes(history, len(history) - 1)
        nn_value = self.nn.get_preds(nodes)[0]
           
        self.print_move(self.mcts, pi, action, nn_value)

        return action, pi[action], nn_value

    def getAV(self, tau):
        pi = np.zeros(game.MOVE_AMOUNT)

        for edge in self.mcts.edges:
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

    def save_outcomes(self, agent_kind):
        loaded = files.load_file("save.json")
        loaded[agent_kind]["version_outcomes"][self.nn.version] = self.outcomes["average"]

        files.write("save.json", json.dumps(loaded))
    
    @staticmethod
    def print_move(root, pi, action, nn_value):
        game.print_values(pi)
        print(f"Move to make is: {action}")
        print(f"Position is now:\n{game.print_board(root.s)}")
        print(f"NN percieved value is: {nn_value:.3f} ({(nn_value * 50):.3f})")
        print(f"Drawn card is: {game.format_card(root.drawn_card)}")
        print(f"Amount of cards left is now: {len(root.deck)}")


class CurrentAgent(Agent):  # Redundant currently
    def __init__(self, nn_class, load, version=None, name=None, to_weights=False):
        super().__init__(nn_class, load, version, name, to_weights)

    # def __lt__(self, other):
    #     return True if type(other) == type(self) and other.nn.version > self.nn.version else False


class BestAgent(Agent):
    def __init__(self, nn_class, load, version=None, name=None):
        super().__init__(nn_class, load, version, name)
    
    def copy_profile(self, agent):
        self.outcomes = agent.outcomes

        loaded = files.load_file("save.json")
        loaded["best_agent"] = loaded["current_agent"]

        files.write("save.json", json.dumps(loaded))

        self.nn.get_preds.cache_clear()
        self.nn.version = agent.nn.version
        self.nn.load_version(self.nn.version)
