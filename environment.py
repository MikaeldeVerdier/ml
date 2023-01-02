import numpy as np
import random
import config
from funcs import format_card, score_row

class Environment:
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.GAME_DIMENSIONS = (5, 5)
        self.NN_INPUT_DIMENSIONS = [(config.DEPTH,) + self.GAME_DIMENSIONS + (52,), (config.DEPTH, 52), (config.DEPTH, 52)]
        self.MOVE_AMOUNT = np.prod(self.GAME_DIMENSIONS) + 1
        self.REPLACE_CARDS = 3
        self.GAME_LENGTH = np.prod(self.GAME_DIMENSIONS) + self.REPLACE_CARDS
        self.REWARD_FACTOR = 0.05

    def step(self, action):
        s, deck, drawn_card = self.game_state.take_action(action)
        self.game_state = GameState(self.game_state.history, s, deck, drawn_card)

        if self.verbose:
            self.print_state()

    def reset(self):
        deck = list(range(1, 53))
        random.shuffle(deck)
        drawn_card = deck.pop()
        
        self.game_state = GameState((None,) * config.DEPTH, np.zeros(np.prod(self.GAME_DIMENSIONS)), deck, drawn_card)

    def print_state(self):
        board = self.game_state.s.astype("<U4")
        board[board == "0.0"] = "---"
        for i, pos in enumerate(board):
            if pos != "---":
                board[i] = format_card(float(pos))
        
        print(f"Position is:\n{board.reshape(self.GAME_DIMENSIONS)}")
        print(f"Drawn card is: {format_card(self.game_state.drawn_card)}")
        print(f"Amount of cards left is now: {len(self.game_state.deck)}")


class GameState(Environment):
    def __init__(self, history, s, deck, drawn_card):
        super().__init__()

        self.history = history[1:] + (self,)
        self.s = s
        self.deck = deck
        self.drawn_card = drawn_card

        self.replace_card = not len(np.where(self.s == 0)[0])
        self.legal_moves = self.get_legal_moves()

    def __hash__(self):
        return hash(self.history[:-1] + tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

    def take_action(self, action):
        board = self.s.copy()
        deck = self.deck.copy()

        if action != np.prod(self.GAME_DIMENSIONS):
            board[action] = self.drawn_card

        return (board, deck, deck.pop())

    def get_legal_moves(self):
        if not len(np.where(self.s != 0)[0]): return list(range(np.prod(self.GAME_DIMENSIONS)))

        if self.replace_card: return list(range(self.MOVE_AMOUNT))

        legal_moves = []

        for index in np.where(self.s != 0)[0]:
            for multiplier in [-1, 0, 1]:
                for add_on in [-1, 0, 1]:
                    if not multiplier and not add_on:
                        continue
                    check_index = index + self.GAME_DIMENSIONS[1] * multiplier + add_on
                    if check_index not in legal_moves and 0 <= check_index < np.prod(self.GAME_DIMENSIONS) and not self.s[check_index] and check_index // self.GAME_DIMENSIONS[1] - index // self.GAME_DIMENSIONS[1] == multiplier:
                        legal_moves.append(check_index)

        return legal_moves

    def check_game_over(self):
        if len(self.deck) == 51 - self.GAME_LENGTH:
            score = 0
            board = self.s.reshape(self.GAME_DIMENSIONS)
            for rowcol in [board, board.T]:
                for row in rowcol:
                    score += score_row(row)
            
            return score * self.REWARD_FACTOR

    def generate_game_states(history, t, key="state"):
        data = history[:t + 1]
        game_states = ({key: None},) * (config.DEPTH - len(data)) + tuple(data)[-config.DEPTH:]
        game_states = tuple([game_state[key] for game_state in game_states])

        return game_states

    def generate_nn_pass(self, modify=False):
        game_state = self.history[-1]

        if modify:
            flips = [None, 0, 1, (0, 1)]
            suit_changes = [0, 13, 26, 39]
        else:
            flips = [None]
            suit_changes = [0]

        nn_pass = []
        for flip in flips:
            s = game_state.s if flip is None else np.flip(game_state.s.reshape(self.GAME_DIMENSIONS), flip).flatten()
            
            for suit_change in suit_changes:
                nn_pass.append([[], [], []])
                for depth in range(config.DEPTH):
                    de = game_state.deck
                    dr = [game_state.drawn_card]
                    for var in [s, de, dr]:
                        for i, card in enumerate(var):
                            if card != 0:
                                var[i] += suit_change
                            if var[i] > 52:
                                var[i] -= 52

                    state = []
                    for i in range(1, 53):
                        position = np.zeros(len(s))
                        position[s == i] = 1
                        state.append(np.reshape(position, self.NN_INPUT_DIMENSIONS[0][1:-1]))

                    state = np.moveaxis(state, 0, -1).tolist()
                    nn_pass[-1][0].append(state)

                    deck = np.zeros(52)
                    for card in de: deck[card - 1] = 1
                    nn_pass[-1][1].append(deck.tolist())

                    drawn_card = np.zeros(52)
                    if dr[0] != 0:
                        drawn_card[dr[0] - 1] = 1
                    nn_pass[-1][2].append(drawn_card.tolist())

                    if depth != config.DEPTH - 1:
                        if self.history[-depth - 2]:
                            game_state = self.history[-depth - 2]
                        else:
                            for _ in range(config.DEPTH - depth - 1):
                                for i, func in enumerate([np.zeros, np.ones, np.zeros]):
                                    nn_pass[-1][i].append(func(self.NN_INPUT_DIMENSIONS[i][:-1]))
                            break
        return nn_pass