import numpy as np
import game

class GameState:
    def __init__(self, state, deck, drawn_card):
        self.s = state
        self.deck = deck
        self.drawn_card = drawn_card

        self.replace_card = not len(np.where(state == 0)[0])

    def __hash__(self):
        return hash(tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

    def update_root(self, action):
        (self.s, self.deck, self.drawn_card) = game.take_action(self, action)
