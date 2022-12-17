import numpy as np
import game

class GameState:
    def __init__(self, state, deck):
        self.s = state
        self.deck = deck

    def __hash__(self):
        return hash(tuple(self.s) + tuple(self.deck))

    def update_root(self, action):
        self.s, self.deck = game.take_action(self, action)
        return GameState(self.s, self.deck)
