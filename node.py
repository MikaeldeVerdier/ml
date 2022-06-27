import numpy as np
import random
import config
import game

class Node:
    def __init__(self, state, parent, parent_action, player, prior, modifier=1, comp=True):
        self.s = state
        self.parent = parent
        self.parent_action = parent_action
        self.player = player

        self.untried_actions = game.get_legal_moves(self.s)
        self.children = []
        self.n = 1
        self.w = 0
        self.q = 0
        self.prior = prior
        self.modifier = modifier
        
        self.tau = 1 if not comp else 10e-15
        self.cpuct = np.sqrt(2)
   
    def u(self):
        return self.cpuct * self.prior * np.sqrt(self.parent.n)/(1 + self.n) * self.modifier

    def selection(self, nn):
        if len(self.children) != len(game.get_legal_moves(self.s)):
            if game.check_game_over(self.s) is not None:
                return
            self.expand(nn)
            self.backfill(nn)
        else:
            self.p = self.probabilities()
            self.children[np.argmax(self.p) % config.move_amount].selection(nn)

    def expand(self, nn):
        action = self.untried_actions.pop(0)
        if action != -1:
            new_state = game.move(self.s.copy(), action, self.player)[0]
            prior = nn.test(game.generate_game_state(self))[1][action % config.move_amount] if nn is not None else 0
            child_node = Node(new_state, self, action, -self.player, prior)
        else:
            child_node = Node(np.full(np.prod(config.game_dimensions), 2), self, action, 0, 1, modifier=0)

        self.children.append(child_node)

    def probabilities(self):
        pi = np.zeros(np.prod(config.game_dimensions))
        for child in self.children: pi[child.parent_action] = child.q + child.u()
        
        mask = np.full(np.prod(config.game_dimensions), True)
        mask[game.get_legal_moves(self.s)] = False
        pi[mask] = -100
        
        odds = np.exp(pi)
        probs = odds / np.sum(odds)
        return probs

    def backfill(self, nn):
        self.n += 1
        self.v = nn.test(game.generate_game_state(self))[0]
        if self.parent:
            self.parent.w += self.v
            self.parent.q = self.parent.w / self.parent.n
            self.parent.backfill(nn)
