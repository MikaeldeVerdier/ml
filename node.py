import numpy as np
import random
import config
import game

class Node:
    def __init__(self, state, parent, parent_action, player, prior):
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
        
        self.cpuct = np.sqrt(2)
   
    def u(self):
        return self.cpuct * self.prior * np.sqrt(self.parent.n)/(1 + self.n)

    def return_root(self, action):
        descendant = [child for child in self.children if np.array_equal(child.s, game.move(self.s.copy(), action, -self.player))]
        if not descendant:
            root = Node(game.move(self.s.copy(), action, -self.player), self, action, self.player, 0)
            self.children.append(root)
        else:
            root = descendant[0]

        return root

    def selection(self, nn):
        if len(self.children) != len(game.get_legal_moves(self.s)):
            if game.check_game_over(self.s) is not None:
                return
            self.expand(nn)
            self.backfill(nn)
        else:
            self.p = self.probabilities()
            self.children[np.argmax(self.p)].selection(nn)

    def expand(self, nn):
        action = self.untried_actions.pop(0)
        if action != -1:
            new_state = game.move(self.s.copy(), action, self.player)
            prior = nn.test(game.generate_game_state(self))[1][action % config.move_amount] if nn is not None else 0
            child_node = Node(new_state, self, action, self.player, prior)
        else:
            child_node = Node(np.full(np.prod(config.game_dimensions), 2), self, -1, 0, 0)

        self.children.append(child_node)

    """def expand_fully(self, nn):
        prior = nn.test(game.generate_game_state(self))[1] if nn is not None else 0
        
        for action in self.untried_actions:
            if action != -1:
                new_state = game.move(self.s.copy(), action, self.player)[0]
                child_node = Node(new_state, self, action, -self.player, prior[action % config.move_amount])
            else:
                child_node = Node(np.full(np.prod(config.game_dimensions), 2), self, action, 0, 1, modifier=0)

        self.children.append(child_node)"""

    def probabilities(self):
        pi = []
        for child in self.children: pi.append(child.q + child.u())
        
        odds = np.exp(pi)
        probs = odds / np.sum(odds)
        return probs

    def backfill(self, nn):
        v = nn.test(game.generate_game_state(self))[0]
        direction = self.player
        parent = self
        while parent:
            parent.n += 1
            parent.w += v * direction
            parent.q = parent.w / parent.n
            parent = parent.parent
            direction *= -1
