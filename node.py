import numpy as np
import config
import game

class Node:
    def __init__(self, state, parent, parent_action, player, prior):
        self.s = state
        self.parent = parent
        self.parent_action = parent_action
        self.player = player

        self.children = []
        
        self.n = 0
        self.w = 0
        self.q = 0
        self.prior = prior
        
        self.cpuct = np.sqrt(2)
   
    def u(self):
        return self.cpuct * self.prior * np.sqrt((np.log(self.parent.n) if self.parent.n != 0 else 0) / (1 + self.n))

    def update_root(self, action):
        descendant = [child for child in self.children if np.array_equal(child.s, game.move(self.s.copy(), action, self.player))]
        if not descendant:
            root = Node(game.move(self.s.copy(), action, self.player), self, action, -self.player, 0)
            self.children.append(root)
        else: root = descendant[0]

        return root

    def simulate_rec(self, nn):
        outcome = game.check_game_over(self.s)
        if self.children:
            if outcome is None:
                p = self.probabilities()
                self.children[np.random.choice(np.flatnonzero(p == np.max(p)))].simulate_rec(nn)
            else: self.backfill(outcome)
        else:
            if outcome is None:
                self.expand_fully(nn)
                v = nn.get_preds(self)[0]
            else: v = outcome
            if self.parent: self.backfill(v)

    def simulate_loop(self, nn):
        root = self
        while root.children:
            p = root.probabilities()
            root = root.children[np.random.choice(np.flatnonzero(p == np.max(p)))]

        if outcome is None:
            root.expand_fully(nn)
            v = nn.get_preds(self)[0]
        else: v = outcome
        if self.parent: self.backfill(v)
        
        outcome = game.check_game_over(root.s)

    def expand_fully(self, nn):
        prior = nn.get_preds(self)[1] if nn is not None else [0] * np.prod(config.game_dimensions)
        
        for action in game.get_legal_moves(self.s):
            if action != -1:
                new_state = game.move(self.s.copy(), action, self.player)
                child_node = Node(new_state, self, action, -self.player, prior[action])
            else:
                child_node = Node(np.full(np.prod(config.game_dimensions), 2), self, -1, 0, -2)
                
            self.children.append(child_node)

    def probabilities(self):
        pi = [child.q + child.u() for child in self.children]
        
        return pi

    def backfill(self, v):
        self.n += 1
        self.w += v * -self.player
        self.q = self.w / self.n
        if self.parent:
            self.parent.backfill(v)
