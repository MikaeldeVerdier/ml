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

    def u(self):
        return config.CPUCT * self.prior * np.sqrt((np.log(self.parent.n) if self.parent.n != 0 else 0) / (1 + self.n))

    def u2(self, epsilon, nuidx):
        return config.CPUCT * ((1 - epsilon) * self.prior + nuidx * epsilon) * np.sqrt(self.parent.n) / (1 + self.n)

    def update_root(self, action):
        if not self.children:
            root = Node(game.move(self.s.copy(), action, self.player), self, action, -self.player, 0)
            self.children.append(root)
        else: root = self.children[action % game.MOVE_AMOUNT]

        return root

    """def simulate2(self, nn):
        if self.children:
            p = self.probabilities()
            self.children[np.random.choice(np.flatnonzero(p == np.max(p)))].simulate(nn)
        else:
            outcome = game.check_game_over(self.s)
            if outcome is None:
                self.expand_fully(nn)
                v = nn.get_preds(self)[0]
            else: v = outcome
            if self.parent:
                self.backfill(v)"""

    def simulate(self, nn):
        root = self
        while root.children:
            p = root.probabilities2(root == self)
            root = root.children[np.random.choice(np.flatnonzero(p == np.max(p)))]
        outcome = game.check_game_over(root.s)
        if outcome is None:
            root.expand_fully(nn)
            v = nn.get_preds(root)[0]
        else: v = outcome
        if root.parent: root.backfill(v)

    def expand_fully(self, nn):
        prior = nn.get_preds(self)[1]
        
        for action in game.get_legal_moves(self.s):
            if action != -1:
                new_state = game.move(self.s.copy(), action, self.player)
                child_node = Node(new_state, self, action, -self.player, prior[action])
            else: child_node = Node(np.full(np.prod(game.GAME_DIMENSIONS), 2), self, -1, 0, -9999)
                
            self.children.append(child_node)

    def probabilities(self):        
        return [child.q + child.u() for child in self.children]

    def probabilities2(self, is_root):
        if is_root:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * len(self.children))
        else:
            epsilon = 0
            nu = [0] * len(self.children)
        return [child.q + child.u2(epsilon, nu[i]) for i, child in enumerate(self.children)]

    """def backfill2(self, v):
        self.n += 1
        self.w += v * -self.player
        self.q = self.w / self.n
        if self.parent: self.parent.backfill2(v)"""

    def backfill(self, v):
        root = self
        while root:
            root.n += 1
            root.w += v * -root.player
            root.q = root.w / root.n
            root = root.parent
