import numpy as np
import config
import game

class Node:
    def __init__(self, state, action, player):
        self.s = state
        self.action = action
        self.player = player

        self.edges = []
    
    def __hash__(self):
        return hash(tuple(self.s))

    def u(self):
        return config.CPUCT * self.prior * np.sqrt((np.log(self.parent.n) if self.parent.n != 0 else 0) / (1 + self.n))

    def u2(self, epsilon, nuidx, nb):
        return config.CPUCT * ((1 - epsilon) * self.prior + nuidx * epsilon) * np.sqrt(nb) / (1 + self.n)

    def update_root(self, action):
        if not self.children:
            root = Node(game.move(self.s.copy(), action, self.player), self, action, -self.player, 0)
            self.children.append(root)
        else: root = [child for child in self.children if child.parent_action == action][0]

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
        breadcrumbs = []
        root = self
        while root.edges:
            p = root.probabilities2(root == self)
            root = root.edges[np.random.choice(np.flatnonzero(p == np.max(p)))]
            # root = root.children[np.argmax(p)]
            breadcrumbs.append(root)
        outcome = game.check_game_over(root.s)
        if outcome is None:
            (v, p) = nn.get_preds(root)
            root.expand_fully(p)
        else: v = outcome * root.player
        root.backfill(v, breadcrumbs)

    def expand_fully(self, prior):
        # for action in sorted(game.get_legal_moves(self.s, False)):
        for action in game.get_legal_moves(self.s):
            # if action != -1:
            new_state = game.move(self.s.copy(), action, self.player)
            new_node = Node(new_state, action, -self.player)
            edge = Edge(self, self, action, prior[action])
            # else: child_node = Node(np.full(np.prod(game.GAME_DIMENSIONS), 2), self, -1, 0, -9999)

            self.edges.append(edge)

    def probabilities2(self, is_root):
        if is_root:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * len(self.edges))
            # nu = [0.1] * len(self.children)
        else:
            epsilon = 0
            nu = [0] * len(self.edges)
        # print(self.n - 1)
        nb = sum(edge.n for edge in self.edges)
        return [edge.q + edge.u2(epsilon, nu[i], nb) for i, edge in enumerate(self.edges)]

    """def backfill2(self, v):
        self.n += 1
        self.w += v * -self.player
        self.q = self.w / self.n
        if self.parent: self.parent.backfill2(v)"""

    def backfill(self, v, breadcrumbs):
        for edge in breadcrumbs:
            edge.n += 1
            edge.w += v * self.player * -edge.player
            edge.q = edge.w / edge.n

class Edge:
    def __init__(self, in_node, out_node, action, prior):
        self.in_node = in_node
        self.out_node = out_node
        self.player = in_node.player
        self.action = action
        
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = prior
