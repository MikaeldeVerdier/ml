import numpy as np
import config
import game

class Node:
    def __init__(self, state, player, tree):
        self.s = state
        self.player = player
        self.tree = tree

        self.edges = []

        self.tree.add_node(self)
    
    def __hash__(self):
        return hash(tuple(self.s))

    def create_node(self, action):
        new_state = game.move(self.s.copy(), action, self.player)
        if self.tree.check_state(new_state):
            # print("NODE EXISTED")
            new_node = self.tree.get_node(new_state)
        else:
            new_node = Node(new_state, -self.player, self.tree)
            # self.add_node(new_node)
        
        return new_node

    def update_root(self, action):
        return self.create_node(action)

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
            # print(p)
            edge = root.edges[np.random.choice(np.flatnonzero(p == np.max(p)))]
            # edge = root.edges[np.argmax(p)]
            breadcrumbs.append(edge)
            root = edge.out_node
        outcome = game.check_game_over(root.s)
        if outcome is None:
            nodes = (self,) + tuple(edge.out_node for edge in breadcrumbs)
            (v, p) = nn.get_preds(root, nodes)
            root.expand_fully(p)
        else: v = outcome * root.player
        root.backfill(v, breadcrumbs)

    def expand_fully(self, prior):
        # for action in sorted(game.get_legal_moves(self.s)):
        for action in game.get_legal_moves(self.s):
            new_node = self.create_node(action)
            edge = Edge(self, new_node, action, prior[action])
            self.edges.append(edge)

    def probabilities2(self, is_root):
        if is_root:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * len(self.edges))
            # nu = [0.1] * len(self.edges)
        else:
            epsilon = 0
            nu = [0] * len(self.edges)
        # print(self.n - 1)
        nb = sum(edge.n for edge in self.edges)
        return [edge.q + edge.u(epsilon, nu[i], nb) for i, edge in enumerate(self.edges)]

    """def backfill2(self, v):
        self.n += 1
        self.w += v * -self.player
        self.q = self.w / self.n
        if self.parent: self.parent.backfill2(v)"""

    def backfill(self, v, breadcrumbs):
        for edge in breadcrumbs:
            edge.n += 1
            edge.w += v * self.player * edge.player
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

    def u(self, epsilon, nuidx, nb):
        return config.CPUCT * ((1 - epsilon) * self.p + nuidx * epsilon) * np.sqrt(nb) / (1 + self.n)

class Tree:
    def __init__(self):
        self.saved_nodes = {}

    def check_state(self, state):
        return hash(tuple(state)) in self.saved_nodes

    def add_node(self, node):
        self.saved_nodes[hash(node)] = node
    
    def get_node(self, state):
        return self.saved_nodes[hash(tuple(state))]

