import numpy as np
import game
import config

class Node:
    def __init__(self, state, deck, drawn_card, tree):
        self.s = state
        self.deck = deck
        self.drawn_card = drawn_card
        self.tree = tree

        self.replace_card = not len(np.where(state == 0)[0])
        self.edges = []

        self.tree.add_node(self)
    
    def __hash__(self):
        return hash(tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

    def create_node(self, action):
        new_node_info = game.take_action(self, action)
        new_nodes = []
        for new_state, new_deck, new_drawn_card in new_node_info:
            if self.tree.check_state(new_state, new_deck, new_drawn_card):
                # print("NODE EXISTED")
                new_nodes.append(self.tree.get_node(new_state, new_deck, new_drawn_card))
            else:
                new_nodes.append(Node(new_state, new_deck, new_drawn_card, self.tree))
                self.tree.add_node(new_nodes[-1])
                # self.add_node(new_node)

        return new_nodes

    def update_root(self, action):
        return self.create_node(action)[-1]

    def simulate(self, nn):
        breadcrumbs = []
        root = self
        while root.edges:
            if len(root.edges) == 1: edge = root.edges[0]
            else:
                p = root.probabilities(root == self)
                edge = root.edges[np.random.choice(np.flatnonzero(p == np.max(p)))]
            root = np.random.choice(edge.out_nodes)
            breadcrumbs.append(edge)

        outcome = game.check_game_over(root)
        if outcome is None:
            (v, p) = nn.get_preds(root)
            root.expand_fully(p)
        else: v = outcome
        
        root.backfill(v, breadcrumbs)

    def expand_fully(self, prior):
        for action in game.get_legal_moves(self):
            new_nodes = self.create_node(action)
            edge = Edge(self, new_nodes, action, prior[action])
            self.edges.append(edge)

    def probabilities(self, is_root):
        if is_root:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * len(self.edges))
        else:
            epsilon = 0
            nu = [0] * len(self.edges)
        nb = sum(edge.n for edge in self.edges)
        probs = np.array([edge.q + edge.u(epsilon, nu[i], nb) for i, edge in enumerate(self.edges)])
        non_visited = [i for i, edge in enumerate(self.edges) if edge.n == 0]
        probs[non_visited] = max(probs) + 1

        return probs

    def backfill(self, v, breadcrumbs):
        for edge in breadcrumbs:
            edge.n += 1
            edge.w += v
            edge.q = edge.w / edge.n


class Edge:
    def __init__(self, in_node, out_nodes, action, prior):
        self.in_node = in_node
        self.out_nodes = out_nodes
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

    def check_state(self, state, deck, drawn_card):
        return hash(tuple(state) + tuple(deck) + (drawn_card,)) in self.saved_nodes

    def add_node(self, node):
        self.saved_nodes[hash(node)] = node
    
    def get_node(self, state, deck, drawn_card):
        return self.saved_nodes[hash(tuple(state) + tuple(deck) + (drawn_card,))]
