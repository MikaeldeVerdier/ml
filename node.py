import numpy as np
import random
import config
import game

class Node:
    def __init__(self, state, parent, parent_action, player, nn, prior, modifier=1, comp=True):
        self.s = state
        self.parent = parent
        self.parent_action = parent_action
        self.player = player

        # self.nn_pass = game.generate_game_state(self)
        self.nn = nn
        self.untried_actions = game.get_legal_moves(self.s)
        self.children = []
        self.results = {-1: 0, 0: 0, 1: 0}
        self.n = 1
        self.w = 0
        self.q = 0
        self.prior = prior
        self.modifier = modifier
        
        self.tau = 1 if not comp else 10e-15
        self.cpuct = np.sqrt(2)

    def play_turn(self, tau):
        for _ in range(config.MCTSSims):
            self.selection()

        pi, values = self.getAV(1)
        
        action, value = self.choose_action(pi, values, tau)
        new_state = game.move(self.s.copy(), action, self.player, True)[0]
        node = None
        for child in self.children:
            if np.array_equal(child.s, new_state):
                node = child
                break
        nn_value = -self.nn.test(game.generate_game_state(node))[0]

        return (node, action, pi, value, nn_value)
   
    def u(self):
        return (self.results[self.player]/self.n + self.cpuct * self.prior * np.sqrt(np.log(self.parent.n)/(1 + self.n))) * self.modifier

    def selection(self):
        if len(self.children) != len(game.get_legal_moves(self.s)):
            if game.check_game_over(self.s) is not None:
                return
            self.expand()
            reward = self.children[-1].simulate()
            self.backfill(reward)
        else:
            self.p = self.probabilities()
            self.children[np.argmax(self.p) % game.move_amount].selection()

    def expand(self):
        action = self.untried_actions.pop(0)
        if action == -1:
            child_node = Node(np.full(np.prod(game.game_dimensions), 2), self, action, 0, None, 1, modifier=0)
        else:
            new_state = game.move(self.s.copy(), action, self.player, False)[0]
            prior = self.nn.test(game.generate_game_state(self))[1][action % game.move_amount]
            child_node = Node(new_state, self, action, -self.player, self.nn, prior)
        self.children.append(child_node)

    def simulate(self):
        current_state = self.s.copy()
        player = self.player

        outcome = game.check_game_over(current_state)
        while outcome is None:
            action = random.choice(game.get_legal_moves(current_state))
            (current_state, player) = game.move(current_state, action, player, True)
            outcome = game.check_game_over(current_state)
        return outcome

    def getAV(self, tau):
        pi = np.zeros(np.prod(config.game_dimensions))
        values = np.zeros(np.prod(config.game_dimensions))

        for child in self.children:
            if child.parent_action != -1:
                pi[child.parent_action] = child.n ** (1 / tau) * self.modifier
                values[child.parent_action] = child.q

        pi /= np.sum(pi)

        return pi, values

    def probabilities(self):
        pi = np.zeros(np.prod(config.game_dimensions))
        for child in self.children: pi[child.parent_action] = child.q + child.u()
        
        mask = np.full(np.prod(config.game_dimensions), True)
        mask[game.get_legal_moves(self.s)] = False
        pi[mask] = -100
        
        odds = np.exp(pi)
        probs = odds / np.sum(odds)
        return probs
        # return [child.q + child.u() for child in self.children]

    def choose_action(self, pi, values, tau):
        if tau == 0: action = np.argmax(pi)
        else:
            actions = np.random.multinomial(1, pi)
            action = np.where(actions==1)[0][0]

        value = values[action]

        return action, value

    def backfill(self, result):
        self.n += 1
        self.results[result] += 1

        self.v = self.nn.test(game.generate_game_state(self))[0]
        if self.parent:
            self.parent.w += self.v
            self.parent.q = self.parent.w/self.parent.n
            self.parent.backfill(result)
        # self.theta_backpropogate(result, self.v, self.probabilities(), res[1:])
