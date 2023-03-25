import numpy as np
import random

import config
from funcs import increment_turn, print_state, print_move, get_card

DECK_LENGTH = 52
SUIT_AMOUNT = 4

GAME_DIMENSIONS = (5, 5)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (DECK_LENGTH * config.DEPTH,), (DECK_LENGTH * config.DEPTH,), (DECK_LENGTH * config.DEPTH,)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1
REPLACE_CARDS = 3
GAME_LENGTH = np.prod(GAME_DIMENSIONS) + REPLACE_CARDS

REWARD_FACTOR = 0.02
REWARD_TRANSFORM = lambda outcome: outcome / REWARD_FACTOR
INVERSE_REWARD_TRANSFORM = lambda transformed_outcome: int(transformed_outcome / REWARD_FACTOR)
REWARD_AVERAGE = True

class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players
		self.epsilons = epsilons or np.full(np.array(players).shape, None)
		self.starts = starts - 1
		self.verbose = verbose

		self.players_turn = -1

		self.deck = list(range(1, DECK_LENGTH + 1))

	def step(self, probs, action):
		s, deck, drawn_card = self.game_state.take_action(action)
		self.game_state = GameState(increment_turn(self.game_state.turn, 1, len(self.current_players)), self.game_state.history, s, deck, drawn_card)

		if self.verbose:
			print_move(self, probs, action)

		self.update_player()

	def update_player(self):
		self.player = self.current_players[self.game_state.turn]
		self.epsilon = self.epsilons[self.players_turn][self.game_state.turn]

	def reset(self):
		self.players_turn = increment_turn(self.players_turn, 1, len(self.players))
		self.current_players = self.players[self.players_turn]

		self.starts = increment_turn(self.starts, 1, len(self.current_players))

		if self.players_turn == 0:
			random.shuffle(self.deck)

		deck = self.deck.copy()
		drawn_card = deck.pop()

		self.game_state = GameState(self.starts, (None,) * config.DEPTH, np.zeros(np.prod(GAME_DIMENSIONS)), deck, drawn_card)

		self.update_player()

		if self.verbose:
			print_state(self.game_state)


class GameState():
	def __init__(self, turn, history, s, deck, drawn_card):
		self.turn = turn
		self.history = history[1:] + (self,)
		self.s = s
		self.deck = deck
		self.drawn_card = drawn_card

		self.replace_card = not len(np.where(self.s == 0)[0])
		self.legal_moves = self.get_legal_moves()
		self.done = self.check_game_over()
		self.scores = self.get_scores()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

	def take_action(self, action):
		board = self.s.copy()
		deck = self.deck.copy()

		if action != np.prod(GAME_DIMENSIONS):
			board[action] = self.drawn_card

		return (board, deck, deck.pop())

	def get_legal_moves(self):
		if not len(np.where(self.s != 0)[0]): return list(range(np.prod(GAME_DIMENSIONS)))

		if self.replace_card: return list(range(MOVE_AMOUNT))

		legal_moves = []

		for index in np.where(self.s != 0)[0]:
			for multiplier in [-1, 0, 1]:
				for add_on in [-1, 0, 1]:
					if not multiplier and not add_on:
						continue
					check_index = index + GAME_DIMENSIONS[1] * multiplier + add_on
					if check_index not in legal_moves and 0 <= check_index < np.prod(GAME_DIMENSIONS) and not self.s[check_index] and check_index // GAME_DIMENSIONS[1] - index // GAME_DIMENSIONS[1] == multiplier:
						legal_moves.append(check_index)

		return legal_moves

	def check_game_over(self):
		return len(self.deck) == DECK_LENGTH - GAME_LENGTH - 1

	def get_scores(self):
		if not self.done:
			return (0,)

		sum_score = 0

		board = self.s.reshape(GAME_DIMENSIONS)
		for rowcol in [board, board.T]:
			for row in rowcol:
				suits, values = tuple(zip(*[get_card(card) for card in row]))
				values = sorted(values)

				histo_dict = {(4,): 20, (3, 2): 15, (3,): 8, (2,): 2}

				histo = tuple(sorted([values.count(value) for value in set(values)]))

				maxes = {num: comb.count(num) for comb in histo_dict.keys() for num in comb}
				for key, value in list(histo_dict.items()):
					key_count = list(zip(*[(histo.count(val) // key.count(val), maxes[val]) for val in key]))

					if min(key_count[0]) and min(key_count[1]) > 0:
						for val in set(key):
							maxes[val] -= min(key_count[0])

						sum_score += min(key_count[0]) * value

				färgrad = len(set(suits)) == 1
				stege = values[-1] - values[0] == len(row) - 1 or values == list(range(1, len(row))) + [DECK_LENGTH / SUIT_AMOUNT + 1]

				if färgrad:
					sum_score += 10
				if stege:
					if values[-2] == DECK_LENGTH / SUIT_AMOUNT:
						sum_score += 40 if färgrad else 20
					else:
						sum_score += 10

		return (sum_score * REWARD_FACTOR,)

	def generate_nn_pass(self, modify=False):
		game_state = self.history[-1]

		if modify:
			flips = [None, 0, 1, (0, 1)]
			suit_changes = [i * DECK_LENGTH / SUIT_AMOUNT for i in range(SUIT_AMOUNT)]
		else:
			flips = [None]
			suit_changes = [0]

		nn_pass = []
		for flip in flips:
			s = game_state.s if flip is None else np.flip(game_state.s.reshape(GAME_DIMENSIONS), flip).flatten()

			for suit_change in suit_changes:
				nn_pass.append([[] for _ in range(len(NN_INPUT_DIMENSIONS))])

				for depth in range(config.DEPTH):
					de = game_state.deck
					dr = [game_state.drawn_card]
					for var in [s, de, dr]:
						for i, card in enumerate(var):
							if card != 0:
								var[i] = int((var[i] + suit_change - 1) % DECK_LENGTH + 1)

					state = np.moveaxis([np.reshape((s == i).astype(int), NN_INPUT_DIMENSIONS[0][:-1]) for i in range(1, DECK_LENGTH + 1)], 0, -1).tolist()
					nn_pass[-1][0] += state

					deck = np.zeros(DECK_LENGTH)
					deck[np.array(de, dtype=np.int32) - 1] = 1
					nn_pass[-1][1] += deck.tolist()

					drawn_card = np.zeros(DECK_LENGTH)
					drawn_card[dr[0] - 1] = (dr[0] != 0)
					nn_pass[-1][2] += drawn_card.tolist()

					if depth != config.DEPTH - 1:
						if self.history[-depth - 2]:
							game_state = self.history[-depth - 2]
						else:
							for _ in range(config.DEPTH - depth - 1):
								for i, func in enumerate([np.zeros, np.ones, np.zeros]):
									nn_pass[-1][i].append(func(NN_INPUT_DIMENSIONS[i][:-1]))
							break
		return nn_pass
