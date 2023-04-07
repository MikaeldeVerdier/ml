import numpy as np
import random

import config
from funcs import increment_turn, print_state, print_move, calculate_legal_moves, score_row, format_state, cache

DECK_LENGTH = 52
SUIT_AMOUNT = 4
SUIT_LENGTH = DECK_LENGTH / SUIT_AMOUNT

GAME_DIMENSIONS = (5, 5)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (config.DEPTH, DECK_LENGTH), (config.DEPTH, DECK_LENGTH), (config.DEPTH, DECK_LENGTH)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1

REPLACE_CARDS = 3
REWARD_FACTOR = 0.02

def reward_transform(reward):
	return reward * REWARD_FACTOR


def inverse_reward_transform(transformed_reward):
	return int(transformed_reward / REWARD_FACTOR)


def results_transform(results):
	return np.mean(results, axis=-1)

	# return np.count_nonzero(results)


class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players
		self.epsilons = epsilons or np.full(np.shape(players), None)
		self.starts = starts - 1
		self.verbose = verbose

		self.players_turn = -1

		self.deck = list(range(1, DECK_LENGTH + 1))

	def step(self, probs, action):
		s, deck, drawn_card = self.game_state.take_action(action)
		new_turn = increment_turn(self.game_state.turn, 1, len(self.current_players))
		self.game_state = GameState(new_turn, self.game_state.history, s, deck, drawn_card)

		if self.verbose:
			print_move(self, probs, action)

		self.update_turn()

	def update_turn(self):
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

		self.update_turn()

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
		if not len(np.where(self.s != 0)[0]):
			return list(range(np.prod(GAME_DIMENSIONS)))

		if self.replace_card:
			return list(range(MOVE_AMOUNT))

		legal_moves = calculate_legal_moves(tuple(self.s))

		return legal_moves

	def check_game_over(self):
		return len(self.deck) == DECK_LENGTH - np.prod(GAME_DIMENSIONS) - REPLACE_CARDS - 1

	def get_scores(self):
		if not self.done:
			return (0,)

		sum_score = 0

		board = self.s.reshape(GAME_DIMENSIONS)
		for rowcol in [board, board.T]:
			for row in rowcol:
				sum_score += score_row(tuple(row))

		return (reward_transform(sum_score),)

	def generate_nn_pass(self, modify=False):
		game_state = self.history[-1]

		if modify:
			flips = [None, 0, 1, (0, 1)]
			suit_changes = [i * SUIT_LENGTH for i in range(SUIT_AMOUNT)]
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

					state = format_state(tuple(s))
					nn_pass[-1][0].append(state)

					deck = np.zeros(DECK_LENGTH)
					deck[np.array(de, dtype=np.int32) - 1] = 1
					nn_pass[-1][1].append(deck.tolist())

					drawn_card = np.zeros(DECK_LENGTH)
					drawn_card[dr[0] - 1] = (dr[0] != 0)
					nn_pass[-1][2].append(drawn_card.tolist())

					if depth != config.DEPTH - 1:
						if self.history[-depth - 2]:
							game_state = self.history[-depth - 2]
						else:
							for i, func in enumerate([np.zeros, np.ones, np.zeros]):
								empty_dim = cache(1)(func)(np.shape(nn_pass[-1][i][-1])).tolist()
								nn_pass[-1][i] += [empty_dim] * (config.DEPTH - depth - 1)
							
							break

				nn_pass[-1] = [np.moveaxis(dim, 0, -2).tolist() for dim in nn_pass[-1]]

		return nn_pass
