import numpy as np
import random

import config
from funcs import increment_turn, print_state, print_action, calculate_legal_moves, score_row, format_state, cache

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
		self.epsilons = epsilons or np.full_like(players, None)
		self.starts = starts - 1
		self.verbose = verbose
		
		self.players_turn = -1

		self.deck = list(range(DECK_LENGTH))

		self.players_names = np.reshape([agent.full_name for agent in np.concatenate(players)], np.shape(players)).tolist()

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

		empty_history = (None,) * config.DEPTH
		empty_state = np.full(np.prod(GAME_DIMENSIONS), -1)
		self.game_state = GameState(self.starts, empty_history, empty_state, deck, drawn_card)

		self.update_turn()

		if self.verbose:
			print_state(self.game_state)

	def step(self, probs, action):
		s, deck, drawn_card = self.game_state.take_action(action)
		new_turn = increment_turn(self.game_state.turn, 1, len(self.current_players))
		self.game_state = GameState(new_turn, self.game_state.history, s, deck, drawn_card)

		if self.verbose:
			print_action(self, probs, action)

		self.update_turn()


class GameState():
	def __init__(self, turn, history, s, deck, drawn_card):
		self.turn = turn
		self.history = history[1:] + (self,)
		self.s = s
		self.deck = deck
		self.drawn_card = drawn_card

		amount_empty = len(np.where(self.s == -1)[0])
		self.first_card = amount_empty == len(s)
		self.replace_card = not amount_empty

		self.legal_moves = self.get_legal_moves()
		self.done = self.check_game_over()
		self.reward = self.get_reward()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

	def take_action(self, action):
		board = self.s.copy()
		deck = self.deck.copy()

		if action != np.prod(GAME_DIMENSIONS):
			board[action] = self.drawn_card

		return (board, deck, deck.pop())

	def get_legal_moves(self):
		if self.first_card:
			return list(range(np.prod(GAME_DIMENSIONS)))

		if self.replace_card:
			return list(range(MOVE_AMOUNT))

		legal_moves = calculate_legal_moves(tuple(self.s))

		return legal_moves

	def check_game_over(self):
		return len(self.deck) == DECK_LENGTH - np.prod(GAME_DIMENSIONS) - REPLACE_CARDS - 1

	def get_reward(self):
		if not self.done:
			return 0

		sum_score = 0

		board = self.s.reshape(GAME_DIMENSIONS)
		for rowcol in [board, board.T]:
			for row in rowcol:
				sum_score += score_row(tuple(row))

		return reward_transform(sum_score)

	@cache(100000)
	def format_game_state(self, flip, suit_change):
		game_state = self.history[-1]

		nn_pass = [[] for _ in range(len(NN_INPUT_DIMENSIONS))]

		for depth in range(config.DEPTH):
			state = np.flip(game_state.s.reshape(GAME_DIMENSIONS), flip).flatten() if flip is not None else game_state.s.copy()
			state_deck = game_state.deck.copy()
			state_drawn_card = [game_state.drawn_card]

			for var in [state, state_deck, state_drawn_card]:
				for i, card in enumerate(var):
					if card != -1:
						var[i] = int((var[i] + suit_change) % DECK_LENGTH)

			formatted_state = format_state(tuple(state))
			nn_pass[0].append(formatted_state)

			deck = np.zeros(DECK_LENGTH)
			deck[np.array(state_deck, dtype=np.int32)] = 1
			nn_pass[1].append(deck.tolist())

			drawn_card = np.zeros(DECK_LENGTH)
			drawn_card[state_drawn_card[0]] = 1
			nn_pass[2].append(drawn_card.tolist())

			if depth != config.DEPTH - 1:
				if self.history[-depth - 2]:
					game_state = self.history[-depth - 2]
				else:
					for i, func in enumerate([np.zeros, np.ones, np.zeros]):
						empty_dim = func(np.shape(nn_pass[i][-1])).tolist()
						nn_pass[i] += [empty_dim] * (config.DEPTH - depth - 1)

					break

		nn_pass[0] = np.moveaxis(nn_pass[0], 0, -2).tolist()

		return nn_pass

	def generate_nn_pass2(self, modify=False):
		if modify:
			flips = [None, 0, 1, (0, 1)]
			suit_changes = [i * SUIT_LENGTH for i in range(SUIT_AMOUNT)]
		else:
			flips = [None]
			suit_changes = [0]

		nn_pass = []
		for flip in flips:
			for suit_change in suit_changes:
				nn_pass.append(self.format_game_state(flip, suit_change))

		return nn_pass

	def generate_nn_pass3(self, modify=False):
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

					state = []
					for i in range(1, DECK_LENGTH + 1):
						position = np.zeros(len(s))
						position[s == i] = 1
						state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-2]))

					state = np.moveaxis(state, 0, -1).tolist()
					nn_pass[-1][0].append(state)

					deck = np.zeros(DECK_LENGTH)
					deck[np.array(de, dtype=np.int32) - 1] = 1
					nn_pass[-1][1].append(deck.tolist())

					drawn_card = np.zeros(DECK_LENGTH)
					drawn_card[dr[0] - 1] = (dr[0] != 0)
					nn_pass[-1][2].append(drawn_card.tolist())

					nn_pass[-1][3].append([game_state.scores[0]])

					if depth != config.DEPTH - 1:
						if self.history[-depth - 2]:
							game_state = self.history[-depth - 2]
						else:
							for _ in range(config.DEPTH - depth - 1):
								for i, func in enumerate([np.zeros, np.ones, np.zeros]):
									nn_pass[-1][i].append(func(NN_INPUT_DIMENSIONS[i][:-1]))
							break

				nn_pass[-1] = [np.moveaxis(dim, 0, -2).tolist() for dim in nn_pass[-1]]

		return nn_pass
	
	def generate_nn_pass(self, modify=False):
		a = self.generate_nn_pass2(modify=modify)
		b = self.generate_nn_pass3(modify=modify)

		c = a == b
		if not c:
			pass

		return a
