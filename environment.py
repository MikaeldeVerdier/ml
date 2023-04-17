import numpy as np
import random

import config
from funcs import increment_turn, print_state, print_action, calculate_legal_moves, score_row, format_state, cache

DECK_LENGTH = 24
SUIT_AMOUNT = 4
SUIT_LENGTH = DECK_LENGTH / SUIT_AMOUNT

GAME_DIMENSIONS = (3, 3)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (config.DEPTH, DECK_LENGTH), (config.DEPTH, DECK_LENGTH), (config.DEPTH, DECK_LENGTH), (config.DEPTH,)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1

REPLACE_CARDS = 1
REWARD_FACTOR = 0.25
INTERMEDIATE_REWARD_FACTOR = 0.15

def reward_transform(reward):
	return reward * REWARD_FACTOR


def intermediate_reward_transform(reward):
	return reward * INTERMEDIATE_REWARD_FACTOR


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
		sum_score = 0

		board = self.s.reshape(GAME_DIMENSIONS)
		for rowcol in [board, board.T]:
			for row in rowcol:
				row = row[np.where(row != -1)].tolist()
				sum_score += score_row(tuple(row))

		return (reward_transform if self.done else intermediate_reward_transform)(sum_score)

	@cache(100000)
	def format_game_state(self, flip, suit_change):
		game_state = self.history[-1]

		nn_pass = [[] for _ in range(len(NN_INPUT_DIMENSIONS))]

		for depth in range(config.DEPTH):
			state = np.rot90(game_state.s.reshape(GAME_DIMENSIONS), k=flip).flatten()
			state_deck = game_state.deck.copy()
			state_drawn_card = [game_state.drawn_card]

			for var in [state, state_deck, state_drawn_card]:
				for i, card in enumerate(var):
					if card != -1:
						var[i] = int((var[i] + suit_change) % DECK_LENGTH)

			formatted_state = format_state(tuple(state))
			nn_pass[0].append(formatted_state)

			deck = np.zeros(DECK_LENGTH, dtype=np.int32)
			deck[np.array(state_deck, dtype=np.int32)] = 1
			nn_pass[1].append(deck.tolist())

			drawn_card = np.zeros(DECK_LENGTH, dtype=np.int32)
			drawn_card[state_drawn_card[0]] = 1
			nn_pass[2].append(drawn_card.tolist())

			nn_pass[3].append([game_state.reward])

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

	def generate_nn_pass(self, modify=False):
		if modify:
			flips = range(SUIT_AMOUNT)
			suit_changes = [i * SUIT_LENGTH for i in range(SUIT_AMOUNT)]
		else:
			flips = [0]
			suit_changes = [0]

		nn_pass = []
		for flip in flips:
			for suit_change in suit_changes:
				nn_pass.append(self.format_game_state(flip, suit_change))

		return nn_pass
