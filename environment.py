import numpy as np
import random
from copy import deepcopy

import config
from funcs import increment_var, print_state, print_action, calculate_legal_moves, score_row, format_game_state

SUIT_AMOUNT = 2
SUIT_LENGTH = 8
DECK_LENGTH = SUIT_AMOUNT * SUIT_LENGTH

GAME_DIMENSIONS = (4, 4)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (config.DEPTH, DECK_LENGTH), (config.DEPTH, DECK_LENGTH), (config.DEPTH,)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS)

REWARD_FACTOR = 0.15
INTERMEDIATE_REWARD_FACTOR = 0.075

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
		self.players_turn = increment_var(self.players_turn, 1, len(self.players))
		self.current_players = self.players[self.players_turn]

		self.starts = increment_var(self.starts, 1, len(self.current_players))

		if self.players_turn == 0:
			random.shuffle(self.deck)

		deck = self.deck.copy()
		drawn_card = deck.pop()

		self.game_state = GameState(self.starts, deck, drawn_card)

		self.update_turn()

		if self.verbose:
			print_state(self.game_state)

	def step(self, probs, action):
		s, deck, drawn_card = self.game_state.take_action(action)
		new_turn = increment_var(self.game_state.turn, 1, len(self.current_players))

		self.game_state = GameState(new_turn, deck, drawn_card, old_history=self.game_state.history, s=s, prev_pairs=self.game_state.amount_pairs)

		if self.verbose:
			print_action(self, probs, action)

		self.update_turn()


class GameState():
	empty_history = (None,) * config.DEPTH
	empty_state = np.full(np.prod(GAME_DIMENSIONS), -1)
	empty_prev_pairs = 0

	def __init__(self, turn, deck, drawn_card, old_history=empty_history, s=empty_state, prev_pairs=empty_prev_pairs):
		self.turn = turn
		self.deck = deck
		self.drawn_card = drawn_card
		self.prev_pairs = prev_pairs
		self.history = old_history[1:] + (self,)
		self.s = s
		self.prev_pairs = prev_pairs

		self.done = self.check_game_over()

		amount_empty = len(np.where(self.s == -1)[0])
		self.first_card = amount_empty == len(s)
		self.replace_card = not amount_empty and not self.done

		self.legal_moves = self.get_legal_moves()
		self.amount_pairs = self.get_pairs()
		self.reward, self.input_reward = self.get_reward()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s) + tuple(self.deck) + (self.drawn_card,))

	def take_action(self, action):
		board = self.s.copy()
		deck = self.deck.copy()

		if action != np.prod(GAME_DIMENSIONS):
			board[action] = self.drawn_card

		drawn_card = deck.pop() if len(deck) else None

		return (board, deck, drawn_card)

	def get_legal_moves(self):
		if self.first_card:
			return list(range(np.prod(GAME_DIMENSIONS)))

		if self.replace_card:
			return list(range(MOVE_AMOUNT))

		if self.done:
			return []

		legal_moves = calculate_legal_moves(tuple(self.s))

		return legal_moves

	def check_game_over(self):
		return len(self.deck) == DECK_LENGTH - np.prod(GAME_DIMENSIONS) - 1 or self.drawn_card is None

	def get_pairs(self):
		sum_score = 0

		board = self.s.reshape(GAME_DIMENSIONS)
		for rowcol in [board, board.T]:
			for row in rowcol:
				row = row[np.where(row != -1)].tolist()
				sum_score += score_row(tuple(row))

		return sum_score
	
	def get_reward(self):
		if self.done:
			return (reward_transform(self.amount_pairs),) * 2
		else:
			return (intermediate_reward_transform(self.amount_pairs - self.prev_pairs), intermediate_reward_transform(self.amount_pairs))

	def generate_nn_pass(self, modify=False):
		if modify:
			rots = [0, 1, 2, 3]
			flips = [0, 1]
			suit_changes = [i * SUIT_LENGTH for i in range(SUIT_AMOUNT)]
		else:
			rots = [0]
			flips = [None]
			suit_changes = [0]

		nn_pass = []
		for suit_change in suit_changes:
			if suit_change:
				history = []
				for old_game_state in self.history:
					game_state = deepcopy(old_game_state)
					for var in [game_state.s, game_state.deck]:
						for i, card in enumerate(var):
							if card != -1:
								var[i] = int(increment_var(card, suit_change, DECK_LENGTH))
					game_state.drawn_card = int(increment_var(game_state.drawn_card, suit_change, DECK_LENGTH))
				
					history.append(game_state)
				history = tuple(history)
			else:
				history = self.history

			for rot in rots:
				for flip in flips:
					nn_pass.append(format_game_state(history, rot, flip))

		return nn_pass
