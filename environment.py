import numpy as np
import random

import config
from funcs import increment_turn, get_card, format_card

GAME_DIMENSIONS = (2, 2)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (52 * config.DEPTH,), (52 * config.DEPTH,), (52 * config.DEPTH,)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1
REPLACE_CARDS = 5
GAME_LENGTH = np.prod(GAME_DIMENSIONS) + REPLACE_CARDS
REWARD_FACTOR = 0.2
REWARD_AVERAGE = True

INVERSE_REWARD_TRANSFORM = lambda outcome: int(outcome / REWARD_FACTOR)
GAME_ADD = lambda left, og_games: np.ceil(left / (GAME_LENGTH * 16) % og_games)

class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players  # [offset_array(game_players, len(game_players)) for game_players in players]  # offset_array(players, 2) axis=-1?
		self.epsilons = epsilons or np.full(np.array(players).shape, None)
		self.starts = starts - 1
		self.verbose = verbose

		self.players_turn = -1

		self.deck = list(range(1, 53))

	def step(self, probs, action):
		s, deck, drawn_card = self.game_state.take_action(action)
		self.game_state = GameState(increment_turn(self.game_state.turn, 1, len(self.current_players)), self.game_state.history, s, deck, drawn_card)

		if self.verbose:
			self.print_state(probs, action)

		self.update_player()

	def update_player(self):
		self.player = self.current_players[self.game_state.turn]
		self.epsilon = self.epsilons[self.players_turn][self.game_state.turn]

	def reset(self):
		self.players_turn = increment_turn(self.players_turn, 1, len(self.players))
		self.current_players = self.players[self.players_turn]

		self.starts = increment_turn(self.starts, 1, len(self.players))

		if self.players_turn == 0:
			random.shuffle(self.deck)

		deck = self.deck.copy()
		drawn_card = deck.pop()

		self.game_state = GameState(self.starts, (None,) * config.DEPTH, np.zeros(np.prod(GAME_DIMENSIONS)), deck, drawn_card)

		self.update_player()

	def print_state(self, probs, action):
		board = self.game_state.s.astype("<U4")

		board = np.array([format_card(float(pos)) if pos != "0.0" else "---" for pos in board])

		if probs is not None:
			print(f"Action values are: {[probs[-1]]}\n{np.round(probs[:-1], 8).reshape(GAME_DIMENSIONS)}")
		
		print(f"Action taken by {self.player.get_name()} is: {action}")
		print(f"Position is:\n{board.reshape(GAME_DIMENSIONS)}")

		if self.game_state.outcome == (None,) * len(self.current_players):
			print(f"Drawn card is: {format_card(self.game_state.drawn_card)}")
			print(f"Amount of cards left is now: {len(self.game_state.deck)}")
		else:
			print(f"Game over! The outcomes were: {self.game_state.outcome}")


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
		return len(self.deck) == 51 - GAME_LENGTH

	def get_scores(self):
		if self.done:
			sum_score = 0

			board = self.s.reshape(GAME_DIMENSIONS)
			for rowcol in [board, board.T]:
				for row in rowcol:
					suits, values = tuple(zip(*[get_card(card) for card in row]))
					values = sorted(values)
					
					histo_dict = {(2,): 10}

					histo = tuple(sorted([values.count(value) for value in set(values)]))

					if histo in histo_dict:
						sum_score += histo_dict[histo]

					färgrad = len(set(suits)) == 1
					stege = values[-1] - values[0] == len(row) - 1 or values == list(range(1, len(row))) + [14]

					score = 0

					scores = [5, 7, 15, 20]  # [{färgrad}, {stege}, {op-stege}, {royal straight flush}]

					if färgrad:
						score += scores[0]
					if stege:
						score += scores[1]

						if values[-2] == 13:
							op_dict = {scores[1]: scores[2], scores[0] + scores[1]: scores[3]}
							sum_score += op_dict[score]
				
			return (sum_score * REWARD_FACTOR,)

		return (0,)

	def generate_nn_pass(self, modify=False):
		game_state = self.history[-1]

		if modify:
			flips = [None, 0, 1, (0, 1)]
			suit_changes = [0, 13, 26, 39]
		else:
			flips = [None]
			suit_changes = [0]

		nn_pass = []
		for flip in flips:
			s = game_state.s if flip is None else np.flip(game_state.s.reshape(GAME_DIMENSIONS), flip).flatten()
			
			for suit_change in suit_changes:
				nn_pass.append([[], [], []])
				for depth in range(config.DEPTH):
					de = game_state.deck
					dr = [game_state.drawn_card]
					for var in [s, de, dr]:
						for i, card in enumerate(var):
							if card != 0:
								var[i] += suit_change
							if var[i] > 52:
								var[i] -= 52

					state = []
					for i in range(1, 53):
						position = np.zeros(len(s))
						position[s == i] = 1
						state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-1]))

					state = np.moveaxis(state, 0, -1).tolist()
					nn_pass[-1][0] += state

					deck = np.zeros(52)
					for card in de: deck[card - 1] = 1
					nn_pass[-1][1] += deck.tolist()

					drawn_card = np.zeros(52)
					if dr[0] != 0:
						drawn_card[dr[0] - 1] = 1
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
