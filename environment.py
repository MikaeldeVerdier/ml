import numpy as np
import random

import config
from funcs import increment_turn, can_move, print_state

DECK_LENGTH = 52
SUIT_AMOUNT = 4

GAME_DIMENSIONS = (DECK_LENGTH,)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (DECK_LENGTH * config.DEPTH,), (DECK_LENGTH * config.DEPTH,)]
MOVE_AMOUNT = DECK_LENGTH * 2 + 1
REWARD_FACTOR = 0.02
REWARD_AVERAGE = True

INVERSE_REWARD_TRANSFORM = lambda outcome: int(outcome / REWARD_FACTOR)

class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players  # [offset_array(game_players, len(game_players)) for game_players in players]  # offset_array(players, 2) axis=-1?
		self.epsilons = epsilons or np.full(np.array(players).shape, None)
		self.starts = starts - 1
		self.verbose = verbose

		self.players_turn = -1

		self.deck = list(range(1, DECK_LENGTH + 1))

	def step(self, probs, action):
		s, deck = self.game_state.take_action(action)
		self.game_state = GameState(increment_turn(self.game_state.turn, 1, len(self.current_players)), self.game_state.history, s, deck)

		if self.verbose:
			print_state(self, probs, action)

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

		self.game_state = GameState(self.starts, (None,) * config.DEPTH, np.zeros(np.prod(GAME_DIMENSIONS)), deck)

		self.update_player()


class GameState():
	def __init__(self, turn, history, s, deck):
		self.turn = turn
		self.history = history[1:] + (self,)
		self.s = s
		self.deck = deck

		self.replace_card = not len(np.where(self.s == 0)[0])
		self.legal_moves = self.get_legal_moves()
		self.done = self.check_game_over()
		self.scores = self.get_scores()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s) + tuple(self.deck))

	def take_action(self, action):
		if action == 0:
			board = self.s.copy()
			deck = self.deck.copy()
			card = deck.pop()

			index = np.where(board == 0)[0][0]
			board[index] = card
		else:
			board = self.s.copy()

			index = int(np.ceil(action / 2)) - 1
			kind = 3 - 2 * (action % 2)
			board[index - kind] = board[index]
			board = np.delete(board, index)
			board = np.append(board, 0)

			deck = self.deck.copy()

		return board, deck

	def get_legal_moves(self):
		legal_moves = []
		checks = [1]
		for i, pos in enumerate(self.s[1:], 1):
			if pos == 0: break
			if i == 3: checks += [3]
			for check in checks:
				if can_move(pos, self.s[i - check]): legal_moves.append(int(2 * i + 0.5 + 0.5 * check))

		return legal_moves if len(legal_moves) or not len(self.deck) else [0]

	def check_game_over(self):
		return not len(self.deck) and not len(self.legal_moves)

	def get_scores(self):
		return ((DECK_LENGTH - np.where(self.s == 0)[0][0]) * REWARD_FACTOR,) if self.done else (0,)			

	def generate_nn_pass(self, modify=False):
		game_state = self.history[-1]

		suit_changes = [0] if not modify else [0, 13, 26, 39]

		nn_pass = []
		for suit_change in suit_changes:
			nn_pass.append([[], []])
			for depth in range(config.DEPTH):
				s = game_state.s.copy()
				de = game_state.deck
				for var in [s, de]:
					for i, card in enumerate(var):
						if card != 0:
							var[i] = int((var[i] + suit_change - 1) % DECK_LENGTH + 1)

				state = []
				for i in range(1, DECK_LENGTH + 1):
					position = np.zeros(len(s))
					position[s == i] = 1
					state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-1]))

				state = np.moveaxis(state, 0, -1).tolist()
				nn_pass[-1][0] += state

				deck = np.zeros(DECK_LENGTH)
				deck[np.array(de, dtype=np.int32) - 1] = 1
				nn_pass[-1][1] += deck.tolist()

				if depth != config.DEPTH -1:
					if self.history[-depth - 2]:
						game_state = self.history[-depth - 2]
					else:
						for _ in range(config.DEPTH - depth - 1):
							for i, func in enumerate([np.zeros, np.ones]):
								nn_pass[-1][i].append(func(NN_INPUT_DIMENSIONS[i][:-1]))
						break

		return nn_pass
