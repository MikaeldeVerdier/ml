import numpy as np

import config
from funcs import increment_turn, print_state, print_action, calculate_legal_moves, score_board

IN_A_ROW = 4

GAME_DIMENSIONS = (6, 7)
PLAYER_AMOUNT = 2
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (config.DEPTH * PLAYER_AMOUNT,), (PLAYER_AMOUNT,)]
MOVE_AMOUNT = GAME_DIMENSIONS[1]

REWARD_FACTOR = 20

def reward_transform(reward):
	return tuple(np.array(reward) * REWARD_FACTOR)


def inverse_reward_transform(transformed_reward):
	return int(transformed_reward / REWARD_FACTOR)


def results_transform(results):
	return results.count(1)


class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players
		self.epsilons = epsilons or np.full(np.shape(players), None)
		self.starts = starts - 1
		self.verbose = verbose

		self.players_turn = -1

	def step(self, probs, action):
		s = self.game_state.take_action(action)
		new_turn = increment_turn(self.game_state.turn, 1, len(self.current_players))
		self.game_state = GameState(new_turn, self.game_state.history, s)

		if self.verbose:
			print_action(self, probs, action)

		self.update_turn()

	def update_turn(self):
		self.player = self.current_players[self.game_state.turn]
		self.epsilon = self.epsilons[self.players_turn][self.game_state.turn]

	def reset(self):
		self.players_turn = increment_turn(self.players_turn, 1, len(self.players))
		self.current_players = self.players[self.players_turn]

		self.starts = increment_turn(self.starts, 1, len(self.current_players))

		empty_history = (None,) * config.DEPTH
		empty_state = np.full(np.prod(GAME_DIMENSIONS), -1)
		self.game_state = GameState(self.starts, empty_history, empty_state)

		self.update_turn()

		if self.verbose:
			print_state(self.game_state)


class GameState():
	def __init__(self, turn, history, s):
		self.turn = turn
		self.history = history[1:] + (self,)
		self.s = s

		self.last_turn = increment_turn(turn, -1, PLAYER_AMOUNT)

		self.legal_moves = self.get_legal_moves()
		self.scores = self.get_scores()
		self.done = self.check_game_over()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s))

	def take_action(self, action):
		board = self.s.copy()
		board[action] = self.turn

		return board

	def get_legal_moves(self):
		legal_moves = calculate_legal_moves(tuple(self.s))

		return legal_moves

	def get_scores(self):
		score = score_board(tuple(self.s))

		return reward_transform(score)

	def check_game_over(self):
		return self.scores != (0, 0)

	def generate_nn_pass(self, modify=False):
		game_state = self.history[-1]

		flips = [None, 0] if modify else [None]

		nn_pass = []
		for flip in flips:
			s = game_state.s if flip is None else np.flip(game_state.s.reshape(GAME_DIMENSIONS), flip).flatten()

			nn_pass.append([[] for _ in range(len(NN_INPUT_DIMENSIONS))])
			for depth in range(config.DEPTH):
				for player in list(range(PLAYER_AMOUNT)):
					state = []

					position = np.zeros(len(s))
					position[s == player] = 1
					state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-1]).tolist())

					nn_pass[-1][0] += state

					nn_pass[-1][1] += [int(player == game_state.turn)]

					if depth != config.DEPTH - 1:
						if self.history[-depth - 2]:
							game_state = self.history[-depth - 2]
						else:
							for _ in range(config.DEPTH - depth - 1):
								nn_pass[-1][0].append(np.full(NN_INPUT_DIMENSIONS[0][:-1], -1))
							break

			nn_pass[-1][0] = np.moveaxis(nn_pass[-1][0], 0, -1).tolist()

		return nn_pass
