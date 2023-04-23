import numpy as np

import config
from funcs import cache, increment_turn, print_state, print_action, calculate_legal_moves

IN_A_ROW = 4

GAME_DIMENSIONS = (6, 7)
PLAYER_AMOUNT = 2
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (config.DEPTH * PLAYER_AMOUNT,), (PLAYER_AMOUNT,)]
MOVE_AMOUNT = GAME_DIMENSIONS[1]

def reward_transform(reward):
	return reward


def inverse_reward_transform(transformed_reward):
	return transformed_reward


def results_transform(results):
	return len(results)


class Environment:
	def __init__(self, players, epsilons=None, starts=0, verbose=False):
		self.players = players
		self.epsilons = epsilons or np.full(np.shape(players), None)
		self.starts = starts - 1
		self.verbose = verbose
		
		self.players_turn = -1

		self.players_names = np.reshape([agent.full_name for agent in np.concatenate(players)], np.shape(players)).tolist()

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

		self.game_state = GameState(self.starts)

		self.update_turn()

		if self.verbose:
			print_state(self.game_state)

	def step(self, probs, action):
		s = self.game_state.take_action(action)
		new_turn = increment_turn(self.game_state.turn, 1, len(self.current_players))

		self.game_state = GameState(new_turn, old_history=self.game_state.history, s=s)

		if self.verbose:
			print_action(self, probs, action)

		self.update_turn()


class GameState():
	empty_history = (None,) * config.DEPTH
	empty_state = np.full(np.prod(GAME_DIMENSIONS), -1)

	def __init__(self, turn, old_history=empty_history, s=empty_state):
		self.turn = turn
		self.history = old_history[1:] + (self,)
		self.s = s

		self.last_turn = increment_turn(turn, -1, PLAYER_AMOUNT)

		self.reward = self.get_reward()
		self.done = self.check_game_over()
		self.legal_moves = self.get_legal_moves()

	def __hash__(self):
		return hash(self.history[:-1] + tuple(self.s))

	def take_action(self, action):
		board = self.s.copy()
		board[action] = self.turn

		return board

	def get_legal_moves(self):
		legal_moves = calculate_legal_moves(tuple(self.s)) if not self.done else []

		return legal_moves
	
	def score_board(self):
		for player in range(PLAYER_AMOUNT):
			for checks in [[[GAME_DIMENSIONS[0], GAME_DIMENSIONS[1] - IN_A_ROW + 1], list(range(IN_A_ROW))], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1]], [element * GAME_DIMENSIONS[1] for element in list(range(IN_A_ROW))]], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1] - IN_A_ROW + 1], [element * (GAME_DIMENSIONS[1] - 1) + IN_A_ROW - 1 for element in list(range(IN_A_ROW))]], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1] - IN_A_ROW + 1], [element * (GAME_DIMENSIONS[1] + 1) for element in list(range(IN_A_ROW))]]]:
				for i in range(checks[0][0]):
					for i2 in range(checks[0][1]):
						pos = [self.s[i * GAME_DIMENSIONS[1] + i2 + i3] == player for i3 in checks[1]]
						if pos.count(True) == IN_A_ROW:
							return 1 if self.turn != player else -1

		if not np.count_nonzero(self.s == -1):
			return 1e-5

		return 0

	def get_reward(self):
		reward = self.score_board()

		return reward_transform(reward)

	def check_game_over(self):
		return self.reward != 0

	@cache(100000)
	def format_game_state(self, flip):
		game_state = self.history[-1]

		nn_pass = [[] for _ in range(len(NN_INPUT_DIMENSIONS))]

		for depth in range(config.DEPTH):
			s = np.rot90(game_state.s.reshape(GAME_DIMENSIONS), k=flip).flatten()

			for player in list(range(PLAYER_AMOUNT)):
				state = []

				position = np.zeros(len(s))
				position[s == player] = 1
				state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-1]).tolist())

				nn_pass[0] += state

				nn_pass[1] += [int(player == game_state.turn)]

				if depth != config.DEPTH - 1:
					if self.history[-depth - 2]:
						game_state = self.history[-depth - 2]
					else:
						for _ in range(config.DEPTH - depth - 1):
							nn_pass[0].append(np.full(NN_INPUT_DIMENSIONS[0][:-1], -1))
						break

		nn_pass[0] = np.moveaxis(nn_pass[0], 0, -1).tolist()

		return nn_pass

	def generate_nn_pass(self, modify=False):
		flips = [0, 1, 2, 3] if modify else [0]

		nn_pass = []
		for flip in flips:
			nn_pass.append(self.format_game_state(flip))

		return nn_pass
