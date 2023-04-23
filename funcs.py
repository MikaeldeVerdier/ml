import numpy as np

import environment

def cache(max_length=5000):
	def cache_dec(f):
		cache = {}

		def caching(*args):
			cache_ref = hash(args)

			if cache_ref in cache:
				return cache[cache_ref]

			v = f(*args)
			cache[cache_ref] = v

			if len(cache) >= max_length:
				cache.popitem()

			return v

		caching.cache_clear = cache.clear

		return caching

	return cache_dec


def linear_wrapper_func(start, end, duration, max_length=1, use_cache=True):
	intercept = start
	slope = (end - start) / duration

	func = max if slope < 0 else min

	def linear_inner_func(x):
		return func(end, slope * x + intercept)
	
	if use_cache:
		linear_inner_func = cache(max_length)(linear_inner_func)

	linear_inner_func.start = start

	return linear_inner_func


@cache()
def increment_turn(turn, increment, length):
	return (turn + increment) % length


@cache(10000)
def calculate_legal_moves(board):
	legal_moves = []
	for dim1 in range(environment.GAME_DIMENSIONS[1]):
		for dim2 in range(environment.GAME_DIMENSIONS[0]):
			if board[dim1 + dim2 * environment.GAME_DIMENSIONS[1]] != -1:
				if dim2 != 0:
					legal_moves.append(dim1 + (dim2 - 1) * environment.GAME_DIMENSIONS[1])
				break
		else:
			legal_moves.append(dim1 + dim2 * environment.GAME_DIMENSIONS[1])

	return legal_moves


@cache()
def format_cell(player):
	player_dict = {"-1": "-", "0": "X", "1": "O"}

	return player_dict[player]


def print_state(state):
	board = state.s.astype("<U4")
	board = np.array([format_cell(cell) for cell in board])

	print(f"Position is:\n{board.reshape(environment.GAME_DIMENSIONS)}")

	if state.done:
		print(f"Game over! The outcomes were: {state.reward}\n")


def print_action(env, probs, action):
	if probs is not None:
		print(f"Action values are:\n{np.round(probs, 6).reshape(environment.GAME_DIMENSIONS)}")

	print(f"Action taken by {env.player.full_name} is: {action}")
	print_state(env.game_state)


@cache()
def order_moves(moves):
	return sorted(moves, key=lambda x: (x[0], x[1]) if x else (0, 0))


@cache()
def format_move(move):
	if move != np.prod(environment.GAME_DIMENSIONS):
		dim1 = move % environment.GAME_DIMENSIONS[0] + 1
		dim2 = environment.GAME_DIMENSIONS[1] - move // environment.GAME_DIMENSIONS[1]

		return (dim1, dim2)
	else:
		return 0


def get_move(moves):
	user_move = None
	while user_move not in moves:
		print(f"Legal moves for you are: {moves}")
		try:
			user_move = string_to_tuple(input("Make your move: "))
		except ValueError:
			print("Please enter a valid move.")
	
	return user_move


@cache()
def string_to_tuple(s):
		a = s.replace(" ", "").replace("(", "").replace(")", "")
		b = a.split(',')
		res = tuple(int(el) for el in b)

		return res


"""def moving_average(data, n):
	return [np.mean(data[i:i + n]) for i in np.arange(0, len(data) - n + 1)]"""
