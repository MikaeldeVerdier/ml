import numpy as np

import environment
import config

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
def increment_var(var, increment, limit):
	return (var + increment) % limit


@cache(10000)
def calculate_legal_moves(board):
	legal_moves = []

	for index in np.where(np.array(board) != -1)[0]:
		for multiplier in [-1, 0, 1]:
			for add_on in [-1, 0, 1]:
				if not multiplier and not add_on:
					continue

				check_index = index + environment.GAME_DIMENSIONS[1] * multiplier + add_on

				if check_index not in legal_moves and 0 <= check_index < np.prod(environment.GAME_DIMENSIONS) and board[check_index] == -1:
					row_diff = check_index // environment.GAME_DIMENSIONS[1] - index // environment.GAME_DIMENSIONS[1]
					if row_diff == multiplier:
						legal_moves.append(check_index)

	return legal_moves


@cache(10000)
def score_row(row):
	if not row:
		return 0

	sum_score = 0

	_, values = tuple(zip(*[get_card(card) for card in row]))
	values = sorted(values)

	histo_scoring_dict = {(2,): 1}

	histo = tuple(sorted([values.count(value) for value in set(values)]))

	max_num_occs = {num: comb.count(num) for comb in histo_scoring_dict.keys() for num in comb}
	for comb, score in list(histo_scoring_dict.items()):
		comb_count = list(zip(*[(histo.count(num) // comb.count(num), max_num_occs[num]) for num in comb]))

		if min(comb_count[0]) and min(comb_count[1]) > 0:
			for num in set(comb):
				max_num_occs[num] -= min(comb_count[0])

			sum_score += min(comb_count[0]) * score

	return sum_score


@cache(10000)
def format_state(board):
	shape = (environment.NN_INPUT_DIMENSIONS[0][-1],) + environment.NN_INPUT_DIMENSIONS[0][:-2]
	binary_map = np.reshape([(np.array(board) == i).astype(np.int32) for i in range(environment.DECK_LENGTH)], shape)

	return np.moveaxis(binary_map, 0, -1).tolist()


@cache(100000)
def format_game_state(history, rot, flip):
	game_state = history[-1]

	nn_pass = [[] for _ in range(len(environment.NN_INPUT_DIMENSIONS))]

	for depth in range(config.DEPTH):
		state = game_state.s.reshape(environment.GAME_DIMENSIONS)
		if rot != 0:
			state = np.rot90(state, k=rot)
		if flip is not None:
			state = np.flip(state, axis=flip)

		state = state.flatten()

		formatted_state = format_state(tuple(state))
		nn_pass[0].append(formatted_state)

		deck = np.zeros(environment.DECK_LENGTH, dtype=np.int32)
		deck[np.array(game_state.deck)] = 1
		nn_pass[1].append(deck.tolist())

		drawn_card = np.zeros(environment.DECK_LENGTH, dtype=np.int32)
		drawn_card[game_state.drawn_card] = 1
		nn_pass[2].append(drawn_card.tolist())

		nn_pass[3].append([game_state.reward])

		if depth != config.DEPTH - 1:
			if history[-depth - 2]:
				game_state = history[-depth - 2]
			else:
				for i, func in enumerate([np.zeros, np.ones, np.zeros]):
					empty_dim = func(np.shape(nn_pass[i][-1])).tolist()
					nn_pass[i] += [empty_dim] * (config.DEPTH - depth - 1)

				break

	nn_pass[0] = np.moveaxis(nn_pass[0], 0, -2).tolist()

	return nn_pass


@cache()
def get_card(value):
	suit, value = divmod(value, environment.SUIT_LENGTH)

	return suit, value + 2


@cache()
def format_card(card):
	suit_dict = {0: "kl", 1: "ru", 2: "hj", 3: "sp"}
	suit, value = get_card(card)

	return f"{suit_dict[suit]}{int(value)}"


def print_state(state):
	board = state.s.astype("<U4")
	board = [format_card(float(cell)) if cell != "-1" else "---" for cell in board]

	print(f"Position is:\n{np.reshape(board, environment.GAME_DIMENSIONS)}")

	if not state.done:
		if state.reward:
			print(f"The reward for this state is: {state.reward}")

		print(f"Drawn card is: {format_card(state.drawn_card)}")
		print(f"Amount of cards left is: {len(state.deck)}\n")
	else:
		print(f"Game over! The outcomes were: {state.reward}\n")


def print_action(env, probs, action):
	if probs is not None:
		print(f"Action values are: {[probs[-1]]}\n{np.round(probs[:-1], 8).reshape(environment.GAME_DIMENSIONS)}")

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
def string_to_tuple(string):
	split_string = string.replace(" ", "").replace("(", "").replace(")", "").split(",")
	result = tuple(int(letter) for letter in split_string)

	return result


"""def moving_average(data, n):
	return [np.mean(data[i:i + n]) for i in np.arange(0, len(data) - n + 1)]"""
