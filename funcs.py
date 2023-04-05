import numpy as np

import environment

def cache_dec(f):
	cache = {}

	def caching(*args):
		cache_ref = hash(args)
		if cache_ref in cache:
			return cache[cache_ref]
		v = f(*args)
		cache[cache_ref] = v
		return v

	caching.cache_clear = cache.clear

	return caching


def linear_wrapper_func(start, end, duration, use_cache=True):
	def linear_inner_func(x):
		b = start
		m = (end - start) / duration

		return (max if m < 0 else min)(end, m * x + b)
	
	if use_cache:
		linear_inner_func = cache_dec(linear_inner_func)

	linear_inner_func.start = start

	return linear_inner_func


@cache_dec
def increment_turn(turn, increment, length):
	return (turn + increment) % length


@cache_dec
def calculate_legal_moves(board):
	legal_moves = []

	for index in np.where(np.array(board) != 0)[0]:
		for multiplier in [-1, 0, 1]:
			for add_on in [-1, 0, 1]:
				if not multiplier and not add_on:
					continue

				check_index = index + environment.GAME_DIMENSIONS[1] * multiplier + add_on

				row_diff = check_index // environment.GAME_DIMENSIONS[1] - index // environment.GAME_DIMENSIONS[1]
				if check_index not in legal_moves and 0 <= check_index < np.prod(environment.GAME_DIMENSIONS) and not board[check_index] and row_diff == multiplier:
					legal_moves.append(check_index)

	return legal_moves


@cache_dec
def score_row(row):
	sum_score = 0
	suits, values = tuple(zip(*[get_card(card) for card in row]))
	values = sorted(values)

	histo_dict = {(4,): 20, (3, 2): 15, (3,): 8, (2,): 2}

	histo = tuple(sorted([values.count(value) for value in set(values)]))

	maxes = {num: comb.count(num) for comb in histo_dict.keys() for num in comb}
	for key, value in list(histo_dict.items()):
		key_count = list(zip(*[(histo.count(val) // key.count(val), maxes[val]) for val in key]))

		if min(key_count[0]) and min(key_count[1]) > 0:
			for val in set(key):
				maxes[val] -= min(key_count[0])

			sum_score += min(key_count[0]) * value

	is_flush = len(set(suits)) == 1

	low_ace_straight = list(range(1, len(row))) + [environment.SUIT_LENGTH + 1]
	is_straight = values[-1] - values[0] == len(row) - 1 or values == low_ace_straight

	if is_flush:
		sum_score += 10
	if is_straight:
		if values[-2] == environment.SUIT_LENGTH:
			sum_score += 40 if is_flush else 20
		else:
			sum_score += 10

	return sum_score


@cache_dec
def format_state(board):
	shape = (environment.NN_INPUT_DIMENSIONS[0][-1],) +  environment.NN_INPUT_DIMENSIONS[0][:-2]
	binary_map = np.reshape([(np.array(board) == i).astype(int) for i in range(1, environment.DECK_LENGTH + 1)], shape)

	return np.moveaxis(binary_map, 0, -1).tolist()


@cache_dec
def get_card(value):
	suit, value = divmod(value - 1, environment.SUIT_LENGTH)

	return suit, value + 2


@cache_dec
def format_card(card):
	suit_dict = {0: "sp", 1: "hj", 2: "ru", 3: "kl"}
	suit, value = get_card(card)

	return f"{suit_dict[suit]}{int(value)}"


def print_state(state):
	board = state.s.astype("<U4")
	board = np.array([format_card(float(cell)) if cell != "0.0" else "---" for cell in board])

	print(f"Position is:\n{board.reshape(environment.GAME_DIMENSIONS)}")

	if not state.done:
		print(f"Drawn card is: {format_card(state.drawn_card)}")
		print(f"Amount of cards left is: {len(state.deck)}\n")
	else:
		print(f"Game over! The outcomes were: {state.scores}\n")


def print_move(env, probs, action):
	if probs is not None:
		print(f"Action values are: {[probs[-1]]}\n{np.round(probs[:-1], 8).reshape(environment.GAME_DIMENSIONS)}")

	print(f"Action taken by {env.player.full_name} is: {action}")
	print_state(env.game_state)


@cache_dec
def order_moves(moves):
	return sorted(moves, key=lambda x: (x[0], x[1]) if x else (0, 0))


@cache_dec
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


@cache_dec
def string_to_tuple(s):
		a = s.replace(" ", "").replace("(", "").replace(")", "")
		b = a.split(',')
		res = tuple(int(el) for el in b)

		return res


"""def moving_average(data, n):
	return [np.mean(data[i:i + n]) for i in np.arange(0, len(data) - n + 1)]"""


"""def inverse_format_card(card):
	suit_dict = {"sp": 0, "hj": 1, "ru": 2, "kl": 3}

	card_num = int(card[2:]) - 1
	card_num += environment.DECK_LENGTH / environment.SUIT_AMOUNT * suit_dict[card[:2]]

	return card_num"""
