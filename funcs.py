import numpy as np

import environment

def linear_wrapper_func(start, end, duration):
	def linear_inner_func(x):
		b = start
		m = (start - end) / duration

		return (max if m < 0 else min)(end, m * x + b)

	linear_inner_func.start = start

	return linear_inner_func


def increment_turn(turn, increment, length):
	return (turn + increment) % length


def get_card(value):
	suit = (value - 1) // (environment.DECK_LENGTH / environment.SUIT_AMOUNT)
	value = ((value - 1) % (environment.DECK_LENGTH / environment.SUIT_AMOUNT)) + 2

	return suit, value


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


def order_moves(moves):
	return np.array(moves).reshape(environment.GAME_DIMENSIONS)[::-1].reshape(np.prod(environment.GAME_DIMENSIONS))


def format_move(move):
	if move != np.prod(environment.GAME_DIMENSIONS):
		dim1 = move % environment.GAME_DIMENSIONS[0] + 1
		dim2 = environment.GAME_DIMENSIONS[1] - move // environment.GAME_DIMENSIONS[1]
		return (dim1, dim2)
	else:
		return (0,)


def get_move(moves):
	user_move = None
	while user_move not in moves:
		print(f"Legal moves for you are: {moves}")
		try:
			user_move = string_to_tuple(input("Make your move: "))
		except ValueError:
			print("Please enter a valid move.")
	
	return user_move


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
