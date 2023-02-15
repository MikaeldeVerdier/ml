import numpy as np

import environment

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


def inverse_format_card(card):
	suit_dict = {"sp": 0, "hj": 1, "ru": 2, "kl": 3}

	card_num = int(card[2:]) - 1
	card_num += environment.DECK_LENGTH / environment.SUIT_AMOUNT * suit_dict[card[:2]]

	return card_num


def print_state(state, probs, action):
	board = state.game_state.s.astype("<U4")

	board = np.array([format_card(float(pos)) if pos != "0.0" else "---" for pos in board])

	if probs is not None:
		print(f"Action values are: {[probs[-1]]}\n{np.round(probs[:-1], 8).reshape(environment.GAME_DIMENSIONS)}")
	
	print(f"Action taken by {state.player.get_name()} is: {action}")
	print(f"Position is:\n{board.reshape(environment.GAME_DIMENSIONS)}")

	if not state.game_state.done:
		print(f"Drawn card is: {format_card(state.game_state.drawn_card)}")
		print(f"Amount of cards left is now: {len(state.game_state.deck)}")
	else:
		print(f"Game over! The outcomes were: {state.game_state.scores}")


def string_to_tuple(s):
		a = s.replace(" ", "").replace("(", "").replace(")", "")
		b = a.split(',')
		res = tuple(int(el) for el in b)

		return res


def moving_average(data, n):
	return [np.mean(data[i:i + n]) for i in np.arange(0, len(data) - n + 1)]
