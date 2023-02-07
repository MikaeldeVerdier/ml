import numpy as np

def offset_array(arr, index):
	return arr[index - 1:] + arr[:index - 1]


def increment_turn(turn, increment, length):
	return (turn + increment) % length


def get_card(value):
	suit = (value - 1) // 13
	value = ((value - 1) % 13) + 2
	return suit, value


def format_card(card):
	suit_dict = {0: "sp", 1: "hj", 2: "ru", 3: "kl"}
	suit, value = get_card(card)

	return f"{suit_dict[suit]}{int(value)}"


def inverse_format_card(card):
	suit_dict = {"sp": 0, "hj": 1, "ru": 2, "kl": 3}

	card_num = int(card[2:]) - 1
	card_num += 13 * suit_dict[card[:2]]

	return card_num


def string_to_tuple(s):
		a = s.replace(" ", "").replace("(", "").replace(")", "")
		b = a.split(',')
		res = tuple(int(el) for el in b)

		return res


def moving_average(data, n):
	return [np.mean(data[i:i + n]) for i in np.arange(0, len(data) - n + 1)]
