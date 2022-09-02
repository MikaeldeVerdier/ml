import numpy as np


GAME_DIMENSIONS = (52,)
NN_INPUT_DIMENSIONS = GAME_DIMENSIONS + (2,) + (1,)
MOVE_AMOUNT = GAME_DIMENSIONS[0] * 2 + 1
SUIT_LENGTH = GAME_DIMENSIONS[0]/4


def generate_tutorial_game_state(node):
    deck = np.zeros(GAME_DIMENSIONS[0])
    for card in node.deck: deck[card - 1] = 1

    game_state = np.moveaxis([np.array(node.s) / GAME_DIMENSIONS[0], deck], 0, 1)
    return game_state


def get_legal_moves(node):  # , all_moves):
    legal_moves = [0] if len(node.deck) else []
    checks = [1]
    for i, pos in enumerate(node.s[1:], 1):
        if pos == 0: break
        if i == 3: checks += [3]
        for check in checks:
            if can_move(pos, node.s[i - check]): legal_moves.append(int(2 * i + 0.5 + 0.5 * check))

    return legal_moves


def get_card(value):
    suit = np.floor((value - 1) / SUIT_LENGTH)
    value %= SUIT_LENGTH
    return suit, value


def can_move(card1, card2):
    card1 = get_card(card1)
    card2 = get_card(card2)
    for i in range(2):
        if card1[i] == card2[i]: return True
    return False


def check_game_over(node):
    if not node.deck and not len(get_legal_moves(node)): return 1 / np.where(node.s == 0)[0][0]
        # return -0.1 * np.where(node.s == 0)[0][0] + 5.2


def take_action(node, move):
    node_info = []
    if move == 0:
        for card in node.deck:
            board = node.s.copy()
            deck = node.deck.copy()

            index = np.where(board == 0)[0][0]
            board[index] = card
            deck.remove(card)

            node_info.append((board, deck))
        # node.deck = np.delete(node.deck, rand_index)
    else:
        board = node.s.copy()

        index = int(np.ceil(move / 2)) - 1
        kind = 3 - 2 * (move % 2)
        board[index - kind] = board[index]
        board = np.delete(board, index)
        board = np.append(board, 0)

        node_info.append((board, node.deck.copy()))
    return node_info


def print_board(board):
    board = board.astype("<U4")
    board[board == "0.0"] = "-"
    suit_dict = {0: "sp", 1: "hj", 2: "ru", 3: "kl"}
    for i, pos in enumerate(board):
        if pos != "-":
            suit, value = get_card(float(pos))
            board[i] = f"{suit_dict[suit]}{int(value % SUIT_LENGTH) + 1}"
    return board.reshape(GAME_DIMENSIONS)


def print_values(values):
    return values.reshape(MOVE_AMOUNT)
