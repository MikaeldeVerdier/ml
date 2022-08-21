import numpy as np
import config

GAME_DIMENSIONS = (52,)
MOVE_AMOUNT = GAME_DIMENSIONS[0] * 2 + 1
SUIT_LENGTH = GAME_DIMENSIONS[0]/4


def generate_tutorial_game_state(nodes):
    nodes = (None,) * (config.DEPTH - len(nodes)) + nodes
    node = nodes[-1]
    board_history = []
    for depth in range(config.DEPTH):
        for i in range(1, 53):
            position = np.zeros(len(node.s))
            position[node.s == i] = 1
            board_history.append(position)
        if nodes[-depth - 1]: node = nodes[-depth - 1]
    # board_history.append(np.array([[[node.player + 1]] * GAME_DIMENSIONS[1]] * GAME_DIMENSIONS[0]))
    # game_state = np.reshape(board_history, (GAME_DIMENSIONS + (config.DEPTH * 2,)))
    return board_history


def mirror_board(board):
    return [board[x + z * GAME_DIMENSIONS[1]] for z in range(GAME_DIMENSIONS[0]) for x in range(GAME_DIMENSIONS[1] - 1, -1, -1)]


def get_legal_moves(node):  # , all_moves):
    legal_moves = [0] if len(node.deck) else []
    checks = [1]
    for i, pos in enumerate(node.s[1:], 1):
        if pos == 0:
            break
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
    if not node.deck and not len(get_legal_moves(node)):
        if node.s[1] == 0:
            pass
        return 1 / np.where(node.s == 0)[0][0]


def make_move(node, move):
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
