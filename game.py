import numpy as np


GAME_DIMENSIONS = (5, 5)
NN_INPUT_DIMENSIONS = (1, 27, 52)
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1
REPLACE_CARDS = 3


def generate_tutorial_game_state(node, mirror=False):
    game_state = []
    for i in range(1, 53):
        position = np.zeros(len(node.s))
        position[node.s == i] = 1
        game_state.append(position)
    game_state = np.array(game_state).reshape((25, 52)).tolist()

    deck = np.zeros(52)
    for card in node.deck: deck[card - 1] = 1
    game_state.append(deck.tolist())

    drawn_card = np.zeros(52)
    drawn_card[node.drawn_card - 1] = 1
    game_state.append(drawn_card.tolist())

    # board_history.append(np.array([[[node.player + 1]] * GAME_DIMENSIONS[1]] * GAME_DIMENSIONS[0]))
    # game_state = np.reshape(board_history, NN_INPUT_DIMENSIONS)
    # game_state = np.moveaxis(game_state, 0, 1)
    game_state = [[game_state]]

    if mirror:
        for flips in [0, 1, (0, 1)]: game_state.append([np.flip(game_state, flips).tolist()])

    return game_state


def check_index(board, index, checking_index, checking_func, multiplier_func, multiplier):
    if 0 <= checking_index < np.prod(GAME_DIMENSIONS):
        if np.floor(checking_index / GAME_DIMENSIONS[1]) == np.floor(index / GAME_DIMENSIONS[1]) + multiplier_func(multiplier):
            if checking_func(board[checking_index]):
                return True


def get_legal_moves(node):  # , all_moves):
    if not len(np.where(node.s != 0)[0]):
        return list(range(np.prod(GAME_DIMENSIONS)))

    if node.replace_card:
        return list(range(np.prod(GAME_DIMENSIONS))) + [25]

    legal_moves = []

    for index in np.where(node.s != 0)[0]:
        for checks, func in [[(GAME_DIMENSIONS[1] + 1, GAME_DIMENSIONS[1] - 1, GAME_DIMENSIONS[1]), lambda m: m], [(1,), lambda m: 0]]:
            for check in checks:
                for multiplier in [1, -1]:
                    checking_index = index + check * multiplier
                    if check_index(node.s, index, checking_index, lambda x: x == 0, func, multiplier):
                        legal_moves.append(checking_index)
                    """if 0 <= checking_index < np.prod(GAME_DIMENSIONS):
                        if checking_index not in legal_moves:
                            if np.floor(checking_index / 5) == np.floor(index / 5) + func(multiplier):
                                if node.s[checking_index] == 0:
                                    legal_moves.append(checking_index)"""

    return legal_moves


def get_card(value):
    suit = np.floor((value - 1) / 13)
    value = ((value  - 1) % 13) + 2
    return suit, value


def get_score(cards):
    values = [get_card(card)[1] for card in cards]
    
    histo_dict = {(1, 4): 20, (2, 3): 15, (1, 1, 3): 8, (1, 2, 2): 4, (1, 1, 1, 2): 2}

    histo = []
    for value in set(values):
        histo.append(values.count(value))

    histo = tuple(sorted(histo))
    if histo in histo_dict:
        return histo_dict[histo]

    färgrad = len(set(get_card(card)[0] for card in cards)) == 1
    stege = max(values) - min(values) == 4
    op = max(values) == 14
    
    score = 0

    if färgrad:
        score += 10
    if stege:
        score += 10
        if op:
            return 3 * score - 10
    
    return score
    

def check_game_over(node):
    if len(node.deck) == 52 - np.prod(GAME_DIMENSIONS) - REPLACE_CARDS - 1:
        score = 0
        # node.s = np.full((5, 5), 1)
        # node.s[0] = [13, 12, 11, 10, 8]
        board = node.s.reshape(GAME_DIMENSIONS)
        # print(print_board(board.flatten()))
        for rowcol in [board, board.T]:
            for row in rowcol:
                score += get_score(row)
        
        return score


def take_action(node, action):
    node_info = []
    for card in node.deck:
        board = node.s.copy()
        deck = node.deck.copy()

        if action != 25: board[action] = node.drawn_card

        deck.remove(card)

        node_info.append((board, deck, card))
    return node_info


def print_board(board):
    board = board.astype("<U4")
    board[board == "0.0"] = "---"
    suit_dict = {0: "sp", 1: "hj", 2: "ru", 3: "kl"}
    for i, pos in enumerate(board):
        if pos != "---":
            suit, value = get_card(float(pos))
            board[i] = f"{suit_dict[suit]}{int(value)}"
    return board.reshape(GAME_DIMENSIONS)


def print_values(values):
    print(f"Action values are: {[values[-1]]}\n{values[:-1].reshape(GAME_DIMENSIONS)}")
