import numpy as np
import config

GAME_DIMENSIONS = (5, 5)
NN_INPUT_DIMENSIONS = [GAME_DIMENSIONS + (52, config.DEPTH), (52, config.DEPTH), (52, config.DEPTH)]
MOVE_AMOUNT = np.prod(GAME_DIMENSIONS) + 1
REPLACE_CARDS = 3

def generate_game_states(history, t):
    data = history[:t + 1]
    game_states = ({"state": None},) * (config.DEPTH - len(data)) + tuple(data)[-config.DEPTH:]
    game_states = tuple([game_state["state"] for game_state in game_states])

    return game_states


def generate_nn_pass(game_states, mirror=False):
    game_state = game_states[-1]

    if mirror:
        flips = [None, 0, 1, (0, 1)]
        suit_changes = [0, 13, 26, 39]
    else:
        flips = [None]
        suit_changes = [0]

    nn_pass = []
    for flip in flips:
        s = game_state.s if flip is None else np.flip(game_state.s.reshape(GAME_DIMENSIONS), flip).flatten()
        
        for suit_change in suit_changes:
            nn_pass.append([[], [], []])
            for depth in range(config.DEPTH):
                de = game_state.deck
                dr = [game_state.drawn_card]
                for var in [s, de, dr]:
                    for i, card in enumerate(var):
                        if card != 0: var[i] += suit_change
                        if var[i] > 52: var[i] -= 52

                state = []
                for i in range(1, 53):
                    position = np.zeros(len(s))
                    position[s == i] = 1
                    state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][:-2]))

                state = np.moveaxis(state, 0, -1).tolist()
                nn_pass[-1][0].append(state)

                deck = np.zeros(52)
                for card in de: deck[card - 1] = 1
                nn_pass[-1][1].append(deck.tolist())

                drawn_card = np.zeros(52)
                drawn_card[dr[0] - 1] = 1
                nn_pass[-1][2].append(drawn_card.tolist())

                if game_states[-depth - 1]: game_state = game_states[-depth - 1]
            
            nn_pass[-1] = [np.array(np.moveaxis(dim, 0, -1)).tolist() for dim in nn_pass[-1]]
            # nn_pass[-1] = [np.array(np.moveaxis(dim, 0, -1), dtype=np.int32).tolist() for dim in nn_pass[-1]]

    return nn_pass


def check_index(board, index, checking_index, checking_func, multiplier_func, multiplier):
    if 0 <= checking_index < np.prod(GAME_DIMENSIONS):
        if np.floor(checking_index / GAME_DIMENSIONS[1]) == np.floor(index / GAME_DIMENSIONS[1]) + multiplier_func(multiplier):
            return checking_func(board[checking_index])


def get_legal_moves(game_state):  # , all_moves):
    if not len(np.where(game_state.s != 0)[0]): return list(range(np.prod(GAME_DIMENSIONS)))

    if game_state.replace_card: return list(range(MOVE_AMOUNT))

    legal_moves = []

    for index in np.where(game_state.s != 0)[0]:
        for checks, func in [[(GAME_DIMENSIONS[1] + 1, GAME_DIMENSIONS[1] - 1, GAME_DIMENSIONS[1]), lambda m: m], [(1,), lambda m: 0]]:
            for check in checks:
                for multiplier in [1, -1]:
                    checking_index = index + check * multiplier
                    if checking_index not in legal_moves and check_index(game_state.s, index, checking_index, lambda x: x == 0, func, multiplier):
                        legal_moves.append(checking_index)

    return legal_moves  # Prob can be more efficient


def get_card(value):
    suit = np.floor((value - 1) / 13)
    value = ((value - 1) % 13) + 2
    return suit, value


def score_row(cards):
    values = sorted([get_card(card)[1] for card in cards])
    
    histo_dict = {(1, 4): 20, (2, 3): 15, (1, 1, 3): 8, (1, 2, 2): 4, (1, 1, 1, 2): 2}

    histo = tuple(sorted([values.count(value) for value in set(values)]))

    if histo in histo_dict:
        return histo_dict[histo]

    färgrad = len(set(get_card(card)[0] for card in cards)) == 1
    stege = values[-1] - values[0] == 4 or values == [2, 3, 4, 5, 14]
    
    score = 0

    if färgrad:
        score += 10
    if stege:
        score += 10

        op = values[-2] == 13
        if op:
            return 3 * score - 10
    
    return score
    

def check_game_over(game_state):
    if len(game_state.deck) == 51 - np.prod(GAME_DIMENSIONS) - REPLACE_CARDS:
        score = 0
        board = game_state.s.reshape(GAME_DIMENSIONS)
        # print(print_board(board.flatten()))
        for rowcol in [board, board.T]:
            for row in rowcol:
                score += score_row(row)
        
        return score * 0.02


def take_action(game_state, action):
    board = game_state.s.copy()
    deck = game_state.deck.copy()
    card = game_state.drawn_card

    if action != 25: board[action] = card
    card = deck.pop()

    return (board, deck, card)


def format_card(card):
    suit_dict = {0: "sp", 1: "hj", 2: "ru", 3: "kl"}
    suit, value = get_card(card)

    return f"{suit_dict[suit]}{int(value)}"


def inverse_format_card(card):
    suit_dict = {"sp": 0, "hj": 1, "ru": 2, "kl": 3}

    c = int(card[2:]) - 1
    c += 13 * suit_dict[card[:2]]

    return c


def print_board(board):
    board = board.astype("<U4")
    board[board == "0.0"] = "---"
    for i, pos in enumerate(board):
        if pos != "---":
            board[i] = format_card(float(pos))
    return board.reshape(GAME_DIMENSIONS)


def print_values(values):
    print(f"Action values are: {[values[-1]]}\n{values[:-1].reshape(GAME_DIMENSIONS)}")
