import numpy as np
import config

GAME_DIMENSIONS = (52,)
NN_INPUT_DIMENSIONS = [(config.DEPTH, GAME_DIMENSIONS[0], 52), (config.DEPTH, 52)]
MOVE_AMOUNT = GAME_DIMENSIONS[0] * 2 + 1
SUIT_LENGTH = GAME_DIMENSIONS[0]/4

def generate_game_states(history, t):
    data = history[:t + 1]
    game_states = ({"state": None},) * (config.DEPTH - len(data)) + tuple(data)[-config.DEPTH:]
    game_states = tuple([game_state["state"] for game_state in game_states])

    return game_states

def generate_nn_pass(game_states, suit_change=False):
    game_state = game_states[-1]

    suit_changes = [0] if not suit_change else [0, 13, 26, 39]
    
    nn_pass = []
    for suit_change in suit_changes:
        nn_pass.append([[], []])
        for depth in range(config.DEPTH):
            s = game_state.s.copy()
            de = game_state.deck
            for var in [s, de]:
                for i, card in enumerate(var):
                    if card != 0: var[i] += suit_change
                    if var[i] > 52: var[i] -= 52

            state = []
            for i in range(1, 53):
                position = np.zeros(len(s))
                position[s == i] = 1
                state.append(np.reshape(position, NN_INPUT_DIMENSIONS[0][1:-1]))

            state = np.moveaxis(state, 0, -1).tolist()
            nn_pass[-1][0].append(state)

            deck = np.zeros(52)
            for card in de: deck[card - 1] = 1
            nn_pass[-1][1].append(deck.tolist())

            if depth != config.DEPTH -1:
                if game_states[-depth - 2]: game_state = game_states[-depth - 2]
                else:
                    for _ in range(config.DEPTH - depth - 1):
                        for i, func in enumerate([np.zeros, np.ones]):
                            nn_pass[-1][i].append(func(NN_INPUT_DIMENSIONS[i][:-1]))
                    break
            # nn_pass[-1] = [np.array(np.moveaxis(dim, 0, -1)).tolist() for dim in nn_pass[-1]]
            # nn_pass[-1] = [np.array(np.moveaxis(dim, 0, -1), dtype=np.int32).tolist() for dim in nn_pass[-1]]

    return nn_pass


def get_legal_moves(node):
    legal_moves = []
    checks = [1]
    for i, pos in enumerate(node.s[1:], 1):
        if pos == 0: break
        if i == 3: checks += [3]
        for check in checks:
            if can_move(pos, node.s[i - check]): legal_moves.append(int(2 * i + 0.5 + 0.5 * check))

    return legal_moves if len(legal_moves) or not len(node.deck) else [0]  # ??? surely this can be done better


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
    if not node.deck and not len(get_legal_moves(node)): return 52 - np.where(node.s == 0)[0][0]


def take_action(node, move):
    if move == 0:
        board = node.s.copy()
        deck = node.deck.copy()
        card = np.random.choice(deck)

        index = np.where(board == 0)[0][0]
        board[index] = card
        deck.remove(card)
        # node.deck = np.delete(node.deck, rand_index)
    else:
        board = node.s.copy()

        index = int(np.ceil(move / 2)) - 1
        kind = 3 - 2 * (move % 2)
        board[index - kind] = board[index]
        board = np.delete(board, index)
        board = np.append(board, 0)

        deck = node.deck.copy()
    return (board, deck)


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
