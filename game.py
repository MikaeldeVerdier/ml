import numpy as np
import config

GAME_DIMENSIONS = (6, 7)
IN_A_ROW = 4
MOVE_AMOUNT = GAME_DIMENSIONS[1]

def generate_game_state(node, mirror):
    root = node
    board_history = []
    for player in [1, -1]:
        node = root
        for _ in range(config.DEPTH):
            s = node.s if not mirror else mirror_board(node.s)
            position = np.zeros(len(s))
            position[s == player] = 1
            board_history.append(position.reshape(GAME_DIMENSIONS))
            if node.parent and node.parent.parent:
                node = node.parent.parent
    # board_history.append(np.array([[[node.player + 1]] * GAME_DIMENSIONS[1]] * GAME_DIMENSIONS[0]))
    game_state = np.moveaxis(np.array(board_history), 0, -1)
    return game_state

def generate_tutorial_game_state(node, mirror):
    root = node
    board_history = []
    for player in [node.player, -node.player]:
        node = root
        for _ in range(config.DEPTH):
            s = node.s if not mirror else np.array(mirror_board(node.s))
            position = np.zeros(len(s))
            position[s == player] = 1
            board_history += position.tolist()
            if node.parent and node.parent.parent: node = node.parent.parent
    # board_history.append(np.array([[[node.player + 1]] * GAME_DIMENSIONS[1]] * GAME_DIMENSIONS[0]))
    game_state = np.reshape(board_history, (GAME_DIMENSIONS + (config.DEPTH * 2,)))
    return game_state

def mirror_board(board):
    return [board[x + z * GAME_DIMENSIONS[1]] for z in range(GAME_DIMENSIONS[0]) for x in range(GAME_DIMENSIONS[1] - 1, -1, -1)]

def get_legal_moves(board, all_moves):
    if MOVE_AMOUNT != np.prod(GAME_DIMENSIONS):
        legal_moves = []
        for dim1 in range(GAME_DIMENSIONS[1]):
            for dim2 in range(GAME_DIMENSIONS[0]):
                if board[dim1 + dim2 * GAME_DIMENSIONS[1]] != 0:
                    if dim2 != 0: legal_moves.append(dim1 + (dim2 - 1) * GAME_DIMENSIONS[1])
                    elif all_moves: legal_moves.append(-1)
                    # legal_moves.append(-1 if dim2 == 0 else dim1 + (dim2 - 1) * GAME_DIMENSIONS[1])
                    break
            else: legal_moves.append(dim1 + dim2 * GAME_DIMENSIONS[1])
    else:
        legal_moves = np.where(board == 0)
        if len(legal_moves) != 0: legal_moves = legal_moves[0]
    return legal_moves

def check_game_over(board):
    board = board.tolist()
    for player in [1, -1]:
        for checks in [[[GAME_DIMENSIONS[0], GAME_DIMENSIONS[1] - IN_A_ROW + 1], list(range(IN_A_ROW))], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1]], [element * GAME_DIMENSIONS[1] for element in list(range(IN_A_ROW))]], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1] - IN_A_ROW + 1], [element * (GAME_DIMENSIONS[1] - 1) + IN_A_ROW - 1 for element in list(range(IN_A_ROW))]], [[GAME_DIMENSIONS[0] - IN_A_ROW + 1, GAME_DIMENSIONS[1] - IN_A_ROW + 1], [element * (GAME_DIMENSIONS[1] + 1) for element in list(range(IN_A_ROW))]]]:
            for i in range(checks[0][0]):
                for i2 in range(checks[0][1]):
                    pos = [board[i * GAME_DIMENSIONS[1] + i2 + i3] == player for i3 in checks[1]]
                    if pos.count(True) == IN_A_ROW: return player

    if np.count_nonzero(board) == np.prod(GAME_DIMENSIONS): return 0

def move(board, a, player):
    board[a] = player
    return board

def print_board(board):
    board = board.astype("<U1")
    board[board == "-"] = "O"
    board[board == "1"] = "X"
    board[board == "0"] = " "
    return board.reshape(GAME_DIMENSIONS)

def print_values(values):
    return values.reshape(GAME_DIMENSIONS)
