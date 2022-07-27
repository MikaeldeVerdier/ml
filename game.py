import numpy as np
import config

game_dimensions = (6, 7)
in_a_row = 4
move_amount = game_dimensions[1]

def generate_game_state(node, mirror):
    root = node
    board_history = []
    for player in [1, -1]:
        node = root
        i = 0
        while i < config.depth:
            s = node.s if not mirror else mirror_board(node.s)
            position = np.zeros(len(s))
            position[s == player] = 1
            board_history.append(position.reshape(game_dimensions))
            if node.parent and node.parent.parent:
                node = node.parent.parent
            i += 1
    board_history.append(np.array([[{1: 0, -1: 1}[node.player]] * game_dimensions[1]] * game_dimensions[0]))
    game_state = np.moveaxis(np.array(board_history), 0, -1)
    return game_state

def mirror_board(board):
    """b = []
    for z in range(game_dimensions[0]):
        for x in range(game_dimensions[1] - 1, -1, -1):
            print(x + z * game_dimensions[1])
            b.append(board[x + z * game_dimensions[1]])"""
    
    mirrored_board = [board[x + z * game_dimensions[1]] for z in range(game_dimensions[0]) for x in range(game_dimensions[1] - 1, -1, -1)]

    return mirrored_board

def get_legal_moves(board):
    if move_amount != np.prod(game_dimensions):
        legal_moves = []
        for dim1 in range(game_dimensions[1]):
            for dim2 in range(game_dimensions[0]):
                if board[dim1 + dim2 * game_dimensions[1]] != 0:
                    legal_moves.append(-1 if dim2 == 0 else dim1 + (dim2 - 1) * game_dimensions[1])
                    break
            else:
                legal_moves.append(dim1 + dim2 * game_dimensions[1])
    else:
        legal_moves = np.where(board == 0)
        if len(legal_moves) != 0: legal_moves = legal_moves[0]
    return legal_moves

def check_game_over(board):
    board = board.tolist()
    for player in [1, -1]:
        for checks in [[[game_dimensions[0], game_dimensions[1] - in_a_row + 1], list(range(in_a_row))], [[game_dimensions[0] - in_a_row + 1, game_dimensions[1]], [element * game_dimensions[1] for element in list(range(in_a_row))]], [[game_dimensions[0] - in_a_row + 1, game_dimensions[1] - in_a_row + 1], [element * (game_dimensions[1] - 1) + in_a_row - 1 for element in list(range(in_a_row))]], [[game_dimensions[0] - in_a_row + 1, game_dimensions[1] - in_a_row + 1], [element * (game_dimensions[1] + 1) for element in list(range(in_a_row))]]]:
            for i in range(checks[0][0]):
                for i2 in range(checks[0][1]):
                    # print([i * game_dimensions[1] + i2 + i3 for i3 in checks[1]])
                    pos = [board[i * game_dimensions[1] + i2 + i3] == player for i3 in checks[1]]
                    if pos.count(True) == in_a_row: return player

    if np.count_nonzero(board) == np.prod(game_dimensions):
        return 0

def move(board, a, player):
    board[a] = player
    return board

def print_board(board):
    board = board.astype("<U1")
    board[board == "-"] = "O"
    board[board == "1"] = "X"
    board[board == "0"] = " "
    return board.reshape(game_dimensions)

def print_values(values):
    return values.reshape(game_dimensions)
