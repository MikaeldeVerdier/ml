import numpy as np
import config

def generate_game_state(node):
    real_node = node
    board_history = []
    for player in [1, -1]:
        node = real_node
        i = 0
        while i < config.depth:
            position = np.zeros(len(node.s))
            position[node.s == player] = 1
            board_history.append(position.reshape(config.game_dimensions))
            if node.parent is not None and node.parent.parent is not None:
                node = node.parent.parent
            i += 1
    board_history.append(np.array([[{1: 1, -1: 0}[node.player]] * config.game_dimensions[1]] * config.game_dimensions[0]))
    game_state = np.moveaxis(np.array(board_history), 0, -1)
    return game_state

def get_legal_moves(board):
    if config.move_amount != np.prod(config.game_dimensions):
        legal_moves = []
        for i in range(config.game_dimensions[1]):
            for i2 in range(config.game_dimensions[0]):
                if board[i + i2 * config.game_dimensions[1]] != 0:
                    legal_moves.append(-1 if i2 == 0 else i + (i2 - 1) * config.game_dimensions[1])
                    break
            else:
                legal_moves.append(i + i2 * config.game_dimensions[1])
    else:
        legal_moves = np.where(board == 0)
        if len(legal_moves) != 0: legal_moves = legal_moves[0]
    return legal_moves

def check_game_over(board):
    board = board.tolist()
    for player in [1, -1]:
        for checks in [[[config.game_dimensions[0], config.game_dimensions[1] - config.in_a_row + 1], list(range(config.in_a_row))], [[config.game_dimensions[0] - config.in_a_row + 1, config.game_dimensions[1]], [element * config.game_dimensions[1] for element in list(range(config.in_a_row))]], [[config.game_dimensions[0] - config.in_a_row + 1, config.game_dimensions[1] - config.in_a_row + 1], [element * (config.game_dimensions[1] - 1) + config.in_a_row - 1 for element in list(range(config.in_a_row))]], [[config.game_dimensions[0] - config.in_a_row + 1, config.game_dimensions[1] - config.in_a_row + 1], [element * (config.game_dimensions[1] + 1) for element in list(range(config.in_a_row))]]]:
            for i in range(checks[0][0]):
                for i2 in range(checks[0][1]):
                    # print([i * config.game_dimensions[1] + i2 + i3 for i3 in checks[1]])
                    pos = [board[i * config.game_dimensions[1] + i2 + i3] == player for i3 in checks[1]]
                    if pos.count(True) == config.in_a_row: return player

    if np.count_nonzero(board) == np.prod(config.game_dimensions):
        return 0

def move(board, a, player):
    board[a] = player
    return board

def print_board(board):
    board = board.astype("<U1")
    board[board == "-"] = "O"
    board[board == "1"] = "X"
    board[board == "0"] = " "
    return board.reshape(config.game_dimensions)

def print_values(values):
    return values.reshape(config.game_dimensions)
