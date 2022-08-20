# Main loop
LOOP_ITERATIONS = 50

# Network architecture
CONVOLUTIONAL_LAYER = {"filter_amount": 75, "kernel_size": (4, 4)}
RESIDUAL_LAYER = {"amount": 10, "filter_amount": 75, "kernel_size": (4, 4)}
DENSE_VALUE_HEAD = 20
USE_BIAS = False

# Self-play
GAME_AMOUNT_SELF_PLAY = 1
POSITION_AMOUNT = 2
MCTS_SIMS = 50
DEPTH = 1
TURNS_UNTIL_TAU = 25
CPUCT = 1
EPSILON = 0.5
ALPHA = 0.8

# Retraining network
TRAINING_ITERATIONS = 3
BATCH_SIZE = 2
EPOCHS = 1
VALIDATION_SPLIT = 0
REG_CONST = 1e-4
LEARNING_RATE = 0.1
MOMENTUM = 0.9

# Evaluating network
GAME_AMOUNT_EVALUATION = 2
WINNING_THRESHOLD = 1.3

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 20

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"
