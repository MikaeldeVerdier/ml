# Main loop
LOOP_ITERATIONS = 50

# Network architecture
CONVOLUTIONAL_LAYER = {"filter_amount": 256, "kernel_size": (3, 3)}
RESIDUAL_LAYER = {"amount": 20, "filter_amount": 256, "kernel_size": (3, 3)}
DENSE_VALUE_HEAD = 20
USE_BIAS = True

# Self-play
GAME_AMOUNT_SELF_PLAY = 1
POSITION_AMOUNT = 30000
MCTS_SIMS = 50
TURNS_UNTIL_TAU = 25
CPUCT = 0.2
EPSILON = 0.5
ALPHA = 0.8

# Retraining network
TRAINING_ITERATIONS = 10
BATCH_SIZE = 256
EPOCHS = 1
VALIDATION_SPLIT = 0.2
REG_CONST = 1e-4
LEARNING_RATE = 1e-3
MOMENTUM = 0.5

# Evaluating network
GAME_AMOUNT_EVALUATION = 10
EVALUATION_FREQUENCY = 2
WINNING_THRESHOLD = 1

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 20

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"
