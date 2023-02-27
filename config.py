# Main loop
LOOP_ITERATIONS = 3001

# Network architecture
CONVOLUTIONAL_LAYERS_POSITION = (16, 32, 64, 128, 256, 256, 256)
CONVOLUTIOANL_SHAPE_POSITION = (3, 3)
# POOL_SHAPE_POSITION = (2, 2)
DENSE_POSITION = [1024, 1024, 1024, 2048, 2048, 2048, 4096, 4096, 4096]

DENSE_DECK = [52, 52, 52, 52, 104, 104, 208, 208, 416, 416, 416]

DENSE_DRAWN_CARD = [52, 52, 52, 52, 104, 104, 208, 208, 416, 416, 416]

TOTAL_NEURONS = DENSE_POSITION[-1] + DENSE_DECK[-1] + DENSE_DRAWN_CARD[-1]
DENSE_SHARED = [TOTAL_NEURONS, TOTAL_NEURONS, TOTAL_NEURONS, TOTAL_NEURONS * 2, TOTAL_NEURONS * 2, TOTAL_NEURONS, TOTAL_NEURONS * 0.5, TOTAL_NEURONS * 0.25]
DENSE_POLICY_HEAD = [TOTAL_NEURONS * 0.25, TOTAL_NEURONS * 0.25, TOTAL_NEURONS * 0.125, TOTAL_NEURONS * 0.06125, TOTAL_NEURONS * 0.03125, TOTAL_NEURONS * 0.03125]
USE_BIAS = True

# Self-play
GAME_AMOUNT_SELF_PLAY = 20
POSITION_AMOUNT = 30000  # Replay buffer size
DEPTH = 1  # Amount of previous states included in nn input
EPSILON = lambda version: max(0.1, 1 - version / LOOP_ITERATIONS)  # Probability of choosing a random move, if not specified

# Retraining network
TRAINING_ITERATIONS = 10
BATCH_SIZE = (256, 32)  # Index 0: amount of positions sampled. Index 1: actual batch_size used in momdel.fit()
EPOCHS = 1
GAMMA = 0.99  # Discounting factor for future rewards when calculating targets
VERSION_OFFSET = 50  # Reciprocal of frequency of target nn copying main nn
SAVING_FREQUENCY = 250  # Reciprocal of frequency of saving progress
VALIDATION_SPLIT = 0.2
REG_CONST = 1e-4  # L2 Regularization Hyperparameter
LEARNING_RATE = lambda version: max(1e-6, 1e-4 + version * (1e-6 - 1e-4) / (LOOP_ITERATIONS * 0.9))

# Evaluating network
GAME_AMOUNT_EVALUATION = 100
EVALUATION_FREQUENCY = 10  # Reciprocal of frequency of evaluating model

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 500

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"  # Path where saving is done
