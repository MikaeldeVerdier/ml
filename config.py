# Main loop
LOOP_ITERATIONS = 1000

# Network architecture
CONVOLUTIONAL_LAYERS_POSITION = [(32, (3, 3, 3)), (64, (3, 3, 3)), (128, (3, 3, 3)), (256, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3))]
DENSE_POSITION = [1024, 512]

CONVOLUTIONAL_LAYERS_DECK = [(32, 3), (64, 3)]
DENSE_DECK = [1024, 512]

CONVOLUTIONAL_LAYERS_DRAWN_CARD = [(32, 3), (64, 3)]
DENSE_DRAWN_CARD = [1024, 512]

DENSE_SHARED = [1024, 512]
DENSE_VALUE_HEAD = [32, 32]
DENSE_POLICY_HEAD = [512, 512]
USE_BIAS = True

# Self-play
GAME_AMOUNT_SELF_PLAY = 30
POSITION_AMOUNT = 30000
MCTS_SIMS = 50
DEPTH = 1
CPUCT = 1.4  # MCTS exploration
ALPHA = 1  # Dirichlet randomness in MCTS
TURNS_UNTIL_TAU = 20  # Turn when agent starts tryharding

# Retraining network
TRAINING_ITERATIONS = 10
BATCH_SIZE = (256, 32)
EPOCHS = 1
GAMMA = 0.5  # Discounting factor
LAMBDA = 0.5  # Factor for trade-off of bias and variance for advantage
EPSILON = 5  # Reglates clipping values for L_clip
OMEGA = 0.5  # Coefficient for J_vf
BETA = 0  # Coefficient for entropy bonus H
VALIDATION_SPLIT = 0.2
REG_CONST = 1e-4
LEARNING_RATE = 1e-5
MOMENTUM = 0.5

# Evaluating network
GAME_AMOUNT_EVALUATION = 50
EVALUATION_FREQUENCY = 2
WINNING_THRESHOLD = 1

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 20

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"
