# Main loop
LOOP_ITERATIONS = 5001

# Network architecture
CONVOLUTIONAL_LAYERS_POSITION = [(32, (3, 3, 3)), (64, (3, 3, 3)), (128, (3, 3, 3)), (256, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3)), (512, (3, 3, 3))]
DENSE_POSITION = [4092, 4092, 4092, 4092, 4092, 4092, 4092, 4092, 4092]

CONVOLUTIONAL_LAYERS_DECK = [(32, 3), (64, 3), (128, 3), (128, 3), (128, 3)]
DENSE_DECK = [2048, 2048, 2048, 1024]

CONVOLUTIONAL_LAYERS_DRAWN_CARD = [(32, 3), (64, 3), (128, 3), (128, 3), (128, 3)]
DENSE_DRAWN_CARD = [2048, 2048, 2048, 1024]

DENSE_SHARED = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512]
DENSE_POLICY_HEAD = [512, 512, 512, 512, 512, 512, 256]
USE_BIAS = True

# Self-play
GAME_AMOUNT_SELF_PLAY = 70
POSITION_AMOUNT = 30000
DEPTH = 1
EPSILON = [1, 0.1, LOOP_ITERATIONS * 0.9]  # Exploration rate in the form of: [initial, final, duration]
EPSILON_STEP_SIZE = (EPSILON[0] - EPSILON[1]) / EPSILON[2]  # Step size of EPSILON

# Retraining network
TRAINING_ITERATIONS = 10
BATCH_SIZE = (256, 32)
EPOCHS = 1
GAMMA = 0.9  # Discounting factor for future rewards when calculating targets
VERSION_OFFSET = 50
SAVING_FREQUENCY = 100
VALIDATION_SPLIT = 0.2
REG_CONST = 1e-4
LEARNING_RATE = 1e-6

# Evaluating network
GAME_AMOUNT_EVALUATION = 100
EVALUATION_FREQUENCY = 10
WINNING_THRESHOLD = 1

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 20

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"
