from funcs import linear_wrapper_func

# Main loop
VERSION_AMOUNT = 2000

# Network architecture
USE_BIAS = True
DROPOUT_FACTOR = 0

DENSE_POSITION = [256, 512, 1024, 2048, 4096, 4096, 2048, 1024, 512]

DENSE_DRAWN_CARD = [16, 32, 64, 128, 258, 512]

DENSE_POLICY_HEAD = [1024, 2048, 4096, 4096, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 16]

# Self-play
GAME_AMOUNT_SELF_PLAY = 42
BUFFER_REQUIREMENT = 10500  # Minimum requirement of new positions generated during training
BUFFER_SIZE = 90000  # Replay buffer size
DEPTH = 1  # Amount of previous state observations used as information for the neural network
epsilon = linear_wrapper_func(1, 0.1, VERSION_AMOUNT * 0.8)  # Function for probability of choosing random action, if not specified (like in training)

# Retraining network
TRAINING_ITERATIONS = 40  # Amount of times a random positions are sampled and used to train the neural network
BATCH_SIZE = (256, 32)  # [0]: amount of positions sampled. [1]: actual batch_size used in model.fit()
EPOCHS = 1  # Amount of times batch is looped over
GAMMA = 0.8  # Discounting factor for future rewards when calculating targets
VERSION_OFFSET = 50  # Reciprocal of frequency of target neural network copying main neural network
SAVING_FREQUENCY = 250  # Reciprocal of frequency of saving progress
MODEL_CHECKPOINT_FREQUENCY = 500  # Reciprocal of frequency of saving checkpoint models
VALIDATION_SPLIT = 0.2  # Share of data used for validation
REG_CONST = 1e-3  # L2 Regularization hyperparameter
learning_rate = linear_wrapper_func(1e-4, 1e-6, VERSION_AMOUNT * 0.9, use_cache=False)

# Evaluating network
GAME_AMOUNT_EVALUATION = 100
EVALUATION_FREQUENCY = 25  # Reciprocal of frequency of evaluating model

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 10000

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"  # Directory for saving
