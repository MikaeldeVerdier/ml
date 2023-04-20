from funcs import linear_wrapper_func

# Main loop
VERSION_AMOUNT = 5000

# Network architecture
CONVOLUTIONAL_LAYERS_POSITION = (64, 128, 128, 128, 128, 128)
CONVOLUTIOANL_SHAPE_POSITION = (3, 3, 3)
DENSE_POSITION = [3200, 3000, 2500, 2048, 2048, 2048, 2048, 2048, 2048, 1024, 512]

DENSE_DECK = [64, 128, 128, 128]

DENSE_DRAWN_CARD = [64, 128, 128, 128]

DENSE_SCORES = [2, 4, 8, 16, 32, 32]

DENSE_POLICY_HEAD = [1024, 1024, 1024, 1024, 1024, 512, 256, 128, 64, 32, 32]
USE_BIAS = True

# Self-play
GAME_AMOUNT_SELF_PLAY = 40
BUFFER_REQUIREMENT = 3000  # Minimum requirement of new positions generated during training
BUFFER_SIZE = 30000  # Replay buffer size
DEPTH = 1  # Amount of previous states included in nn input
epsilon = linear_wrapper_func(1, 0.1, VERSION_AMOUNT * 0.7)  # Linearly annealing function for probability of choosing random action, if not specified (like in training)

# Retraining network
TRAINING_ITERATIONS = 10
BATCH_SIZE = (256, 32)  # Index 0: amount of positions sampled. Index 1: actual batch_size used in model.fit()
EPOCHS = 1  # Amount of times batch is looped over
GAMMA = 0.95  # Discounting factor for future rewards when calculating targets
VERSION_OFFSET = 50  # Reciprocal of frequency of target nn copying main nn
SAVING_FREQUENCY = 250  # Reciprocal of frequency of saving progress
MODEL_CHECKPOINT_FREQUENCY = 1000  # Reciprocal of frequency of saving checkpoint models
VALIDATION_SPLIT = 0.2  # Share of data used as validation
REG_CONST = 1e-4  # L2 Regularization Hyperparameter
learning_rate = linear_wrapper_func(1e-4, 1e-6, VERSION_AMOUNT * 0.7, use_cache=False)

# Evaluating network
GAME_AMOUNT_EVALUATION = 100
EVALUATION_FREQUENCY = 25  # Reciprocal of frequency of evaluating model

# Play versions
GAME_AMOUNT_PLAY_VERSIONS = 1000

# Play-test
GAME_AMOUNT_PLAY_TEST = 4

# General
SAVE_PATH = "save_folder/"  # Path where saving is done
