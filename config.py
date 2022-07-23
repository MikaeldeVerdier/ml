# Game rules
game_dimensions = (6, 7)
in_a_row = 4
move_amount = game_dimensions[1]

# Main loop
loop_iterations = 30

# Network architecture
convolutional_layer = {"filter_amount": 75, "kernel_size": (4, 4)}
residual_layer = {"amount": 5, "filter_amount": 75, "kernel_size": (4, 4)}
dense_value_head = 64

# Self-play
game_amount_self_play = 10
position_amount = 10000
MCTSSims = 50
depth = 1
turns_until_tau = 10

# Retraining network
training_iterations = 3
batch_size = 256
epochs = 1
validation_split = 0.3
reg_const = 1e-4
lr = 0.1
momentum = 0.9

# Evaluating network
game_amount_evaluation = 5
winning_threshold = 1.2

# Play-test
game_amount_play_test = 5

# General
save_folder = "save_folder/"
