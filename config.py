# Game rules
game_dimensions = (5, 5)
in_a_row = 4
move_amount = game_dimensions[1]

# Main loop
loop_iterations = 10

# Network architecture
convolutional_layer = {"filter_amount": 75, "kernel_size": (4, 4)}
residual_layer = {"amount": 5, "filter_amount": 75, "kernel_size": (4, 4)}
dense_value_head = 64

# Self-playing network
game_amount_self_playing = 1
MCTSSims = 25
depth = 1
turns_until_tau = 10

# Retraining network
training_iterations = 1
batch_size = 4
epochs = 1
validation_split = 0.2
reg_const = 0.0001
lr = 0.1
momentum = 0.9

# Evaluating network
game_amount_evaluation = 1
winning_threshold = 1.2

# Play-test
game_amount_play_test = 2