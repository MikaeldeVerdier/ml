# Main loop
loop_iterations = 36

# Network architecture
convolutional_layer = {"filter_amount": 75, "kernel_size": (4, 4)}
residual_layer = {"amount": 5, "filter_amount": 75, "kernel_size": (4, 4)}
dense_value_head = 64

# Self-play
game_amount_self_play = 30
position_amount = 30000
MCTSSims = 50
depth = 1
turns_until_tau = 10
cpuct = 1

# Retraining network
training_iterations = 10
batch_size = 256
epochs = 1
validation_split = 0.3
reg_const = 1e-4
lr = 1
momentum = 0.9

# Evaluating network
game_amount_evaluation = 20
winning_threshold = 1.3

# Play-test
game_amount_play_test = 4

# General
save_folder = "save_folder/"
