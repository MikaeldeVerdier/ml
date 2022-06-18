import numpy as np
import random
import copy
import config
import game
from agent import Agent

agents = [None, Agent(False, 1), Agent(True, 2)]
best_agent = 1
# current_agent = -best_agent

def self_play(agents, games):
    for i in [1, -1]: agents[i].mcts.player = i

    training_set = [[]]
    outcomes = [0, 0, 0]
    game_count = 0
    turn = 0
    root = agents[1].mcts
    while game_count < games:
        root, action, pi, mcts_value, nn_value = root.run(1)

        training_set[-1].append([game.generate_game_state(root), pi])

        print(f"Action values are: \n {game.print_values(np.round(pi, 3))}")
        print(f"Move to make is: {action}")
        print(f"Position is now: \n {game.print_board(root.s)}")
        print(f"MCTS percieved value is: {np.round(mcts_value, 3)}")
        print(f"NN percieved value is: {np.round(nn_value * 1000)/1000} \n")

        outcome = game.check_game_over(root.s)
        if outcome is not None:
            game_count += 1
            print(f"Game outcome was: {outcome}")
            print(f"Amount of games played is now: {game_count} \n")
            root = agents[-outcome if outcome != 0 else -1].mcts
            [Set.append(outcome) for Set in training_set[-1]]
            training_set.append([])
            outcomes[outcome] += 1
    
    return training_set[:-1], outcomes

def retrain_network(agent, batch):
    for _ in range(config.training_iterations):
        positions = []
        for position in batch: positions += position
        minibatch = random.sample(positions, config.batch_size)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)
    agent.nn.plot_losses()

    return (x, y)

def evaluate_network(agents, best_agent):
    results = self_play(agents, config.game_amount_evaluation)[1]
    print(f"The results were: {results}")
    if results[-best_agent]/(results[best_agent] if results[best_agent] != 0 else 0.1) > config.winning_threshold:
        best_agent = -best_agent
        print(f"{best_agent} is now best player!")

for _ in range(config.loop_iterations):
    [agent.reset_mcts() for agent in agents[1:]]
    copyAgent = copy.copy(agents[best_agent])
    copyAgent.reset_mcts()
    batch = self_play([None, agents[best_agent], copyAgent], config.game_amount_self_playing)[0]
    (x, y) = retrain_network(agents[-best_agent], batch)
    evaluate_network(agents, best_agent)

for agent in agents:
    agent.nn.plot_losses()







"""
game_count = 0

saves = [[]]

player_turn = 1

root = agents[player_turn].mcts
while game_count < 2:
    for _ in range(100):
        root.selection()

    saves[-1].append([root.nn_pass, root.p])
    root = root.children[np.argmax(root.p)]

    outcome = root._check_game_over(root.s)
    if outcome is not None:
        print("GAME RESET")
        game_count += 1
        root = agents[-player_turn].mcts
        [save.append(outcome) for save in saves[-1]]
        saves.append([])

    player_turn *= -1

x = []
for save in saves: x += save
print(np.array(saves[:-1]).shape)
minibatch = [random.choice(x) for _ in range(64)]
print(np.array(minibatch).shape)"""