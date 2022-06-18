import numpy as np
import random
import copy
import config
import game
from agent import Agent

agents = [None, Agent(False, 1), Agent(False, 2)]
best_agent = 1
# current_agent = -best_agent

def reset_all_mcts(agents):
    for agent in agents[1:]: agent.reset_mcts()
    for i in [1, -1]: agents[i].mcts.player = i

def reset_enemy_mcts(agent):
    agent.reset_mcts()
    agent.mcts.player = 1

def self_play(agents, games, tau):
    reset_all_mcts(agents)

    training_set = [[]]
    outcomes = [0, 0, 0]
    game_count = 0
    turn = 0
    root = agents[1].mcts
    while game_count < games:
        if turn == config.turns_until_tau:
            tau == 10e-45
            print("Tau is now 0")
        root, action, pi, mcts_value, nn_value = root.play_turn(tau)
        turn += 1

        training_set[-1].append([game.generate_game_state(root), pi])

        print(f"Action values are: \n {game.print_values(np.round(pi, 3))}")
        print(f"Move to make is: {action}")
        print(f"Position is now: \n {game.print_board(root.s)}")
        print(f"MCTS percieved value is: {np.round(mcts_value, 3)}")
        print(f"NN percieved value is: {np.round(nn_value * 1000)/1000} \n")

        outcome = game.check_game_over(root.s)
        if outcome is not None:
            reset_all_mcts(agents)
            game_count += 1
            print(f"Game outcome was: {outcome}")
            print(f"Amount of games played is now: {game_count} \n")
            root = agents[-outcome if outcome != 0 else -1].mcts
            [Set.append(outcome) for Set in training_set[-1]]
            training_set.append([])
            outcomes[outcome] += 1
    
    return training_set[:-1], outcomes

def retrain_network(agent, batch, best_agent):
    for _ in range(config.training_iterations):
        positions = []
        for position in batch: positions += position
        minibatch = random.sample(positions, config.batch_size)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)
        agent.nn.save_progress(best_agent)
    agent.nn.plot_losses()

    return (x, y)

def evaluate_network(agents, best_agent):
    results = self_play(agents, config.game_amount_evaluation, 10e-45)[1]
    print(f"The results were: {results}")
    if results[-best_agent]/(results[best_agent] if results[best_agent] != 0 else 0.1) > config.winning_threshold:
        best_agent *= -1
        print(f"{best_agent} is now best player!")
        agents[1].nn.save_progress(best_agent)
        log(results, best_agent)
    
def play_test(agent, games):
    reset_enemy_mcts(agent)

    outcomes = [0, 0, 0]
    game_count = 0
    player_turn = False
    root = agent.mcts
    while game_count < games:
        if not player_turn:
            root, action, pi, mcts_value, nn_value = root.play_turn(10e-45)
            player_turn = True

            print(f"Action values are: \n {game.print_values(np.round(pi, 3))}")
            print(f"Move to make is: {action}")
            print(f"Position is now: \n {game.print_board(root.s)}")
            print(f"MCTS percieved value is: {np.round(mcts_value, 3)}")
            print(f"NN percieved value is: {np.round(nn_value * 1000)/1000} \n")
        else:
            for _ in range(config.move_amount):
                root.selection()
            root = root.children[int(input("Make your move: "))]
            print(f"Move to make is: {root.parent_action}")
            print(f"Position is now: \n {game.print_board(root.s)}")
            player_turn = False

        outcome = game.check_game_over(root.s)
        if outcome is not None:
            agent.reset_mcts()
            game_count += 1
            print(f"Game outcome was: {outcome}")
            print(f"Amount of games played is now: {game_count} \n")
            root = agents[-outcome if outcome != 0 else -1].mcts
            outcomes[outcome] += 1
            
    return outcomes
    
def log(results, best_agent):
    file = open("/Users/mikaeldeverdier/ai/try 2/binary/reinforcement/probs_move_amount -100/log.txt", "w")
    file.write(f"Results are: {results}\n The best_agent is now {best_agent}\n")
    file.close()

for _ in range(config.loop_iterations):
    [agent.reset_mcts() for agent in agents[1:]]
    copyAgent = copy.copy(agents[best_agent])
    copyAgent.reset_mcts()
    batch = self_play([None, agents[best_agent], copyAgent], config.game_amount_self_playing, 1)[0]
    (x, y) = retrain_network(agents[-best_agent], batch, best_agent)
    evaluate_network(agents, best_agent)
    play_test(agents[best_agent], 5)







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