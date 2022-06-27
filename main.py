import numpy as np
import random
import json
import copy
import datetime
import config
import game
from player import *

load = [False, False, False]
agents = [None, Agent(load[0], 1), Agent(load[1], 2)]
best_agent = json.loads(open("save.json", "r").read())["best_agent"]
if not load[0] and not load[1]: open("log.txt", "w").truncate(0)

def assign_players(players):
    for i in [1, -1]: players[i].root.player = i

def self_play(players, games, training):
    if training: training_set = [[]]
    outcomes = [0, 0, 0]
    game_count = 0
    starts = 1
    assign_players(players)
    while game_count < games:
        player_turn = starts
        player = players[player_turn]
        root = player.start_node
        turn = 1
        tau = 1 if training else 1e-10
        outcome = None
        while outcome is None:
            if turn > config.turns_until_tau: tau = 1e-10
            player = players[player_turn]
            root, pi = player.play_turn(root, tau)
            player_turn *= -1
            turn += 0.5

            outcome = game.check_game_over(root.s)

            if training: training_set[-1].append([game.generate_game_state(root), pi])

        game_count += 1
        starts *= -1
        outcomes[outcome] += 1
        print(f"Game outcome was: {outcome}")
        print(f"Amount of games played is now: {game_count} \n")
        if training:
            [Set.append(outcome) for Set in training_set[-1]]
            training_set.append([])
    
    print(f"Results from self_play were: {outcomes}")
    return training_set[:-1] if training else outcomes

def retrain_network(agent, batch):
    for _ in range(config.training_iterations):
        positions = []
        for position in batch: positions += position
        minibatch = random.sample(positions, config.batch_size)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)
    agent.nn.save_progress()
    agent.nn.plot_losses(True)

    return (x, y)

def evaluate_network(agents, best_agent):
    results = self_play(agents, config.game_amount_evaluation, False)
    print(f"The results were: {results}")
    if results[-best_agent]/(results[best_agent] if results[best_agent] != 0 else 0.1) > config.winning_threshold:
        best_agent *= -1
        print(f"{best_agent} is now best player!")
        agents[1].nn.save_progress(best_agent)

    log(results, best_agent)
    return best_agent
    
def play_test(agent, games):
    you = User()
    results = self_play([None, agent, you], games, False)
    print(f"The results were: {results}")
    if results[1] > results[2]: print("You were worse than the bot")
    elif results[2] > results[1]: print("You were better than the bot")
    else: print("You tied with the bot")

def log(results, best_agent):
    open("log.txt", "a").write(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Results are: {results}\nBest_agent is now: {best_agent}\n")

for _ in range(config.loop_iterations):
    copyAgent = copy.copy(agents[best_agent])
    batch = self_play([None, agents[best_agent], copyAgent], config.game_amount_self_playing, True)
    (x, y) = retrain_network(agents[-best_agent], batch)
    best_agent = evaluate_network(agents, best_agent)
    play_test(agents[best_agent], config.game_amount_play_test)

for agent in agents: agent.nn.plot_losses()
while True: exec(input("do something: "))





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