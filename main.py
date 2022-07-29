import numpy as np
import matplotlib.pyplot as plt
import random
import json
import datetime
import config
import game
from player import *

load = [False, False, False]
agents = {1: Agent(load[0], 1), -1: Agent(load[1], 2)}

loads = list(np.where(load)[0])
best_agent = 1 if not loads else json.loads(open(f"{config.save_folder}save.json", "r").read())["best_agent"] if len(loads) == 2 else 1 - 2 * int(loads[0])
if not load[2]:
    open(f"{config.save_folder}log.txt", "w").truncate(0)
    open(f"{config.save_folder}positions.json", "w").write(json.dumps([]))

loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())
loaded["best_agent"] = best_agent
for agent in agents.values():
    if not agent.nn.load:
        empty = json.loads(open(f"{config.save_folder}empty_save.json", "r").read())[f"agent_{agent.nn.name}"]
        loaded[f"agent_{agent.nn.name}"] = empty
open(f"{config.save_folder}save.json", "w").write(json.dumps(loaded))

def setup_mcts(players, starts):
    for player in players.values(): player.mcts = Node(np.zeros(np.prod(game.game_dimensions))[::], None, None, starts, None)

def play(players, games, training):
    game_count = 0
    outcomes = [0, 0, 0]
    starts = 1
    while game_count < games:
        setup_mcts(players, starts)
        action = None
        player_turn = starts
        turn = 1
        tau = 1 if training else 1e-2
        if training: training_set = []
        outcome = None
        while outcome is None:
            if turn == config.turns_until_tau: tau = 1e-2
            action, pi = players[player_turn].play_turn(action, tau)

            outcome = game.check_game_over(players[player_turn].mcts.s)
            if training: training_set.append([players[-player_turn].mcts, pi])

            turn += 1
            player_turn *= -1

        game_count += 1
        outcomes[outcome] += 1
        starts *= -1

        print(f"We are " + ("training" if training else "evaluating"))
        print(f"Game outcome was: {outcome}")
        print(f"Amount of games played is now: {game_count}\n")

        if training:
            positions = [[game.generate_game_state(position[0], mirror).tolist()] + [game.mirror_board(position[1].tolist()) if mirror else position[1].tolist()] + [outcome * position[0].player] for position in training_set for mirror in [False, True]]
            loaded = json.loads(open(f"{config.save_folder}positions.json", "r").read())
            loaded += positions
            loaded = loaded[-config.position_amount:]
            print(f"Positions length is now {len(loaded)}\n")
            open(f"{config.save_folder}positions.json", "w").write(json.dumps(loaded))
            if len(loaded) != config.position_amount and game_count == games: games += 1

    return outcomes

def self_play(agent):
    copyAgent = Agent(agent.nn.load, agent.nn.name)
    results = play({1: agent, -1: copyAgent}, config.game_amount_self_play, True)

    print(f"The results from self-play were: {results}")

def retrain_network(agent):
    for _ in range(config.training_iterations):
        minibatch = random.sample(json.loads(open(f"{config.save_folder}positions.json", "r").read()), config.batch_size)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch], dtype="float64"), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)

    agent.nn.save_progress()

    return (x, y)

def evaluate_network(agents, best_agent):
    results = play(agents, config.game_amount_evaluation, False)
    print(f"The results were: {results}")
    if results[-best_agent] > results[best_agent] * config.winning_threshold:
        best_agent *= -1
        print(f"{best_agent} is now best player!")
        agents[1].nn.save_progress(best_agent)

    log(agents, results, best_agent)

    return best_agent
    
def play_test(agent, games):
    you = User()
    agents = {1: agent, -1: you}
    results = play(agents, games, False)

    best = agents[np.argmax(results[1:])].full_name
    print(f"The results were: {results}")
    log(agents, results, best)

def play_versions(versions, games):
    agents = {1 - 2 * i: Agent(True, name, version = v) for i, (name, v) in enumerate(versions)}
    results = play(agents, games, False)
    
    print(f"The results between {versions[0]} and {versions[1]} were: {results}")
    best = versions[np.argmax(results[1:])]
    print(f"The best version was: {best}")
    log(agents, results, best)

def plot_metrics_vertical(agents, show_lines):
    loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())

    fig, axs = plt.subplots(2, 2, figsize=(25, 7))
    plt.xlabel("Training Iteration")

    for i, agent in enumerate(agents.values()):
        for metric in loaded[f"agent_{agent.nn.name}"]["metrics"]:
            ax_index = 0 if metric.find("loss") != -1 else 1
            ax = axs[ax_index, i]
            ax.plot(loaded[f"agent_{agent.nn.name}"]["metrics"][metric], label=metric)
            # if not ax.get_title():
    
    for ax_index, metric in enumerate(["Loss", "Accuracy"]):
        for i, agent in enumerate(agents.values()):
            ax = axs[ax_index, i]
            ax.set_title(f"{agent.nn.name}: {metric}")
            ax.set_ylabel(metric)
            box = ax.get_position()
            ax.set_position([box.x0 * ([0.6, 1] * 2)[i], box.y0, box.width, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            if show_lines:
                iterations = loaded[f"agent_{agent.nn.name}"]["iterations"]
                [ax.axvline(np.sum(iterations[:i2 + 1]) - 1, color="black") for i2 in range(len(iterations))]

    plt.savefig(f"{config.save_folder}metrics.png", dpi=600)
    plt.close(fig)

def plot_metrics_horizontal(agents, show_lines):
    loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())

    fig, axs = plt.subplots(4, figsize=(15, 15))
    plt.xlabel("Training Iteration")

    for i, agent in enumerate(agents.values()):
        for metric in loaded[f"agent_{agent.nn.name}"]["metrics"]:
            ax_index = 0 if metric.find("loss") != -1 else 1
            ax = axs[ax_index + i * 2]
            ax.plot(loaded[f"agent_{agent.nn.name}"]["metrics"][metric], label=metric)
            # if not ax.get_title():
    
    for i, agent in enumerate(agents.values()):
        for ax_index, metric in enumerate(["Loss", "Accuracy"]):
            ax = axs[ax_index + i * 2]
            ax.set_title(f"{agent.nn.name}: {metric}")
            ax.set_ylabel(metric)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            if show_lines:
                iterations = loaded[f"agent_{agent.nn.name}"]["iterations"]
                [ax.axvline(np.sum(iterations[:i2 + 1]) - 1, color="black") for i2 in range(len(iterations))]

    plt.savefig(f"{config.save_folder}metrics.png", dpi=600)
    plt.close(fig)

def log(agents, results, best_agent):
    message = f"""{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}:
------------------ {agents[1].get_full_name()} vs {agents[-1].get_full_name()} ------------------
Results are: {results}
best_agent is: {best_agent}
------------------------------------------------------
"""
    open(f"{config.save_folder}log.txt", "a").write(message)

for _ in range(config.loop_iterations):
    self_play(agents[best_agent])
    (x, y) = retrain_network(agents[-best_agent])
    plot_metrics_vertical(agents, False)
    best_agent = evaluate_network(agents, best_agent)

play_versions([(1, 0), (2, 0)], 20)
play_test(agents[best_agent], config.game_amount_play_test)
