import numpy as np
import matplotlib.pyplot as plt
import random
import json
import datetime
import time
import config
import game
from player import *

def initiate():
    load = [False, False, False]
    agents = {1: Agent(load[0], 1), -1: Agent(load[1], 2)}
    loads = list(np.where(load[:-1])[0])

    if not load[2]:
        with open(f"{config.SAVE_FOLDER}log.txt", "w") as log: log.truncate(0)
        with open(f"{config.SAVE_FOLDER}positions.json", "w") as positions: positions.write(json.dumps([]))

    with open(f"{config.SAVE_FOLDER}save.json", "r") as save_r:
        loaded = json.loads(save_r.read())

        if not loads: best_agent = 1
        elif len(loads) == 1: best_agent = 1 - 2 * int(loads[0])
        else: best_agent = loaded["best_agent"]

        loaded["best_agent"] = best_agent
        for agent in agents.values():
            if not agent.nn.load:
                with open(f"{config.SAVE_FOLDER}empty_save.json", "r") as empty_save:
                    empty = json.loads(empty_save.read())[f"agent_{agent.nn.name}"]
                    loaded[f"agent_{agent.nn.name}"] = empty
        with open(f"{config.SAVE_FOLDER}save.json", "w") as save_w: save_w.write(json.dumps(loaded))

    return agents, best_agent

def setup_mcts(players, starts):
    for player in players.values(): player.mcts = Node(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], None, None, starts, None)

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
            if turn == config.TURNS_UNTIL_TAU: tau = 1e-2
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
            with open(f"{config.SAVE_FOLDER}positions.json", "r") as positions_r:
                positions = [[game.generate_game_state(position[0], mirror).tolist()] + [game.mirror_board(position[1].tolist()) if mirror else position[1].tolist()] + [outcome * position[0].player] for position in training_set for mirror in [False, True]]
                loaded = json.loads(positions_r.read())
                loaded += positions
                loaded = loaded[-config.POSITION_AMOUNT:]
                print(f"Positions length is now {len(loaded)}\n")
                with open(f"{config.SAVE_FOLDER}positions.json", "w") as positions_w: positions_w.write(json.dumps(loaded))
            
            if len(loaded) != config.POSITION_AMOUNT and game_count == games: games += 1

    return outcomes

def self_play(agent):
    copyAgent = Agent(agent.nn.load, agent.nn.name)
    results = play({1: agent, -1: copyAgent}, config.GAME_AMOUNT_SELF_PLAY, True)

    print(f"The results from self-play were: {results}")

def retrain_network(agent):
    for _ in range(config.TRAINING_ITERATIONS):
        with open(f"{config.SAVE_FOLDER}positions.json", "r") as positions:
            minibatch = random.sample(json.loads(positions.read()), config.BATCH_SIZE)

            x = np.array([batch[0] for batch in minibatch])
            y = {"value_head": np.array([batch[2] for batch in minibatch], dtype="float64"), "policy_head": np.array([batch[1] for batch in minibatch])}

            agent.nn.train(x, y)

    agent.nn.save_progress()

def evaluate_network(agents, best_agent):
    results = play(agents, config.GAME_AMOUNT_EVALUATION, False)
    print(f"The results were: {results}")
    if results[-best_agent] > results[best_agent] * config.WINNING_THRESHOLD:
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

def plot_metrics_horizontal(agents, show_lines):
    with open(f"{config.SAVE_FOLDER}save.json", "r") as save:
        loaded = json.loads(save.read())

        _, axs = plt.subplots(4, 2, sharey="row", figsize=(25, 15))
        plt.xlabel("Training Iteration")

        for i, agent in enumerate(agents.values()):
            for metric in loaded[f"agent_{agent.nn.name}"]["metrics"]:
                ax_index = (2, 3) if "val_" in metric else (0, 1)
                ax_index = ax_index[0] if "loss" in metric else ax_index[1]

                ax = axs[ax_index, i]
                ax.plot(loaded[f"agent_{agent.nn.name}"]["metrics"][metric], label=metric)
        
        for ax_index, metric in enumerate(["Loss", "Accuracy", "Validation Loss", "Validation Accuracy"]):
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

        plt.show(block=False)
        plt.savefig(f"{config.SAVE_FOLDER}metrics.png", dpi=300)
        time.sleep(1)
        plt.pause(0.1)
        plt.close("all")

def log(agents, results, best_agent):
    message = f"""{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}:
------------------ {agents[1].get_full_name()} vs {agents[-1].get_full_name()} ------------------
Results are: {results}
best_agent is: {best_agent}
------------------------------------------------------
"""
    with open(f"{config.SAVE_FOLDER}log.txt", "a") as log: log.write(message)

def main():
    agents, best_agent = initiate()
    for _ in range(config.LOOP_ITERATIONS):
        self_play(agents[best_agent])
        retrain_network(agents[-best_agent])
        plot_metrics_horizontal(agents, False)
        best_agent = evaluate_network(agents, best_agent)

    play_versions([(1, 1), (2, 1)], 20)
    play_test(agents[best_agent], config.GAME_AMOUNT_PLAY_TEST)

if __name__ == "__main__":
    main()
