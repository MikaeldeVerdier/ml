import numpy as np
import random
import json
import datetime
import config
import game
from player import *

load = [False, False]
agents = {1: Agent(load[0], 1), -1: Agent(load[1], 2)}

loads = list(np.where(load)[0])
best_agent = 1 if not loads else json.loads(open(f"{config.save_folder}save.json", "r").read())["best_agent"] if len(loads) == 2 else 1 - 2 * int(loads[0])
if not load[0] or not load[1]:
    open(f"{config.save_folder}log.txt", "w").truncate(0)
    open(f"{config.save_folder}positions.json", "w").write(json.dumps([]))

loaded = json.loads(open(f"{config.save_folder}save.json", "r").read())
for agent in agents.values():
    if not agent.nn.load:
        empty = json.loads(open(f"{config.save_folder}empty_save.json", "r").read())[f"agent_{agent.nn.name}"]
        loaded[f"agent_{agent.nn.name}"] = empty
open(f"{config.save_folder}save.json", "w").write(json.dumps(loaded))

def setup_mcts(players, starts):
    for player in players.values(): player.mcts = Node(np.zeros(np.prod(config.game_dimensions))[::], None, None, starts, None)

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
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)

    agent.nn.save_progress()
    agent.nn.plot_losses(False)

    return (x, y)

def evaluate_network(agents, best_agent):
    results = play(agents, config.game_amount_evaluation, False)
    print(f"The results were: {results}")
    if results[-best_agent] > results[best_agent] * config.winning_threshold:
        best_agent *= -1
        print(f"{best_agent} is now best player!")
        agents[1].nn.save_progress(best_agent)

    log(results, best_agent)
    return best_agent
    
def play_test(agent, games):
    you = User()
    results = play({1: agent, -1: you}, games, False)
    print(f"The results were: {results}")
    if results[1] > results[2]:
        log(results, "the bot")
        print("You were worse than the bot")
    elif results[2] > results[1]:
        log(results, "you")
        print("You were better than the bot")
    else:
        log(results, "both")
        print("You tied with the bot")

def play_versions(versions, games):
    agents = {1 - 2 * i: Agent(True, name, version = v) for i, (name, v) in enumerate(versions)}
    results = play(agents, games, False)
    print(f"The results between {versions[0]} and {versions[1]} were: {results}")
    best = versions[np.argmax(results[1:])]
    print(f"The best version was: {best}")
    log(results, best)

def log(results, best_agent):
    open(f"{config.save_folder}log.txt", "a").write(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Results are: {results}\nBest_agent is now: {best_agent}\n")

for _ in range(config.loop_iterations):
    self_play(agents[best_agent])
    (x, y) = retrain_network(agents[-best_agent])
    best_agent = evaluate_network(agents, best_agent)

play_versions([(1, 0), (2, 0)], 20)
play_test(agents[best_agent], config.game_amount_play_test)
