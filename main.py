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
best_agent = 1 if not loads else json.loads(open("save.json", "r").read())["best_agent"] if len(loads) == 2 else 2 * loads[0] - 1
if not load[0] or not load[1]: open("log.txt", "w").truncate(0)

def setup_mcts(players, starts):
    for player in players.values(): player.mcts = Node(np.zeros(np.prod(config.game_dimensions))[::], None, None, starts, None)

def play(players, games, training):
    if training: training_set = [[]]
    outcomes = [0, 0, 0]
    game_count = 0
    starts = 1
    while game_count < games:
        setup_mcts(players, starts)
        action = None
        player_turn = starts
        turn = 1
        tau = 1 if training else 1e-2
        outcome = None
        while outcome is None:
            if turn > config.turns_until_tau: tau = 1e-2
            player = players[player_turn]
            action, pi = player.play_turn(action, tau)
            player_turn *= -1
            turn += 0.5

            outcome = game.check_game_over(player.mcts.s)

            if training: training_set[-1].append([player.mcts, pi])

        game_count += 1
        starts *= -1
        outcomes[outcome] += 1
        print(f"Game outcome was: {outcome}")
        print(f"Amount of games played is now: {game_count} \n")
        print(f"We are " + ("training" if training else "evaluating"))
        if training:
            [position.append(outcome * position[0].player) for position in training_set[-1]]
            training_set.append([])
    
    return training_set[:-1] if training else outcomes

def self_play(agent):
    copyAgent = Agent(agent.nn.load, agent.nn.name)
    training_data = play({1: agent, -1: copyAgent}, config.game_amount_self_playing, True)
    
    return training_data

def retrain_network(agent, batch):
    for _ in range(config.training_iterations):
        positions = []
        for position in batch: positions += position
        minibatch = random.sample(positions, config.batch_size)

        x = np.array([game.generate_game_state(batch[0]) for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)
    agent.nn.save_progress()
    agent.nn.plot_losses(False)

    return (x, y)

def evaluate_network(agents, best_agent):
    results = play(agents, config.game_amount_evaluation, False)
    print(f"The results were: {results}")
    if results[-best_agent]/(results[best_agent] if results[best_agent] != 0 else 0.1) > config.winning_threshold:
        best_agent *= -1
        print(f"{best_agent} is now best player!")
        agents[1].nn.save_progress(best_agent)

    log(results, best_agent)
    return best_agent
    
def play_test(agent, games):
    you = User()
    results = play({1: agent, -1: you}, games, False)
    print(f"The results were: {results}")
    if results[1] > results[2]: print("You were worse than the bot")
    elif results[2] > results[1]: print("You were better than the bot")
    else: print("You tied with the bot")

def log(results, best_agent):
    open("log.txt", "a").write(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Results are: {results}\nBest_agent is now: {best_agent}\n")

for _ in range(config.loop_iterations):
    batch = self_play(agents[best_agent])
    (x, y) = retrain_network(agents[-best_agent], batch)
    best_agent = evaluate_network(agents, best_agent)

play_test(agents[best_agent], config.game_amount_play_test)
