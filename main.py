import json
import os
import numpy as np
import datetime
import random
import game
import config
import files
from nn import NeuralNetwork, CurrentNeuralNetwork, BestNeuralNetwork
from mcts import Node, Tree
from player import User, Agent, CurrentAgent, BestAgent

def initiate():
    load = [False, False]

    files.setup_files()
    if not any(load):
        files.reset_file("save.json")
        files.reset_file("positions.json")
        files.reset_file("log.txt")

    current_agent = CurrentAgent(CurrentNeuralNetwork, load[0])
    best_agent = BestAgent(BestNeuralNetwork, load[1])
    agents = {1: best_agent, -1: current_agent}

    return agents


def play(players, games, training=False):
    players = set(players.values())

    if training: training_data = []
    loaded = files.load_file("positions.json")

    game_count = 0
    if not training: outcomes = [[], []]
    starts = 1
    while game_count < games:
        game_count += 1
        
        deck = list(range(1, 53))
        random.shuffle(deck)
        drawn_card = deck.pop()
        for i, player in enumerate(players):
            player.mcts = Node(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], deck, drawn_card, Tree())
    
            turn = 1
            if training:
                training_data.append([])
                tau = 1
            else: tau = 1e-2

            outcome = None
            while outcome is None:
                if turn == config.TURNS_UNTIL_TAU: tau = 1e-2
                if training: training_data[-1].append([player.mcts])

                pi = player.play_turn(tau)

                if training: training_data[-1][-1].append(pi)
                outcome = game.check_game_over(player.mcts)

                turn += 1

            starts *= -1

            print(f"We are " + ("training" if training else "evaluating"))
            print(f"Game outcome was: {outcome} (Agent name: {player.get_name()[0]})")
            print(f"Amount of games played for this agent is now: {game_count}\n")

            if not training:
                outcomes[i].append(outcome)
                player.outcomes["length"] += 1
                player.outcomes["average"] = (player.outcomes["average"] * (player.outcomes["length"] - 1) + outcome) / player.outcomes["length"]
            else:
                for data in training_data[-1]: data.append(outcome)

                # away_from_full = config.POSITION_AMOUNT - len(loaded)
                # if training_length > config.POSITION_AMOUNT / 25 or away_from_full and training_length >= away_from_full:
                product = []
                for game_data in training_data:
                    for data in game_data:
                        state = np.array(game.generate_tutorial_game_state(data[0], True)).tolist()
                        for flip in state: product.append([flip, data[1].tolist(), data[2]])
                        # data[0] = np.array(game.generate_tutorial_game_state(data[0])).tolist()
                        # data[1] = data[1].tolist()
                # training_data = np.vstack(training_data).tolist()

                loaded += product
                loaded = loaded[-config.POSITION_AMOUNT:]
                files.write("positions.json", json.dumps(loaded))

                is_full = len(loaded) == config.POSITION_AMOUNT
                if is_full:
                    if not os.path.exists(f"{config.SAVE_PATH}/backup/positions_{config.POSITION_AMOUNT}.json"): files.make_backup("positions.json", f"positions_{config.POSITION_AMOUNT}.json")
                else:
                    if games == game_count: games += 1

                training_data = []
            
                print(f"Position length is now: {len(loaded)}")
            
    if not training: return outcomes


def self_play(agent):
    play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    # outcome = agent.outcomes["average"]
    # print(f"The average outcome from self-play was: {outcome}")


def retrain_network(agent):
    for _ in range(config.TRAINING_ITERATIONS):
        positions = files.load_file("positions.json")
        minibatch = random.sample(positions, config.BATCH_SIZE)

        x = [[], [], []]
        y = {"value_head": [], "policy_head": []}
        for batch in minibatch:
            for i, dim in enumerate(batch[0]):
                x[i].append(np.array(dim))

            y["value_head"].append(np.array([batch[2]], dtype=np.float32))
            y["policy_head"].append(np.array(batch[1]))

        for i, dim in enumerate(x):
            x[i] = np.array(dim)
        
        y["value_head"] = np.array(y["value_head"])
        y["policy_head"] = np.array(y["policy_head"])

        # x["position_input"] = np.expand_dims(x["position_input"], 1)
        # x["deck_input"] = np.expand_dims(x["deck_input"], 1)
        # x["drawn_card_input"] = np.expand_dims(x["drawn_card_input"], 1)

        # y["policy_head"] = np.expand_dims(y["policy_head"], 1)
        # y["value_head"] = np.expand_dims(y["value_head"], 1)

        agent.nn.train(x, y)

    agent.nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
    agent.nn.version += 1
    agent.nn.model.save(f"{config.SAVE_PATH}/training/v.{agent.nn.version}")
    agent.nn.save_metrics("current_agent")
    agent.outcomes = {"average": 0, "length": 0}
    agent.nn.plot_metrics(derivative_lines=True)


def evaluate_network(agents):
    outcomes = play(agents, config.GAME_AMOUNT_EVALUATION)

    results = []
    for agent in agents.values():
        results.append(agent.outcomes["average"])
        agent.save_outcomes("current_agent")

    for i, player in enumerate(outcomes):
        outcomes[i] = np.sum(player) / len(player)

    log(agents, outcomes)
    agents[-1].nn.plot_outcomes(derivative_line=True)

    print(f"The results were: {outcomes}")
    if outcomes[1] > outcomes[0] * config.WINNING_THRESHOLD:
        agents[1].copy_profile(agents[-1])
        print(f"The best_agent has copied the current_agent's profile")

    return agents


def log(agents, results):
    names = [agents[1].get_name(), agents[-1].get_name()]

    best_name = names[np.argmax(results)]
    best = f"{best_name[0]} {best_name[1]}" if results[0] != results[1] else "They both are" 
    message = f"""{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}:
------------ {names[0][0]} vs {names[1][0]} ------------
Results are: {results}
{best} the best!

"""
    files.write("log.txt", message, "a")


def play_test(version, games):
    you = User()
    agents = {1: Agent(NeuralNetwork, True, version=version), -1: you}
    results = play(agents, games)

    print(f"The results were: {results}")
    log(agents, results)


def play_versions(versions, games):
    agents = {1 - 2 * i: Agent(NeuralNetwork, True, version=v) for i, v in enumerate(versions)}
    results = play(agents, games)
    
    print(f"The results between versions {versions[0]} and {versions[1]} were: {results}")
    best = versions[np.argmax(results)]
    print(f"The best version was: version {best}")
    log(agents, results)


def main():
    agents = initiate()

    for _ in range(config.LOOP_ITERATIONS):
        self_play(agents[1])
        retrain_network(agents[-1])
        if (agents[-1].nn.version - 1) % config.EVALUATION_FREQUENCY == 0: agents = evaluate_network(agents)

    # play_versions([1, agents[1].nn.version], config.GAME_AMOUNT_PLAY_VERSIONS)
    # play_test(agents[1].nn.version, config.GAME_AMOUNT_PLAY_TEST)
    # print(files.add_to_file("positions.json", files.load_file("poss.json"), config.POSITION_AMOUNT)[0])


if __name__ == "__main__":
    main()
