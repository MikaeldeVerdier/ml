import json
import numpy as np
import random
import datetime
import game
import config
import files
from nn import NeuralNetwork, BestNeuralNetwork, CurrentNeuralNetwork
from mcts import Node, Tree
from player import User, Agent


def initiate():
    load = [False, False]

    files.setup_files()
    if not any(load):
        files.reset_file("save.json")
        files.reset_file("positions.json")
        files.reset_file("log.txt")

    current_agent = Agent(CurrentNeuralNetwork, load[0])
    best_agent = Agent(BestNeuralNetwork, load[1])
    agents = {1: best_agent, -1: current_agent}

    return agents


def play(players, games, training):
    players = set(players.values())

    if training: training_data = []
    loaded = files.load_file("positions.json")

    game_count = 0
    starts = 1
    while game_count < games:
        for i, player in enumerate(players):
            player.mcts = Node(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], list(range(1, game.GAME_DIMENSIONS[0] + 1)), Tree())
    
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

                if training:
                    length = len(np.where(pi != 0)[0])
                    if length > 1:
                        training_data[-1][-1].append(pi)
                    else: del training_data[-1][-1] 
                outcome = game.check_game_over(player.mcts)

                turn += 1

            game_count += 1
            starts *= -1

            print(f"We are " + ("training" if training else "evaluating"))
            print(f"Game outcome was: 1/{(1 / outcome):} = {outcome:.5f} (Agent: {i})")
            print(f"Amount of games played is now: {game_count}\n")

            if not training:
                player.outcomes["length"] += 1
                player.outcomes["average"] = (player.outcomes["average"] * (player.outcomes["length"] - 1) + outcome) / player.outcomes["length"]
                
                if outcome in player.outcomes["piles"]: player.outcomes["piles"][round(1 / outcome)] += 1  # int() is maybe redundant but w/e
                else: player.outcomes["piles"][round(1 / outcome)] = 0
            else: 
                for data in training_data[-1]: data.append(outcome)

                training_length = len(np.vstack(training_data))

                if len(loaded) + training_length < config.POSITION_AMOUNT:
                    if len(loaded) + training_length < config.POSITION_AMOUNT and games == game_count: games += 1

                away_from_full = config.POSITION_AMOUNT - len(loaded)
                if training_length > config.POSITION_AMOUNT / 25 or away_from_full and training_length >= away_from_full:
                    for game_data in training_data:
                        for data in game_data:
                            data[0] = np.array(game.generate_tutorial_game_state(data[0])).tolist()
                            data[1] = data[1].tolist()
                    training_data = np.vstack(training_data).tolist()
                    
                    recent = len(loaded) != config.POSITION_AMOUNT
                    loaded += training_data
                    loaded = loaded[-config.POSITION_AMOUNT:]
                    files.write("positions.json", json.dumps(loaded))

                    is_full = len(loaded) == config.POSITION_AMOUNT
                    if is_full and recent: files.make_backup("positions.json", f"positions_{config.POSITION_AMOUNT}.json")

                    training_data = []
                
                print(f"Position length is now: {len(loaded) + training_length}")


def self_play(agent):
    play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    # outcome = agent.outcomes["average"]
    # print(f"The average outcome from self-play was: {outcome}")


def retrain_network(agent):
    for _ in range(config.TRAINING_ITERATIONS):
        positions = files.load_file("positions.json")
        minibatch = random.sample(positions, config.BATCH_SIZE)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch]), "policy_head": np.array([batch[1] for batch in minibatch])}

        agent.nn.train(x, y)

    agent.nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
    agent.nn.version += 1
    agent.nn.model.save(f"{config.SAVE_PATH}/training/v.{agent.nn.version}")
    agent.nn.save_to_file("current_agent")
    agent.outcomes = {"piles": {}, "average": 0, "length": 0}
    agent.nn.plot_metrics(False, False)


def evaluate_network(agents):
    play(agents, config.GAME_AMOUNT_EVALUATION, False)
    
    results = [agent.outcomes["average"] for agent in agents.values()]

    log(agents, results)
    print(f"The results were: {results}")
    if results[1] > results[0] * config.WINNING_THRESHOLD:
        agents[1].copy_profile(agents[-1])
        agents[-1].nn.save_to_file("best_agent")
        print(f"The best_agent has copied the current_agent's weights")

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
    results = play(agents, games, False)

    print(f"The results were: {results}")
    log(agents, results)


def play_versions(versions, games):
    agents = {1 - 2 * i: Agent(NeuralNetwork, True, version=v) for i, v in enumerate(versions)}
    results = play(agents, games, False)
    
    print(f"The results between versions {versions[0]} and {versions[1]} were: {results}")
    best = versions[np.argmax(results)]
    print(f"The best version was: version {best}")
    log(agents, results)


def main():
    agents = initiate()

    for i in range(config.LOOP_ITERATIONS):
        print(f"Now starting main loop iteration: {i}")
        self_play(agents[1])
        retrain_network(agents[-1])
        if i % config.EVALUATION_FREQUENCY == 0: agents = evaluate_network(agents)

    # play_versions([1, agents[1].nn.version], config.GAME_AMOUNT_PLAY_VERSIONS)
    # play_test(agents[1].nn.version, config.GAME_AMOUNT_PLAY_TEST)
    # print(files.add_to_file("positions.json", files.load_file("poss.json"), config.POSITION_AMOUNT)[0])


if __name__ == "__main__":
    main()
