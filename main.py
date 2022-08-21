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
    load = True

    files.setup_files()
    if not load:
        files.reset_file("save.json")
        files.reset_file("positions.json")
        files.reset_file("log.txt")

    best_agent = Agent(BestNeuralNetwork, load)
    current_agent = Agent(CurrentNeuralNetwork, load)
    agents = {1: best_agent, -1: current_agent}

    return agents


def setup_mcts(players):
    for player in players: player.mcts = Node(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], 1, Tree())


def play(players, games, training):
    game_count = 0
    outcomes = [0, 0, 0]
    starts = 1
    while game_count < games:
        setup_mcts(set(players.values()))
        action = None
        player_turn = starts
        turn = 1
        if training:
            tau = 1
            training_set = []
        else: tau = 1e-2
        outcome = None
        while outcome is None:
            if turn == config.TURNS_UNTIL_TAU: tau = 1e-2
            if training: training_set.append([players[player_turn].mcts])

            action, pi = players[player_turn].play_turn(action, tau)

            if training: training_set[-1].append(pi)
            outcome = game.check_game_over(players[player_turn].mcts.s)

            turn += 1
            player_turn *= -1

        game_count += 1
        outcomes[outcome * starts] += 1
        starts *= -1

        print(f"We are " + ("training" if training else "evaluating"))
        print(f"Game outcome was: {outcome} (Agent: {outcome * -starts})")
        print(f"Amount of games played is now: {game_count}\n")

        if training:
            positions = [[game.generate_tutorial_game_state((position[0],), mirror).tolist()] + [game.mirror_board(position[1].tolist()) if mirror else position[1].tolist()] + [outcome * position[0].player] for position in training_set for mirror in [False, True]]
            len_file, recent = files.add_to_file("positions.json", positions, config.POSITION_AMOUNT)
            print(f"Position length is now: {len_file}")

            is_full = len_file == config.POSITION_AMOUNT
            if is_full and recent: files.make_backup("positions.json", f"positions_{config.POSITION_AMOUNT}.json")
            
            if not is_full and game_count == games: games += 1

    return outcomes


def self_play(agent):
    results = play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    print(f"The results from self-play were: {results}")


def retrain_network(network):
    for _ in range(config.TRAINING_ITERATIONS):
        positions = files.load_file("positions.json")
        minibatch = random.sample(positions, config.BATCH_SIZE)

        x = np.array([batch[0] for batch in minibatch])
        y = {"value_head": np.array([batch[2] for batch in minibatch], dtype="float64"), "policy_head": np.array([batch[1] for batch in minibatch])}

        network.train(x, y)

    network.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
    network.version += 1
    network.plot_metrics(False, False)


def evaluate_network(agents):
    results = play(agents, config.GAME_AMOUNT_EVALUATION, False)

    log(agents, results)
    print(f"The results were: {results}")
    if results[-1] > results[1] * config.WINNING_THRESHOLD:
        agents[1].nn.copy_weights(agents[-1].nn)
        agents[-1].nn.save_to_file()
        print(f"The best_agent has copied the current_agent's weights")

    return agents


def log(agents, results):
    names = [agents[1].get_name(), agents[-1].get_name()]

    best_name = names[np.argmax(results[1:])]
    best = f"{best_name[0]} {best_name[1]}" if results[1] != results[2] else "They both are" 
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
    best = versions[np.argmax(results[1:])]
    print(f"The best version was: version {best}")
    log(agents, results)


def main():
    agents = initiate()

    for i in range(config.LOOP_ITERATIONS):
        print(f"Now starting main loop iteration: {i}")
        self_play(agents[1])
        retrain_network(agents[-1].nn)
        agents = evaluate_network(agents)

    # play_versions([1, agents[1].nn.version], config.GAME_AMOUNT_PLAY_VERSIONS)
    # play_test(agents[1].nn.version, config.GAME_AMOUNT_PLAY_TEST)
    # files.add_to_file("positions.json", files.load_file("poss.json"), config.POSITION_AMOUNT)


if __name__ == "__main__":
    main()
