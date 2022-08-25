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
    load = False

    files.setup_files()
    if not load:
        files.reset_file("save.json")
        files.reset_file("positions.json")
        files.reset_file("log.txt")

    current_agent = Agent(CurrentNeuralNetwork, load)
    best_agent = Agent(BestNeuralNetwork, load)
    agents = {1: best_agent, -1: current_agent}

    return agents


def play(players, games, training):
    players = set(players.values())

    game_count = 0
    starts = 1
    while game_count < games:
        for i, player in enumerate(players):
            player.mcts = Node(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], list(range(1, game.GAME_DIMENSIONS[0] + 1)), Tree())
    
            turn = 1
            if training:
                training_set = []
                tau = 1
            else: tau = 1e-2
            outcome = None
            while outcome is None:
                if turn == config.TURNS_UNTIL_TAU: tau = 1e-2
                if training: training_set.append([player.mcts])

                pi = player.play_turn(tau)

                if training: training_set[-1].append(pi)
                outcome = game.check_game_over(player.mcts)

                turn += 1

            game_count += 1
            player.outcome_len += 1
            player.average_outcome = (player.average_outcome * (player.outcome_len - 1) + outcome) / player.outcome_len
            starts *= -1

            print(f"We are " + ("training" if training else "evaluating"))
            print(f"Game outcome was: 1/{(1 / outcome):} = {outcome:.5f} (Agent: {i})")
            print(f"Amount of games played is now: {game_count}\n")

            if training:
                positions = [[np.array(game.generate_tutorial_game_state((position[0],))).tolist()] + [position[1].tolist()] + [outcome] for position in training_set]
                len_file, recent = files.add_to_file("positions.json", positions, config.POSITION_AMOUNT)
                print(f"Position length is now: {len_file}")

                is_full = len_file == config.POSITION_AMOUNT
                if is_full and recent: files.make_backup("positions.json", f"positions_{config.POSITION_AMOUNT}.json")
                
                if not is_full and game_count == games: games += 1


def self_play(agent):
    outcome = play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    print(f"The average outcome from self-play was: {outcome}")


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
    agent.reset_outcomes()
    agent.nn.plot_metrics(False, False)


def evaluate_network(agents):
    agents = [agent for agent in agents if agent.outcome_len != config.GAME_AMOUNT_EVALUATION]
    play(agents, config.GAME_AMOUNT_EVALUATION, False)
    
    results = [agent.average_outcome for agent in agents]

    log(agents, results)
    print(f"The results were: {results}")
    if results[1] > results[0] * config.WINNING_THRESHOLD:
        agents[-1].reset_outcomes()
        agents[1].nn.copy_weights(agents[-1].nn)
        agents[-1].nn.save_to_file()
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
