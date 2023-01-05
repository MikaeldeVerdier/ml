import os
import numpy as np
import random
import config
import files
from datetime import datetime
from player import User, Agent

def initiate():
    load = False

    files.setup_files()
    if not load:
        files.reset_file("save.json")
        files.reset_file("positions.npy")
        files.reset_file("log.txt")

    agent = Agent(load=load, trainable=True)

    return agent


def play(players, games, starts=0, epsilons=[None, None], training=False):
    if training:
        length = 0
        product = []
    else:
        results = [0 for _ in range(len(players))]

    og_games = games
    game_count = 0
    while game_count < games:
        game_count += 1

        for i, player in enumerate(players[starts - 1:] + players[:starts - 1]):
            player.env.reset()

            cumulative_q_value = 0
            storage = []

            outcome = None
            while outcome is None:
                storage.append({"state": player.env.game_state})

                action, q_value = player.get_action(epsilons[i])
                player.env.step(action)

                cumulative_q_value += q_value

                outcome = player.env.game_state.check_game_over()

                if training:
                    for i2, var in enumerate(["action", "reward"]):
                        storage[-1][var] = [action, 0.0 if outcome is None else outcome][i2]

            starts += 1
            starts %= len(players)

            player.main_nn.metrics["average_q_value"].append(cumulative_q_value / player.env.GAME_LENGTH)

            if not game_count % games:
                print(f"Amount of games played is now: {game_count} ({player.get_name()})\n")

            if not training:
                outcome = int(outcome / player.env.REWARD_FACTOR)
                results[i] += outcome
            else:
                for t, data in enumerate(storage):
                    data["target"] = player.calculate_target(storage, t) if t != len(storage) - 1 else data["reward"]

                    states = np.array(data["state"].generate_nn_pass(modify=True), dtype=object).tolist()
                    for flip in states: product.append(np.array([flip, data["action"], data["target"]], dtype=object))

                if not game_count % games:
                    length = files.add_to_file(files.get_path("positions.npy"), np.array(product, dtype=object), config.POSITION_AMOUNT)
                    product = []

                    print(f"Position length is now: {length}")

                left = config.POSITION_AMOUNT - length
                if not left:
                    if not os.path.exists(f"{config.SAVE_PATH}backup/positions_{config.POSITION_AMOUNT}.json"):
                        files.make_backup("positions.npy", f"positions_{config.POSITION_AMOUNT}.npy")
                else:
                    if games == game_count:
                        games += np.ceil(left / (player.env.GAME_LENGTH * 16) % og_games)

    if not training:
        for i, player in enumerate(players):
            results[i] /= games
            player.main_nn.metrics["outcomes"][player.main_nn.version] = results[i]

        return results


def self_play(agent):
    print("\nSelf-play started!\n")
    play([agent], config.GAME_AMOUNT_SELF_PLAY, training=True)


def retrain_network(agent):
    print("\nRetraining started!\n")

    positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

    for _ in range(config.TRAINING_ITERATIONS):
        minibatch = random.sample(positions, config.BATCH_SIZE[0])

        x = [[] for _ in range(len(agent.env.NN_INPUT_DIMENSIONS))]
        y = []

        for position in minibatch:
            y.append(position[1:])
            for i, dim in enumerate(position[0]):
                x[i].append(np.array(dim))

        for i, var in enumerate(x):
            x[i] = np.array(var)
        y = np.array(y)

        agent.main_nn.train(x, y)
    agent.change_version()

    # data = [np.expand_dims(dat, 0) for dat in positions[-1][0]]
    # real = positions[-1]
    # p = agent.main_nn.model.predict(data)


def evaluate_network(agent):
    print("\nEvaluation of agent started!\n")

    outcome = play([agent], config.GAME_AMOUNT_EVALUATION, epsilons=[0.05])[0]

    # log([agent], outcome)

    print(f"The result was: {outcome}")


def log(agent_s, average_s):
    message = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {' vs '.join([agent.get_name() for agent in agent_s])}:\nAverage results were {average_s}\n"
    files.write("log.txt", message, "a")


def play_test(load, games, starts=1):
    you = User()
    agents = [Agent(verbose=True, load=load), you]
    outcomes = play(agents, games, starts=starts)

    print(f"The results were: {outcomes}")


def play_versions(loads, games, starts=0):
    agents = [Agent(verbose=True, load=load) for load in loads]
    outcomes = play(agents, games, starts=starts, epsilons=[0.05, 0.05])

    log(loads, outcomes)

    print(f"The results between versions named {loads[0]} and {loads[1]} were: {outcomes}")
    best = loads[np.argmax(outcomes)].get_name()
    print(f"The best version was: version {best}")


def main():
    agent = initiate()

    for _ in range(config.LOOP_ITERATIONS):
        if not agent.main_nn.version % config.EVALUATION_FREQUENCY:
            evaluate_network(agent)
        self_play(agent)
        retrain_network(agent)

    # play_versions(["untrained_version", "trained version"], config.GAME_AMOUNT_PLAY_VERSIONS)
    # play_test("trained_version", config.GAME_AMOUNT_PLAY_TEST)
    # print(files.add_to_file("positions.json", files.load_file("poss.json"), config.POSITION_AMOUNT)[0])


if __name__ == "__main__":
    main()
