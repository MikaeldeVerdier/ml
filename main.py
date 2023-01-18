import numpy as np
import random
import environment
import config
import files
from datetime import datetime
from player import User, Agent
from environment import Environment

def initiate():
    load = False

    files.setup_files()
    if not load:
        files.reset_file("save.json")
        files.reset_file("positions.npy")
        files.reset_file("log.txt")

    agent = Agent(load=load, trainable=True)

    return agent


def play(env, games, training=False):
    if training:
        length = 0
        product = []
    else:
        results = [0 for _ in range(len(env.players))]

    og_games = games
    game_count = 0
    while game_count < games:
        game_count += 1

        env.reset()

        q_values = [[] for _ in range(len(env.players))]
        storage = []

        while env.game_state.outcome is None:
            storage.append({"state": env.game_state})

            probs, action = env.players[env.game_state.turn].get_action(env.game_state, env.epsilons[env.game_state.turn])
            
            q_values[env.game_state.turn].append(probs[action])

            env.step(probs, action)

            if training:
                for i2, var in enumerate(["action", "reward"]):
                    storage[-1][var] = [action, env.game_state.outcome or 0.0][i2]

        for i, player in enumerate(env.players):
            player.main_nn.metrics["average_q_value"].append(float(np.mean(q_values[i])))

        if not game_count % games:
            print(f"Amount of games played is now: {game_count} ({player.get_name()})\n")

        if not training:
            formatted_outcome = int(env.game_state.outcome / environment.REWARD_FACTOR)
            results[i] += formatted_outcome
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
            if left and games == game_count:
                games += np.ceil(left / (environment.GAME_LENGTH * 16) % og_games)

    if not training:
        for i, player in enumerate(env.players):
            results[i] /= games
            player.main_nn.metrics["outcomes"][player.main_nn.version] = results[i]

        return results


def self_play(agent):
    print("\nSelf-play started!\n")
    play(Environment([agent]), config.GAME_AMOUNT_SELF_PLAY, training=True)


def retrain_network(agent):
    print("\nRetraining started!\n")

    positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

    for _ in range(config.TRAINING_ITERATIONS):
        minibatch = random.sample(positions, config.BATCH_SIZE[0])

        x = [[] for _ in range(len(environment.NN_INPUT_DIMENSIONS))]
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

    outcome = play(Environment([agent], epsilons=[0.05]), config.GAME_AMOUNT_EVALUATION)[0]

    # log([agent], outcome)

    print(f"The result was: {outcome}")


def log(agent_s, average_s):
    message = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {' vs '.join([agent.get_name() for agent in agent_s])}:\nAverage results were {average_s}\n"
    files.write("log.txt", message, "a")


def play_test(load, games, starts=1):
    you = User()
    agents = [Agent(verbose=True, load=load), you]
    outcomes = play(agents, games, starts=starts, epsilons=[0.05])

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
