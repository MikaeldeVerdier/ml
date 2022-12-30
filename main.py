import os
import numpy as np
import random
import game
import config
import files
from game import GameState
from player import User, Agent

def initiate():
    load = False

    files.setup_files()
    if not load:
        files.reset_file("save.json")
        files.reset_file("positions.npy")
        files.reset_file("log.txt")

    agent = Agent(load, to_weights=True)

    return agent


def play(players, games, starts=False, epsilons=[None, None], training=False):
    if training:
        length = 0
        product = []
    else:
        result = [[] for _ in range(len(players))]

    og_games = games
    game_count = 0
    while game_count < games:
        game_count += 1
        
        deck = list(range(1, 53))
        random.shuffle(deck)
        drawn_card = deck.pop()
        for i, player in sorted(enumerate(players), reverse=starts):
            player.game_state = GameState(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], deck, drawn_card)

            storage = []

            outcome = None
            while outcome is None:
                storage.append({"state": player.game_state})

                action = player.play_turn(storage, epsilons[i])

                outcome = game.check_game_over(player.game_state)

                if training:
                    for i2, var in enumerate(["action", "reward", "next_state"]):
                        storage[-1][var] = [action, 0.0 if outcome is None else outcome, player.game_state][i2]

            starts ^= starts

            if not game_count % games:
                print(f"Amount of games played is now: {game_count} ({player.get_name()})\n")

            if not training:
                outcome = int(outcome / game.REWARD_FACTOR)
                result[i].append(outcome)
                player.main_nn.register_result(outcome)
            else:
                for t, data in enumerate(storage):
                    data["target"] = player.calculate_target(storage, t) if t != len(storage) - 1 else data["reward"]

                    game_states = game.generate_game_states(storage, t)

                    states = np.array(game.generate_nn_pass(game_states, True), dtype=object).tolist()
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
                        games += np.ceil(left / (game.GAME_LENGTH * 16 * left // (game.GAME_LENGTH * 16 * og_games)))

    if not training:
        return result


def self_play(agent):
    print("\nSelf-play started!\n")
    play([agent], config.GAME_AMOUNT_SELF_PLAY, training=True)


def retrain_network(agent):
    print("\nRetraining started!\n")

    positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

    for _ in range(config.TRAINING_ITERATIONS):
        minibatch = random.sample(positions, config.BATCH_SIZE[0])

        x = [[] for _ in range(len(game.NN_INPUT_DIMENSIONS))]
        y = []

        for position in minibatch:
            y.append(position[1:])
            for i, dim in enumerate(position[0]):
                x[i].append(np.array(dim))

        for i, var in enumerate(x):
            x[i] = np.array(var)
        y = np.array(y)

        agent.main_nn.train(x, y)

    if not (agent.main_nn.version - 1) % config.VERSION_OFFSET:
        agent.copy_network()

    # data = [np.expand_dims(dat, 0) for dat in positions[-1][0]]
    # real = positions[-1]
    # p = agent.main_nn.model.predict(data)

    agent.change_version()
    agent.main_nn.plot_agent()


def evaluate_network(agent):
    print("\nEvaluation of agent started!\n")

    outcome = play([agent], config.GAME_AMOUNT_EVALUATION, epsilons=[0.05])
    agent.main_nn.save_outcomes()

    outcome = np.mean(outcome)

    log(agent, outcome)
    agent.main_nn.plot_agent()

    print(f"The result was: {outcome}")


def log(agent, result):
    message = f"{agent.get_name()} had an average score of: {result}\n"
    files.write("log.txt", message, "a")


def play_test(version, games, starts=False):
    you = User()
    agents = [Agent(True, version=version), you]
    results = play(agents, games, starts=starts)

    print(f"The results were: {results}")
    log(agents, results)


def play_versions(versions, games, starts=False):
    agents = [Agent(True, version=version) for version in versions]
    results = play(agents, games, starts=starts, epsilons=[0.05, 0.05])

    outcomes = np.mean(results)
    
    print(f"The results between versions {versions[0]} and {versions[1]} were: {outcomes}")
    best = versions[np.argmax(outcomes)]
    print(f"The best version was: version {best}")


def main():
    agent = initiate()

    for _ in range(config.LOOP_ITERATIONS):
        self_play(agent)
        retrain_network(agent)
        if not (agent.main_nn.version - 1) % config.EVALUATION_FREQUENCY:
            evaluate_network(agent)

    # play_versions([1, agents[1].nn.version], config.GAME_AMOUNT_PLAY_VERSIONS)
    # play_test(agents[1].nn.version, config.GAME_AMOUNT_PLAY_TEST)
    # print(files.add_to_file("positions.json", files.load_file("poss.json"), config.POSITION_AMOUNT)[0])


if __name__ == "__main__":
    main()
