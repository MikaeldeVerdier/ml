import os
import numpy as np
import datetime
import random
import game
import config
import files
from nn import NeuralNetwork, CurrentNeuralNetwork, BestNeuralNetwork
from game_state import GameState
from player import User, Agent, CurrentAgent, BestAgent

def initiate():
    load = [False, False]

    files.setup_files()
    if not any(load):
        files.reset_file("save.json")
        files.reset_file("positions.npy")
        files.reset_file("log.txt")

    current_agent = CurrentAgent(CurrentNeuralNetwork, load[0], to_weights=True)
    best_agent = BestAgent(BestNeuralNetwork, load[1])
    agents = {1: best_agent, -1: current_agent}

    return agents


def play(players, games, training=False):
    players = sorted(set(players.values()))

    if training:
        length = 0
        product = []
    else:
        outcomes = [[], []]

    og_games = games
    game_count = 0
    starts = 1
    while game_count < games:
        game_count += 1
        
        deck = list(range(1, 53))
        random.shuffle(deck)
        # deck = [0] + deck[:(np.prod(game.GAME_DIMENSIONS) + game.REPLACE_CARDS)]
        drawn_card = deck.pop()
        for i, player in enumerate(players):
            player.mcts = GameState(np.zeros(np.prod(game.GAME_DIMENSIONS))[::], deck, drawn_card)

            storage = []
            turn = 1
            tau = 1 if training else 1e-2

            outcome = None
            while outcome is None:
                if turn == config.TURNS_UNTIL_TAU: tau = 1e-2
                storage.append({"state": player.mcts})

                action, pi_action, value = player.play_turn(storage, tau)

                outcome = game.check_game_over(player.mcts)

                if training:
                    for i, var in enumerate(["action", "pi_action", "value", "reward"]):
                        storage[-1][var] = [action, pi_action, value, 0.0][i]

                turn += 1
            starts *= -1

            # print(f"We are " + ("training" if training else "evaluating"))
            # print(f"Game outcome was: {outcome} ({int(outcome * 50)}), (Agent name: {player.get_name()[0]})")
            if not game_count % games:
                print(f"Amount of games played is now: {game_count} ({player.get_name()})\n")

            if not training:
                outcome = int(outcome * 50)
                outcomes[i].append(outcome)
                player.outcomes["length"] += 1
                player.outcomes["average"] = (player.outcomes["average"] * (player.outcomes["length"] - 1) + outcome) / player.outcomes["length"]
            else:
                storage.append({"state": player.mcts})
                storage[-1]["value"] = outcome

                for i, data in sorted(enumerate(storage[:-1]), reverse=True):
                    data["V_targ"] = V_targ(storage, i)
                    data["delta"] = delta(storage, i)

                for t, data in sorted(enumerate(storage[:-1]), reverse=True):
                    data["advantage"] = advantage(storage, t)

                    legal_moves = np.zeros(game.MOVE_AMOUNT)
                    legal_moves[game.get_legal_moves(data["state"])] = 1

                    game_states = game.generate_game_states(storage, t)

                    states = np.array(game.generate_nn_pass(game_states, True), dtype=object).tolist()
                    for flip in states: product.append(np.array([flip, data["action"], data["pi_action"], data["advantage"]] + list(legal_moves) + [data["V_targ"]], dtype=object))

                if not game_count % games:
                    length = files.add_to_file(files.get_path("positions.npy"), np.array(product[::-1], dtype=object), config.POSITION_AMOUNT)
                    product = []

                    print(f"Position length is now: {length}")

                left = config.POSITION_AMOUNT - length
                if not left:
                    if not os.path.exists(f"{config.SAVE_PATH}backup/positions_{config.POSITION_AMOUNT}.json"): files.make_backup("positions.npy", f"positions_{config.POSITION_AMOUNT}.npy")
                else:
                    if games == game_count: games += np.ceil(left / (game.GAME_LENGTH * 16 * left // (game.GAME_LENGTH * 16 * og_games)))

    if not training: return outcomes


# V_targ is how good a position is
def V_targ(data, t):
    return data[t]["reward"] + config.GAMMA * data[t + 1]["value"]


# delta is how much good a position is compared to the last one = V_targ - V(s_t)
def delta(data, t):
    return data[t]["V_targ"] - data[t]["value"]


# advantage is how much better a the future is (I think)
def advantage(data, t):
    li = [(config.GAMMA * config.LAMBDA) ** i * data[t + i]["delta"] for i in range(len(data) - t - 1)]
    return sum(li)


def self_play(agent):
    print("\nSelf-play started!\n")
    play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    # outcome = agent.outcomes["average"]
    # print(f"The average outcome from self-play was: {outcome}")


def retrain_network(agent):
    print("\nRetraining started!\n")

    positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

    for _ in range(config.TRAINING_ITERATIONS):
        minibatch = random.sample(positions, config.BATCH_SIZE[0])


        x = [[], [], []]
        y = {"value_head": [], "policy_head": []}

        for position in minibatch:
            for head in y:
                y[head].append([])

            for i, var in enumerate(position[1:(5 + game.MOVE_AMOUNT)]):
                y["policy_head" if i != 3 + game.MOVE_AMOUNT else "value_head"][-1].append(var)

            for i, dim in enumerate(position[0]):
                x[i].append(np.array(dim))

        for i, var in enumerate(x):
            x[i] = np.array(var)
        
        for head in y:
            y[head] = np.array(y[head])

        agent.nn.train(x, y)

    # data = [np.expand_dims(dat, 0) for dat in positions[-1][0]]
    # real = positions[-1]
    # (v, p) = agent.nn.model.predict(data)
    # a = np.exp(p)
    # probs = a / np.sum(a)

    agent.nn.iterations.append(config.TRAINING_ITERATIONS * config.EPOCHS)
    agent.nn.version += 1
    agent.nn.save_model()
    agent.nn.save_metrics("current_agent")
    agent.outcomes = {"average": 0, "length": 0}
    agent.nn.plot_metrics()


def evaluate_network(agents):
    print("\nEvaluation of agents started!\n")

    outcomes = play(agents, config.GAME_AMOUNT_EVALUATION)

    for agent in agents.values():
        agent.save_outcomes("current_agent")

    for i, player in enumerate(outcomes):
        outcomes[i] = np.sum(player) / len(player)

    log(agents, outcomes)
    agents[-1].nn.plot_outcomes()

    print(f"The results were: {outcomes}")
    if outcomes[1] > outcomes[0] * config.WINNING_THRESHOLD:
        agents[1].copy_profile(agents[-1])
        print(f"The best_agent has copied the current_agent's profile")

    return agents


def log(agents, results):
    names = [agents[1].get_name(), agents[-1].get_name()]

    best_name = names[np.argmax(results)]
    best = f"{best_name} is the best" if results[0] != results[1] else "They are equally good"

    message = f"""{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}:
------------ {names[0]} vs {names[1]} ------------
Results are: {results}
{best}!

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
