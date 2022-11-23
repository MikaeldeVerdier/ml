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

    current_agent = CurrentAgent(CurrentNeuralNetwork, load[0])
    best_agent = BestAgent(BestNeuralNetwork, load[1])
    agents = {1: best_agent, -1: current_agent}

    return agents


def play(players, games, training=False):
    players = sorted(set(players.values()))

    if training:
        product = []
        length = 0
        is_full = False
    else:
        outcomes = [[], []]

    game_count = 0
    starts = 1
    while game_count < games:
        game_count += 1
        
        deck = list(range(1, 53))
        random.shuffle(deck)
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

                action, pi_action, y_a, value = player.play_turn(storage, tau)

                if training:
                    for i, var in enumerate(["action", "pi_action", "logit_a", "value", "reward"]):
                        storage[-1][var] = [action, pi_action, y_a, value, 0 if outcome is None else outcome][i]
                outcome = game.check_game_over(player.mcts)

                turn += 1

            starts *= -1

            print(f"We are " + ("training" if training else "evaluating"))
            print(f"Game outcome was: {outcome} ({int(outcome * 50)}), (Agent name: {player.get_name()[0]})")
            print(f"Amount of games played for this agent is now: {game_count}\n")

            if not training:
                outcomes[i].append(int(outcome * 50))
                player.outcomes["length"] += 1
                player.outcomes["average"] = (player.outcomes["average"] * (player.outcomes["length"] - 1) + int(outcome * 50)) / player.outcomes["length"]
            else:
                for i, data in sorted(enumerate(storage), reverse=True):
                    # data["reward"] = outcome
                    if i != len(storage) - 1:
                        data["delta"] = delta(storage, i)

                for i, data in enumerate(storage[:-1]):
                    data["advantage"] = advantage(storage, i)
                # storage[-1]["advantage"] = outcome

                # away_from_full = config.POSITION_AMOUNT - len(loaded)
                # if training_length > config.POSITION_AMOUNT / 25 or away_from_full and training_length >= away_from_full:
                
                index = len(product)

                for t, data in sorted(enumerate(storage), reverse=True):
                    legal_moves = np.zeros(game.MOVE_AMOUNT)
                    legal_moves[game.get_legal_moves(data["state"])] = 1

                    game_states = game.generate_game_states(storage, t)

                    if t != len(storage) - 1:
                        states = np.array(game.generate_nn_pass(game_states, True), dtype=object).tolist()
                        for flip in states: product.append(np.array([flip, data["action"].item(), data["pi_action"], data["advantage"]] + list(legal_moves) + [data["reward"], product[-1][0]], dtype=object))  # [s, a, pi_action, advantage, nn_value, nn_value_s+1, logits]
                    else:
                        product.append([np.array(game.generate_nn_pass(game_states, False), dtype=object).tolist()])
                
                del product[index]
                    # data[0] = np.array(game.generate_nn_pass(data[0])).tolist()
                    # data[1] = data[1].tolist()
                # training_data = np.vstack(training_data).tolist()

                # np.save(f"{config.SAVE_PATH}positions.npy", np.array(product[1:], dtype="object"))
                if not game_count % np.ceil(config.GAME_AMOUNT_SELF_PLAY / 5):
                    length = files.add_to_file(files.get_path("positions.npy"), np.array(product, dtype=object), config.POSITION_AMOUNT)
                    product = []
                # loaded += product[1:]
                # loaded = loaded[-config.POSITION_AMOUNT:]
                # files.write("positions.json", json.dumps(loaded))
                
                is_full = length == config.POSITION_AMOUNT

                if is_full:
                    if not os.path.exists(f"{config.SAVE_PATH}backup/positions_{config.POSITION_AMOUNT}.json"): files.make_backup("positions.npy", f"positions_{config.POSITION_AMOUNT}.npy")
                else:
                    if games == game_count: games += 1
 
                print(f"Position length is now: {length}")

    if not training: return outcomes


def delta(data, t):
    if "delta" in data[t].keys():
        return data[t]["delta"]
    delt = data[t]["reward"] + config.GAMMA * data[t + 1]["value"] - data[t]["value"]
    return delt


def advantage(data, t):
    T = len(data)
    li = [(config.GAMMA * config.LAMBDA) ** i * delta(data, t + i) for i in range(T - t - 1)]
    return sum(li)


def self_play(agent):
    play({1: agent, -1: agent}, config.GAME_AMOUNT_SELF_PLAY, True)

    # outcome = agent.outcomes["average"]
    # print(f"The average outcome from self-play was: {outcome}")


def retrain_network(agent):
    positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

    for _ in range(config.TRAINING_ITERATIONS):
        minibatch = random.sample(positions, config.BATCH_SIZE)

        x = [[], [], []]
        y = {"value_head": [], "policy_head": []}

        for position in minibatch:
            for head in y:
                y[head].append([])
            # y["value_head"].append([])
            # y["policy_head"].append([])
            for i, var in enumerate(position[1:(5 + game.MOVE_AMOUNT)]):
                y["policy_head" if i != 3 + game.MOVE_AMOUNT else "value_head"][-1].append(var)

            posses = [pos[-1] for pos in positions]
            y["value_head"][-1].append(posses.index(position[-1]))

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
    agent.nn.model.save(f"{config.SAVE_PATH}training/v.{agent.nn.version}")
    agent.nn.save_metrics("current_agent")
    agent.outcomes = {"average": 0, "length": 0}
    agent.nn.plot_metrics()


def evaluate_network(agents):
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
