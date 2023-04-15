import numpy as np
import random
from datetime import datetime

import environment
import config
import files
from player import User, Agent
from environment import Environment

def initiate(load):
	files.setup_files()
	if not load:
		files.reset_file("save.json")
		files.reset_file("positions.npy")
		files.reset_file("log.txt")

	agent = Agent(load=load, is_trainable=True)

	return agent


def play(env, games, training=False):
	if training:
		buffer = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()
		length_generated = 0

	results = np.empty(np.shape(env.players) + (0,)).tolist()

	og_games = games
	game_count = 0
	while game_count < games:
		env.reset()

		if env.players_turn == 0:
			game_count += 1

		if env.player.is_trainable:
			q_values = np.empty(np.shape(env.current_players) + (0,)).tolist()

		storage = []

		while not env.game_state.done:
			storage.append({"state": env.game_state})

			probs, action = env.player.get_action(env.game_state, env.epsilon)

			if env.player.is_trainable and probs is not None:
				q_values[env.game_state.turn].append(probs[action])
			
			last_turn = env.game_state.turn

			env.step(probs, action)

			if training:
				for key, var in [("action", action), ("reward", env.game_state.reward), ("next_state", env.game_state)]:
					storage[-1][key] = var

		transformed_result = environment.inverse_reward_transform(env.game_state.reward)
		results[env.players_turn][last_turn].append(transformed_result)

		for i, player in enumerate(env.current_players):
			if player.is_trainable and len(q_values[i]):
				player.main_nn.metrics["average_q_value"].append(float(np.mean(q_values[i])))

		if not game_count % games:
			print(f"Amount of games played is now: {game_count} ({env.players_names})")

		if training:
			for data in storage:
				data["target"] = env.player.target_nn.calculate_target(data)

				generated_states = data["state"].generate_nn_pass(modify=True)
				buffer = (buffer + [[state, data["action"], data["target"]] for state in generated_states])[-config.BUFFER_SIZE:]
				length_generated += len(generated_states)

			if not game_count % games:
				files.write("positions.npy", np.array(buffer, dtype=object))

				print(f"New positions generated is now: {length_generated}\n")

			if (length_generated < config.BUFFER_REQUIREMENT or len(buffer) != config.BUFFER_SIZE) and games == game_count:
				games += og_games

	for i, players in enumerate(env.players):
		for i2, player in enumerate(players):
			results[i][i2] = environment.results_transform(results[i][i2])

			if not training and player.is_trainable:
				player.main_nn.metrics["outcomes"][f"{player.main_nn.version}"] = results[i][i2]

	return results


def self_play(agent):
	print("\nSelf-play started!\n")

	env = Environment([[agent]])
	outcomes = play(env, config.GAME_AMOUNT_SELF_PLAY, training=True)

	print(f"The results were: {outcomes}")


def retrain_network(agent):
	print("\nRetraining started!\n")

	positions = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

	for _ in range(config.TRAINING_ITERATIONS):
		minibatch = random.sample(positions, config.BATCH_SIZE[0])

		x = []
		y = []

		for position in minibatch:
			y.append(position[1:])
			x.append([np.array(dim) for dim in position[0]])

		x = [np.array(var) for var in zip(*x)]
		y = np.array(y)

		agent.main_nn.train(x, y)
	agent.change_version()

	# data = [np.expand_dims(dat, 0) for dat in positions[-1][0]]
	# real = positions[-1]
	# p = agent.main_nn.model.predict(data)


def evaluate_network(agent):
	print("\nEvaluation of agent started!\n")

	env = Environment([[agent]], epsilons=[[0.05]])
	outcomes = play(env, config.GAME_AMOUNT_EVALUATION)

	# log([agent], outcome)

	print(f"\nThe results were: {outcomes}")


def log(names, average_s, games):
	message = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {' vs '.join(names)} ({games} games):\nAverage results were {average_s}\n"
	files.write("log.txt", message, "a")


def compete(agents, epsilons, games, starts, verbose=False):
	env = Environment(agents.tolist(), epsilons=epsilons.tolist(), starts=starts, verbose=verbose)
	outcomes = play(env, games)

	for name, outcome in zip(env.players_names, outcomes):
		log(name, outcome, games)

		print(f"\nThe results between agents named {' and '.join(name)} were: {outcome}")
		best = name[np.argmax(outcome)]
		print(f"The best agent was: {best}")


def load_opponents(loads):
	for load in loads.flatten():
		is_not_random = load is not None

		agent = Agent(load=load, name=load or "Random agent", uses_nn=is_not_random)
		epsilon = 0.05 if is_not_random else 1
		yield (agent, epsilon)


def play_test(loads, games, starts=0, verbose=True):
	you = User(name="You")

	loads = np.array(loads, dtype=object)
	idx = loads != "You"

	opposing_agents, opposing_epsilons = zip(*load_opponents(loads[idx]))

	agents = np.full_like(loads, None, dtype=object)
	epsilons = np.full_like(loads, None, dtype=object)

	agents[idx] = opposing_agents
	epsilons[idx] = opposing_epsilons

	agents[~idx] = you

	compete(agents, epsilons, games, starts, verbose=verbose)


def play_versions(loads, games, starts=0, verbose=False):
	loads = np.array(loads, dtype=object)

	agents, epsilons = zip(*load_opponents(loads))

	agents = np.reshape(agents, loads.shape)
	epsilons = np.reshape(epsilons, loads.shape)

	compete(agents, epsilons, games, starts, verbose=verbose)


def main():
	# play_versions([[None], ["main_nn"]], config.GAME_AMOUNT_PLAY_VERSIONS)
	# play_test([[None], ["You"]], config.GAME_AMOUNT_PLAY_TEST)

	load = False
	agent = initiate(load)

	while agent.main_nn.version <= config.VERSION_AMOUNT:
		if not agent.main_nn.version % config.EVALUATION_FREQUENCY:
			evaluate_network(agent)
		self_play(agent)
		retrain_network(agent)


if __name__ == "__main__":
	main()
