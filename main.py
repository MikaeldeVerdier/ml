import numpy as np
import random
from datetime import datetime

import environment
import config
import files
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
		replay = np.load(files.get_path("positions.npy"), allow_pickle=True).tolist()

	results = np.empty(np.array(env.players).shape + (0,)).tolist()

	og_games = games
	game_count = 0
	while game_count < games:
		game_count += 1

		env.reset()

		q_values = np.empty(np.array(env.current_players).shape + (0,)).tolist()
		storage = []

		while not env.game_state.done:
			storage.append({"state": env.game_state})

			probs, action = env.player.get_action(env.game_state, env.epsilon)

			if env.player.trainable:
				q_values[env.game_state.turn].append(probs[action])

			env.step(probs, action)

			if training:
				for key, var in [("action", action), ("reward", env.game_state.scores[env.game_state.turn])]:
					storage[-1][key] = var

		for (i, player), score in zip(enumerate(env.current_players), env.game_state.scores):
			results[env.players_turn][i].append(environment.INVERSE_REWARD_TRANSFORM(score))

			if player.trainable:
				player.main_nn.metrics["average_q_value"].append(float(np.mean(q_values[i])))

		if not game_count % games:
			print(f"Amount of games played is now: {game_count} ({env.player.get_name()})\n")

		if training:
			for t, data in enumerate(storage):
				data["target"] = env.player.calculate_target(storage, t) if t != len(storage) - 1 else data["reward"]

				states = np.array(data["state"].generate_nn_pass(modify=True), dtype=object).tolist()
				replay = (replay + [np.array([state, data["action"], data["target"]], dtype=object) for state in states])[-config.POSITION_AMOUNT:]

			if not game_count % games:
				files.write("positions.npy", np.array(replay, dtype=object))

				print(f"Position length is now: {len(replay)}")

			left = config.POSITION_AMOUNT - len(replay)
			if left and games == game_count:
				games += og_games

	for i, players in enumerate(env.players):
		for i2, player in enumerate(players):
			results[i][i2] = np.mean(results[i][i2], axis=-1) if environment.REWARD_AVERAGE else len(np.nonzero(results[i][i2]))
			
			if not training and player.trainable:
				player.main_nn.metrics["outcomes"][f"{player.main_nn.version}"] = results[i][i2]

	return results


def self_play(agent):
	print("\nSelf-play started!\n")
	outcomes = play(Environment([[agent]]), config.GAME_AMOUNT_SELF_PLAY, training=True)

	print(f"The result were: {outcomes}")


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

		x = [np.array(var) for var in x]
		y = np.array(y)

		agent.main_nn.train(x, y)
	agent.change_version()

	# data = [np.expand_dims(dat, 0) for dat in positions[-1][0]]
	# real = positions[-1]
	# p = agent.main_nn.model.predict(data)


def evaluate_network(agent):
	print("\nEvaluation of agent started!\n")

	outcomes = play(Environment([[agent]], epsilons=[[0.05]]), config.GAME_AMOUNT_EVALUATION)

	# log([agent], outcome)

	print(f"The result were: {outcomes}")


def compete(agents, games, starts, verbose=False):
	outcomes = play(Environment(np.array(agents).reshape(-1, 1), epsilons=[[0.05]] * len(agents), starts=starts, verbose=verbose), games)

	log(agents, outcomes)

	print(f"The results between agents named {' and '.join([agent.get_name() for agent in agents])} were: {outcomes}")
	best = agents[np.argmax(outcomes)].get_name()
	print(f"The best version was: version {best}")


def log(agent_s, average_s):
	message = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {' vs '.join([agent.get_name() for agent in agent_s])}:\nAverage results were {average_s}\n"
	files.write("log.txt", message, "a")


def play_test(load, games, starts=0):
	you = User()
	agents = [Agent(load=load, name=load), you]
	compete(agents, games, starts, verbose=True)


def play_versions(loads, games, starts=0):
	agents = [Agent(load=load, name=load) for load in loads]
	compete(agents, games, starts)


def main():
	# play_versions([None, "trained_version"], config.GAME_AMOUNT_PLAY_VERSIONS)
	# play_test(None, config.GAME_AMOUNT_PLAY_TEST)

	agent = initiate()

	while agent.main_nn.version <= config.LOOP_ITERATIONS:
		if not agent.main_nn.version % config.EVALUATION_FREQUENCY:
			evaluate_network(agent)
		self_play(agent)
		retrain_network(agent)


if __name__ == "__main__":
	main()
