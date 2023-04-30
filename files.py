import numpy as np
import os
import json
from shutil import copyfile, copytree, rmtree

import config

EMPTY_SAVE = json.dumps(
	{
		"main_nn_version": 0,
		"target_nn_version": 0,
		"metrics": {
			"loss": [],
			"val_loss": [],
			"outcomes": {},
			"average_q_value": []
		}
	}
)
EMPTY_POSITIONS = np.array([])
EMPTY_LOG = ""

EMPTY_FILES = {
	"save.json": EMPTY_SAVE,
	"positions.npy": EMPTY_POSITIONS,
	"log.txt": EMPTY_LOG
}

def get_path(file):
	return f"{config.SAVE_PATH}{file}"


def setup_files():
	save_folder = (get_path(""), os.mkdir)
	backup_folder = (get_path(f"backup/"), os.mkdir)
	save_file = (get_path("save.json"), open, "x")
	positions_file = (get_path("positions.npy"), open, "x")
	log_file = (get_path("log.txt"), open, "x")

	for file, func, *args in [save_folder, backup_folder, save_file, positions_file, log_file]:
		if not os.path.exists(file):
			func(file, *args)


def reset_files():
	reset_file("save.json")
	reset_file("positions.npy")
	reset_file("log.txt")


def read(file):
	file = get_path(file)
	with open(file, "r") as f:
		return f.read()


def write(file, content, mode="w"):
	file = get_path(file)
	if not file.endswith(".npy"):
		with open(file, mode) as f:
			f.write(content)
	else:
		np.save(file, content)


def load_file(file):
	return json.loads(read(file))


def copy_file(file, new_path):
	copyfile(get_path(file), get_path(new_path))


def make_backup(file, new_name=None):
	new_name = new_name if new_name else file
	copy_file(file, f"backup/{new_name}")


def reset_file(file):
	make_backup(file)
	if not file.endswith(".npy"):
		write(file, EMPTY_FILES[file])
	else:
		file_path = get_path(file)
		np.save(file_path, EMPTY_FILES[file])


def edit_keys(file, keys, values):
	loaded = load_file(file)
	for key, value in zip(keys, values):
		loaded[key] = value
	write(file, json.dumps(loaded))


def copy_dir(old_dir, new_dir):
	old_path = get_path(old_dir)
	new_path = get_path(new_dir)

	if os.path.exists(new_path):
		rmtree(new_path)
	copytree(old_path, new_path)


def find_dir(file, path=""):
	for root, dirs, _ in os.walk(get_path(path)):
		if file in dirs:
			return os.path.join(root, file)


"""def reset_key(file, key):
	loaded = load_file(file)
	loaded[key] = EMPTY_FILES[file][key]
	write(file, json.dumps(loaded))"""


"""def copy_dir(old_dir, new_dir):
	old_path = get_path(old_dir)
	new_path = get_path(new_dir)

	if os.path.exists(new_path):
		rmtree(new_path)
	copytree(old_path, new_path)"""
