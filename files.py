import numpy as np
import os
import json
import config
from shutil import copyfile

EMPTY_SAVE = json.dumps({
    "main_nn_version": 0,
    "target_nn_version": 0,
    "version_outcomes": {},
    "iterations": [],
    "metrics": {
        "loss": [],
        "val_loss": [],
    }
})
EMPTY_POSITIONS = np.array([])
EMPTY_LOG = ""

EMPTY_FILES = {"save.json": EMPTY_SAVE, "positions.npy": EMPTY_POSITIONS, "log.txt": EMPTY_LOG}


def setup_files():
    save_folder = (config.SAVE_PATH, os.mkdir)
    backup_folder = (f"{config.SAVE_PATH}backup/", os.mkdir)
    save_file = (f"{config.SAVE_PATH}save.json", open, *"x")
    positions_file = (f"{config.SAVE_PATH}positions.npy", open, *"x")
    log_file = (f"{config.SAVE_PATH}log.txt", open, *"x")

    for file, func, *kwargs in [save_folder, backup_folder, save_file, positions_file, log_file]:
        if not os.path.exists(file): func(file, *kwargs)


def get_path(file):
    return f"{config.SAVE_PATH}{file}"


def read(file):
    file = get_path(file)
    with open(file, "r") as f: return f.read()


def write(file, content, mode="w"):
    file = get_path(file)
    with open(file, mode) as f: f.write(content)


def load_file(file):
    return json.loads(read(file))


def reset_file(file):
    make_backup(file)
    if not file.endswith(".npy"): write(file, EMPTY_FILES[file])
    else:
        file_path = get_path(file)
        np.save(file_path, EMPTY_FILES[file])


def edit_key(file, keys, values):
    loaded = load_file(file)
    for key, value in zip(keys, values):
        loaded[key] = value
    write(file, json.dumps(loaded))


def reset_key(file, key):
    loaded = load_file(file)
    loaded[key] = EMPTY_FILES[file][key]
    write(file, json.dumps(loaded))


def add_to_file(file, content, max_len):
    if not file.endswith(".npy"):
        loaded = load_file(file)
        recent = len(loaded) != max_len
        loaded += content
        loaded = loaded[-max_len:]
        write(file, json.dumps(loaded))
        return len(loaded), recent
    else:
        loaded = np.load(file, allow_pickle=True)
        if len(loaded): loaded = np.append(loaded, content, axis=0)[-max_len:]
        else: loaded = content[-max_len:]
        np.save(file, loaded)
        return len(loaded)


def copy_file(file, new_path):
    copyfile(get_path(file), new_path)


def make_backup(file, new_name=None):
    new_name = new_name if new_name else file
    copyfile(get_path(file), f"{config.SAVE_PATH}backup/{new_name}")
