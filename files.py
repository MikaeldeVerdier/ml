import os
import json
from shutil import copyfile
import config

EMPTY_SAVE = {
    "best_agent": 1,
    "agent_1": {
        "version": 1,
        "iterations": [

        ],
        "metrics": {
            "loss": [

            ],
            "value_head_loss": [

            ],
            "policy_head_loss": [

            ],
            "value_head_accuracy": [

            ],
            "policy_head_accuracy": [

            ],
            "val_loss": [

            ],
            "val_value_head_loss": [

            ],
            "val_policy_head_loss": [

            ],
            "val_value_head_accuracy": [

            ],
            "val_policy_head_accuracy": [

            ]
        }
    },
    "agent_2": {
        "version": 1,
        "iterations": [

        ],
        "metrics": {
            "loss": [

            ],
            "value_head_loss": [

            ],
            "policy_head_loss": [

            ],
            "value_head_accuracy": [

            ],
            "policy_head_accuracy": [

            ],
            "val_loss": [

            ],
            "val_value_head_loss": [

            ],
            "val_policy_head_loss": [

            ],
            "val_value_head_accuracy": [

            ],
            "val_policy_head_accuracy": [
                
            ]
        }
    }
}
EMPTY_POSITIONS = []
EMPTY_LOG = ""

EMPTY_FILES = {"save.json": EMPTY_SAVE, "positions.json": EMPTY_POSITIONS, "log.txt": EMPTY_LOG}

backup_path = f"{config.SAVE_PATH}backup/"

def setup_files():
    save_folder = (config.SAVE_PATH, os.mkdir)
    backup_folder = (f"{config.SAVE_PATH}backup/", os.mkdir)
    save_file = (f"{config.SAVE_PATH}save.json", open, *"x")
    positions_file = (f"{config.SAVE_PATH}positions.json", open, *"x")
    log_file = (f"{config.SAVE_PATH}log.txt", open, *"x")

    for file, func, *kwargs in [save_folder, backup_folder, save_file, positions_file, log_file]:
        if not os.path.exists(file): func(file, *kwargs)

    reset_file("save.json")

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
    write(file, json.dumps(EMPTY_FILES[file]))

def reset_key(file, key):
    loaded = load_file(file)
    loaded[key] = EMPTY_FILES[file][key]
    write(file, json.dumps(loaded))

def add_to_file(file, content, max_len):
    loaded = load_file(file)
    recent = len(loaded) != config.POSITION_AMOUNT
    loaded += content
    loaded = loaded[-max_len:]
    write(file, json.dumps(loaded))
    return len(loaded) == max_len, recent

def make_backup(file, new_name=None):
    new_name = new_name if new_name else file
    copyfile(get_path(file), f"{config.SAVE_PATH}backup/{new_name}")