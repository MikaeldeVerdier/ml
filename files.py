import os
import json
import config
from shutil import copyfile

EMPTY_SAVE = json.dumps({
    "best_agent": {
        "version": 0,
        "version_outcomes": {
            
        },
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
    "current_agent": {
        "version": 0,
        "version_outcomes": {
            
        },
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
})
EMPTY_POSITIONS = json.dumps([])
EMPTY_LOG = ""

EMPTY_FILES = {"save.json": EMPTY_SAVE, "positions.json": EMPTY_POSITIONS, "log.txt": EMPTY_LOG}


def setup_files():
    save_folder = (config.SAVE_PATH, os.mkdir)
    backup_folder = (f"{config.SAVE_PATH}backup/", os.mkdir)
    save_file = (f"{config.SAVE_PATH}save.json", open, *"x")
    positions_file = (f"{config.SAVE_PATH}positions.json", open, *"x")
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
    write(file, EMPTY_FILES[file])


def reset_key(file, key):
    loaded = load_file(file)
    loaded[key] = EMPTY_FILES[file][key]
    write(file, json.loads(loaded))


def add_to_file(file, content, max_len):
    loaded = load_file(file)
    recent = len(loaded) != max_len
    loaded += content
    loaded = loaded[-max_len:]
    write(file, json.dumps(loaded))
    return len(loaded), recent


def make_backup(file, new_name=None):
    new_name = new_name if new_name else file
    copyfile(get_path(file), f"{config.SAVE_PATH}backup/{new_name}")
