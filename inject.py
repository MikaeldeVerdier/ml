import json
import config

with open(f"{config.SAVE_FOLDER}positions.json", "r") as positions_r:
    with open(f"poss.json", "r") as poss:
        positions = json.loads(poss.read())
        loaded = json.loads(positions_r.read())
        loaded += positions
        loaded = loaded[-config.POSITION_AMOUNT:]
        print(f"Positions length is now {len(loaded)}\n")
        with open(f"{config.SAVE_FOLDER}positions.json", "w") as positions_w: positions_w.write(json.dumps(loaded))
