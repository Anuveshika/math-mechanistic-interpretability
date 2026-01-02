import json
import random

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_gsm8k(path="data/gsm8k_filter.json"):
    return load_json(path)

def load_synthetic(path="data/math.json"):
    return load_json(path)

def load_dataset(name):
    if name == "gsm8k":
        return load_gsm8k()
    elif name == "synthetic":
        return load_synthetic()
    else:
        raise ValueError(f"Unknown dataset {name}")
