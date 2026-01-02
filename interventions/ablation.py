import numpy as np

def ablate_direction(h, v):
    v = v / np.linalg.norm(v)
    return h - np.dot(h, v) * v
