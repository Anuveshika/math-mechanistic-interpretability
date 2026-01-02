import numpy as np

def inject_direction(h, v, alpha=1.0):
    v = v / np.linalg.norm(v)
    return h + alpha * v
