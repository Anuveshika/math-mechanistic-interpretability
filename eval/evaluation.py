def evaluate_steps(pred_steps, true_steps):
    return sum(p == t for p, t in zip(pred_steps, true_steps)) / len(true_steps)
