import torch

class ActivationStore:
    def __init__(self):
        self.activations = []

    def hook_fn(self, act, hook):
        self.activations.append(act.detach().cpu())

    def clear(self):
        self.activations = []
