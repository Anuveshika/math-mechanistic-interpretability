import torch
from models.load_models import load_model
from activations.hooks import ActivationStore

OPERATOR_MAP = {
    "add": 0,
    "subtract": 1,
    "multiply": 2
}

def collect_activations(
    dataset,
    layer=9,
    device="cpu"
):
    model = load_model(device=device)
    store = ActivationStore()

    hook_name = f"blocks.{layer}.mlp.hook_post"
    activations = []
    labels = []

    for ex in dataset:
        # 🔒 Skip examples without a valid operator
        if "operator" not in ex or ex["operator"] not in OPERATOR_MAP:
            continue

        store.clear()
        tokens = model.to_tokens(ex["prompt"], prepend_bos=True)

        model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, store.hook_fn)]
        )

        # [seq_len, hidden_dim]
        acts = store.activations[0][0]

        labeled_tokens = ex.get("labeled_tokens")
        if not labeled_tokens:
            labeled_tokens = list(range(1, tokens.shape[1]))

        for token_idx in labeled_tokens:
            activations.append(acts[token_idx].cpu().numpy())
            labels.append(OPERATOR_MAP[ex["operator"]])

        if len(activations) == 0:
          raise RuntimeError(
        "No activations collected. "
        "Check that your dataset contains valid operator labels "
        "and that OPERATOR_MAP matches the data."
    )


    return activations, labels
