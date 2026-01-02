from transformer_lens import HookedTransformer

def load_model(model_name="gpt2-small", device="cpu"):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model
