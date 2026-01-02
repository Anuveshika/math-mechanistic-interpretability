from data.datasets import load_dataset
from activations.collect_activations import collect_activations
from analysis.probes import run_linear_probe

def main():
    dataset = load_dataset("synthetic")

    activations, labels = collect_activations(
        dataset,
        layer=9,
        device="cpu"
    )

    results = run_linear_probe(activations, labels)
    print("Probe results:", results)

if __name__ == "__main__":
    main()
