import numpy as np
from sklearn.decomposition import PCA

def operator_subspace(activations, labels, operator, k=10):
    X = np.array([a for a, l in zip(activations, labels) if l == operator])
    pca = PCA(n_components=k)
    pca.fit(X)

    return {
        "components": pca.components_,
        "explained_variance": pca.explained_variance_ratio_
    }
