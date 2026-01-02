from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def run_linear_probe(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    return {
        "train_acc": clf.score(X_train, y_train),
        "test_acc": clf.score(X_test, y_test),
        "coef_norm": (clf.coef_ ** 2).sum() ** 0.5
    }
