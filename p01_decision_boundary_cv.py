"""
Program 1: Toy classification using decision boundaries (manual coding)
- Manual train/test split, manual k-fold cross-validation
- Nearest-centroid classifier (decision boundary = perpendicular bisector between class centroids)
- Manual accuracy, precision, recall, F1 (macro), checked against scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

rng = np.random.default_rng(42)

# ---------- Data (Iris, petal length & petal width -> 2D so the boundary can be plotted) ----------
iris = load_iris()
X, y = iris.data[:, 2:4], iris.target
class_names = iris.target_names

# ---------- Manual classifier ----------
class NearestCentroid:
    """Assign a point to the class whose centroid is closest.
    The decision boundary between two classes is the perpendicular bisector of their centroids."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        d = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(d, axis=1)]

# ---------- Manual helpers ----------
def train_test_split_manual(X, y, test_size=0.3):
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def kfold_indices(n, k=5):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test

def metrics_manual(y_true, y_pred, classes):
    acc = np.mean(y_true == y_pred)
    P, R, F = [], [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        P.append(p); R.append(r); F.append(f)
    return acc, np.mean(P), np.mean(R), np.mean(F)

# ---------- 1. Train/test split ----------
X_tr, X_te, y_tr, y_te = train_test_split_manual(X, y, 0.3)
model = NearestCentroid().fit(X_tr, y_tr)
pred = model.predict(X_te)
acc, p, r, f1 = metrics_manual(y_te, pred, model.classes_)
print("=== Hold-out (70/30) ===")
print(f"Manual    : Acc={acc:.4f}  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
sp, sr, sf, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
print(f"sklearn   : Acc={accuracy_score(y_te, pred):.4f}  Precision={sp:.4f}  Recall={sr:.4f}  F1={sf:.4f}")

# ---------- 2. k-fold cross-validation ----------
k = 5
rows = []
for fold, (tr, te) in enumerate(kfold_indices(len(X), k), 1):
    m = NearestCentroid().fit(X[tr], y[tr])
    rows.append(metrics_manual(y[te], m.predict(X[te]), m.classes_))
    print(f"Fold {fold}: Acc={rows[-1][0]:.4f}  P={rows[-1][1]:.4f}  R={rows[-1][2]:.4f}  F1={rows[-1][3]:.4f}")
rows = np.array(rows)
print(f"\n=== {k}-fold CV (mean ± std) ===")
for name, col in zip(["Accuracy", "Precision", "Recall", "F1-score"], rows.T):
    print(f"{name:10s}: {col.mean():.4f} ± {col.std():.4f}")

# ---------- 3. Plot decision regions ----------
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                     np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
sc = plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="viridis", edgecolor="k", label="train")
plt.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="viridis", edgecolor="k", marker="s", s=70, label="test")
plt.scatter(*model.centroids_.T, c="red", marker="X", s=200, label="centroids")
plt.xlabel("Petal length (cm)"); plt.ylabel("Petal width (cm)")
plt.title("Nearest-centroid decision boundaries (Iris)"); plt.legend()
plt.tight_layout(); plt.show()
