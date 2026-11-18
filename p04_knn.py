"""
Program 4: k-Nearest Neighbours - implementation, visualisation, tuning k, curse of dimensionality
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class KNNScratch:
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X, self.y = X, y; return self
    def predict(self, X):
        out = []
        for x in X:
            d = np.linalg.norm(self.X - x, axis=1)
            nn = self.y[np.argsort(d)[:self.k]]
            out.append(Counter(nn).most_common(1)[0][0])
        return np.array(out)

iris = load_iris()
X, y = iris.data[:, 2:4], iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)

# ---- scratch vs sklearn ----
sc = KNNScratch(5).fit(X_tr, y_tr).predict(X_te)
sk = KNeighborsClassifier(5).fit(X_tr, y_tr).predict(X_te)
print(f"Scratch kNN acc={np.mean(sc == y_te):.4f} | sklearn kNN acc={np.mean(sk == y_te):.4f} | predictions identical: {np.array_equal(sc, sk)}")

# ---- decision regions for different k ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 250), np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 250))
for ax, k in zip(axes, [1, 5, 25]):
    m = KNeighborsClassifier(k).fit(X_tr, y_tr)
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.3, cmap="viridis"); ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="viridis", edgecolor="k")
    ax.set(title=f"k={k}  (test acc={m.score(X_te, y_te):.3f})", xlabel="Petal length", ylabel="Petal width")
plt.tight_layout(); plt.show()

# ---- tune k with cross-validation ----
Xf = iris.data
ks = range(1, 31)
cv = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), Xf, iris.target, cv=5).mean() for k in ks]
best_k = list(ks)[int(np.argmax(cv))]
print(f"\nBest k (5-fold CV, all 4 features) = {best_k}, CV accuracy = {max(cv):.4f}")
plt.figure(figsize=(7, 4)); plt.plot(ks, cv, "o-"); plt.axvline(best_k, ls="--", c="r")
plt.xlabel("k"); plt.ylabel("CV accuracy"); plt.title("Tuning k"); plt.grid(alpha=.3); plt.show()

# ---- curse of dimensionality ----
rng = np.random.default_rng(0)
n = 300
base = iris.data[:, 2:4]; yb = iris.target
dims = [0, 2, 5, 10, 20, 50, 100, 200, 500]
accs = []
for extra in dims:
    Xn = np.c_[base, rng.normal(size=(len(base), extra))] if extra else base      # add irrelevant noise features
    accs.append(cross_val_score(KNeighborsClassifier(5), Xn, yb, cv=5).mean())
print("\nAccuracy vs number of irrelevant noise features:")
for d, a in zip(dims, accs): print(f"  +{d:4d} noise dims -> acc={a:.3f}")

# distance concentration: (max-min)/min distance between random points shrinks as d grows
D = [1, 2, 5, 10, 50, 100, 500, 1000]; ratio = []
for d in D:
    P = rng.random((500, d)); q = rng.random(d)
    dist = np.linalg.norm(P - q, axis=1); ratio.append((dist.max() - dist.min()) / dist.min())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(dims, accs, "o-"); ax[0].set(title="kNN accuracy drops as noise dimensions grow", xlabel="# irrelevant features added", ylabel="CV accuracy"); ax[0].grid(alpha=.3)
ax[1].semilogx(D, ratio, "s-r"); ax[1].set(title="Distance concentration", xlabel="Dimension d", ylabel="(d_max - d_min)/d_min"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
