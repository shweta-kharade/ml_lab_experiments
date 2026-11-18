"""
Program 9: Linear and kernel (RBF, polynomial) SVM
Visualise margins & support vectors; tune C and gamma
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

def plot_svm(ax, clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300), np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0, alpha=.15, cmap="coolwarm")
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])     # margins & boundary
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.scatter(*clf.support_vectors_.T, s=140, facecolors="none", edgecolors="g", linewidths=2, label=f"SVs ({len(clf.support_vectors_)})")
    ax.set_title(title); ax.legend(loc="lower right")

# ---------- Linear SVM on separable blobs ----------
Xb, yb = make_blobs(n_samples=100, centers=2, cluster_std=1.3, random_state=6)
lin = SVC(kernel="linear", C=1).fit(Xb, yb)
w, b = lin.coef_[0], lin.intercept_[0]
print(f"Linear SVM: w={w.round(3)}, b={b:.3f}, margin width = 2/||w|| = {2 / np.linalg.norm(w):.3f}, #SV={len(lin.support_vectors_)}")

# ---------- Kernels on non-linear data ----------
Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(Xm, ym, test_size=.3, random_state=0)
models = {"Linear": SVC(kernel="linear", C=1), "Polynomial (deg=3)": SVC(kernel="poly", degree=3, C=1, coef0=1),
          "RBF": SVC(kernel="rbf", C=1, gamma="scale")}
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
plot_svm(ax[0], lin, Xb, yb, "Linear SVM (C=1) - margins & SVs")
print("\nKernel comparison (moons):")
for a, (name, m) in zip(ax[1:], models.items()):
    m.fit(X_tr, y_tr); print(f"  {name:20s} train={m.score(X_tr, y_tr):.3f} test={m.score(X_te, y_te):.3f}")
    plot_svm(a, m, Xm, ym, f"{name} (test acc={m.score(X_te, y_te):.2f})")
plt.tight_layout(); plt.show()

# ---------- Effect of C and gamma ----------
fig, ax = plt.subplots(3, 3, figsize=(14, 12))
for i, C in enumerate([0.1, 1, 100]):
    for j, g in enumerate([0.1, 1, 50]):
        m = SVC(kernel="rbf", C=C, gamma=g).fit(X_tr, y_tr)
        plot_svm(ax[i, j], m, Xm, ym, f"C={C}, gamma={g}\ntrain={m.score(X_tr, y_tr):.2f} test={m.score(X_te, y_te):.2f}")
plt.suptitle("RBF SVM: small C/gamma -> underfit, large C/gamma -> overfit"); plt.tight_layout(); plt.show()

# ---------- Grid search ----------
grid = GridSearchCV(SVC(kernel="rbf"), {"C": [0.01, 0.1, 1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5).fit(X_tr, y_tr)
print(f"\nBest params: {grid.best_params_}  CV acc={grid.best_score_:.4f}  test acc={grid.score(X_te, y_te):.4f}")
S = grid.cv_results_["mean_test_score"].reshape(6, 6)
plt.figure(figsize=(7, 5.5)); plt.imshow(S, cmap="viridis")
plt.xticks(range(6), grid.param_grid["gamma"]); plt.yticks(range(6), grid.param_grid["C"])
for (r, c), v in np.ndenumerate(S): plt.text(c, r, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
plt.xlabel("gamma"); plt.ylabel("C"); plt.title("Grid search CV accuracy"); plt.colorbar(); plt.show()
