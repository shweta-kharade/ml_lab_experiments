"""
Program 3: Decision tree classifier - visualisation, pruning, depth control, overfitting on synthetic data
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ---------- A. Visualise a tree (Iris) ----------
iris = load_iris()
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(iris.data, iris.target, test_size=.3, random_state=1)
tree = DecisionTreeClassifier(max_depth=3, random_state=1).fit(Xi_tr, yi_tr)
print(f"Iris tree (max_depth=3) test accuracy: {tree.score(Xi_te, yi_te):.4f}")
plt.figure(figsize=(13, 7))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision tree on Iris (max_depth=3)"); plt.show()

# ---------- B. Synthetic noisy data to demonstrate overfitting ----------
X, y = make_classification(n_samples=600, n_features=10, n_informative=4, n_redundant=2,
                           flip_y=0.15, class_sep=1.0, random_state=7)   # flip_y adds label noise
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=7)

full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("\n=== Unrestricted tree ===")
print(f"depth={full.get_depth()} leaves={full.get_n_leaves()} train acc={full.score(X_tr, y_tr):.3f} test acc={full.score(X_te, y_te):.3f}  <-- overfit")

# ---------- C. Depth control ----------
depths = range(1, 21)
tr_acc, te_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    tr_acc.append(m.score(X_tr, y_tr)); te_acc.append(m.score(X_te, y_te))
best_d = list(depths)[int(np.argmax(te_acc))]
print(f"\nBest max_depth by test accuracy = {best_d} (test acc={max(te_acc):.3f})")

# ---------- D. Cost-complexity (post) pruning ----------
path = full.cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]
p_tr, p_te, leaves = [], [], []
for a in alphas:
    m = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_tr, y_tr)
    p_tr.append(m.score(X_tr, y_tr)); p_te.append(m.score(X_te, y_te)); leaves.append(m.get_n_leaves())
best_a = alphas[int(np.argmax(p_te))]
pruned = DecisionTreeClassifier(ccp_alpha=best_a, random_state=0).fit(X_tr, y_tr)
print(f"Best ccp_alpha={best_a:.5f}: leaves={pruned.get_n_leaves()} train acc={pruned.score(X_tr, y_tr):.3f} test acc={pruned.score(X_te, y_te):.3f}")

print("\n=== Summary ===")
print(f"{'Model':28s}{'Train':>8s}{'Test':>8s}")
print(f"{'Unrestricted':28s}{full.score(X_tr, y_tr):8.3f}{full.score(X_te, y_te):8.3f}")
dm = DecisionTreeClassifier(max_depth=best_d, random_state=0).fit(X_tr, y_tr)
print(f"{'Depth-controlled (d=' + str(best_d) + ')':28s}{dm.score(X_tr, y_tr):8.3f}{dm.score(X_te, y_te):8.3f}")
print(f"{'Pruned (ccp_alpha)':28s}{pruned.score(X_tr, y_tr):8.3f}{pruned.score(X_te, y_te):8.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(depths, tr_acc, "o-", label="train"); ax[0].plot(depths, te_acc, "s-", label="test")
ax[0].axvline(best_d, ls="--", c="gray"); ax[0].set(title="Depth control: overfitting as depth grows", xlabel="max_depth", ylabel="Accuracy"); ax[0].legend()
ax[1].plot(alphas, p_tr, "o-", label="train", ms=3); ax[1].plot(alphas, p_te, "s-", label="test", ms=3)
ax[1].set_xscale("log"); ax[1].set(title="Cost-complexity pruning", xlabel="ccp_alpha (log)", ylabel="Accuracy"); ax[1].legend()
plt.tight_layout(); plt.show()

plt.figure(figsize=(14, 6)); plot_tree(pruned, filled=True, max_depth=3, fontsize=7)
plt.title("Pruned tree (top 3 levels shown)"); plt.show()
