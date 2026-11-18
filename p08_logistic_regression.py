"""
Program 8: Logistic regression (binary) - sigmoid, decision boundary, ROC & AUC
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

sigmoid = lambda z: 1 / (1 + np.exp(-z))

# ---------- Data ----------
X, y = make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=1.2, random_state=4)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=4, stratify=y)

# ---------- Model ----------
clf = LogisticRegression().fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
w, b = clf.coef_[0], clf.intercept_[0]
print(f"Weights={w.round(3)}, bias={b:.3f}")
print(f"Test accuracy = {accuracy_score(y_te, pred):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_te, pred))
fpr, tpr, thr = roc_curve(y_te, proba)
auc = roc_auc_score(y_te, proba)
print(f"AUC = {auc:.4f}")
# Optimal threshold (Youden's J)
j = np.argmax(tpr - fpr); print(f"Best threshold (Youden J) = {thr[j]:.3f}  (TPR={tpr[j]:.3f}, FPR={fpr[j]:.3f})")

# ---------- Plots ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
# 1. sigmoid
z = np.linspace(-8, 8, 200)
ax[0].plot(z, sigmoid(z), lw=2); ax[0].axhline(.5, ls="--", c="gray"); ax[0].axvline(0, ls="--", c="gray")
zt = X_te @ w + b; ax[0].scatter(zt, sigmoid(zt), c=y_te, cmap="coolwarm", edgecolor="k", s=25, zorder=3)
ax[0].set(title="Sigmoid  σ(z)=1/(1+e^-z)", xlabel="z = w·x + b", ylabel="P(y=1)")
# 2. decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
P = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
cs = ax[1].contourf(xx, yy, P, levels=20, cmap="coolwarm", alpha=.6); fig.colorbar(cs, ax=ax[1], label="P(y=1)")
ax[1].contour(xx, yy, P, levels=[.5], colors="k", linewidths=2)
ax[1].scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolor="k"); ax[1].set(title="Decision boundary (P=0.5)", xlabel="x1", ylabel="x2")
# 3. ROC
ax[2].plot(fpr, tpr, lw=2, label=f"Logistic (AUC={auc:.3f})"); ax[2].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
ax[2].scatter(fpr[j], tpr[j], c="r", zorder=3, label="Best threshold")
ax[2].set(title="ROC curve", xlabel="False Positive Rate", ylabel="True Positive Rate"); ax[2].legend()
plt.tight_layout(); plt.show()
