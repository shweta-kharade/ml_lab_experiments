"""
Program 5: PCA and t-SNE for dimensionality reduction (Digits, 64-D)
Visualise in 2D and measure accuracy before/after feature reduction.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X, y = digits.data, digits.target
Xs = StandardScaler().fit_transform(X)
print("Original shape:", X.shape)

# ---------- PCA ----------
pca2 = PCA(n_components=2).fit(Xs)
X_pca2 = pca2.transform(Xs)
print(f"Variance explained by first 2 PCs: {pca2.explained_variance_ratio_.sum():.3f}")
pca_full = PCA().fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95)) + 1
print(f"Components needed for 95% variance: {n95}")

# ---------- t-SNE ----------
X_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Xs)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(cum, lw=2); ax[0].axhline(.95, c="r", ls="--"); ax[0].axvline(n95, c="g", ls="--")
ax[0].set(title="PCA cumulative explained variance", xlabel="# components", ylabel="Cumulative variance")
for a, E, t in zip(ax[1:], [X_pca2, X_tsne], ["PCA (2D)", "t-SNE (2D)"]):
    s = a.scatter(E[:, 0], E[:, 1], c=y, cmap="tab10", s=10); a.set_title(t)
fig.colorbar(s, ax=ax[2], label="digit")
plt.tight_layout(); plt.show()

# ---------- Accuracy before / after reduction ----------
# (PCA is fitted on the training split only to avoid data leakage)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr); X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

def evaluate(Xa, Xb):
    lr = LogisticRegression(max_iter=3000).fit(Xa, y_tr)
    kn = KNeighborsClassifier(5).fit(Xa, y_tr)
    return lr.score(Xb, y_te), kn.score(Xb, y_te)

print(f"\n{'Features':22s}{'LogReg':>10s}{'kNN':>10s}")
lr_a, kn_a = evaluate(X_tr_s, X_te_s); print(f"{'Original (64)':22s}{lr_a:10.4f}{kn_a:10.4f}")
for k in [2, 5, 10, 20, 30, 40]:
    p = PCA(n_components=k).fit(X_tr_s)
    lr_a, kn_a = evaluate(p.transform(X_tr_s), p.transform(X_te_s))
    print(f"{'PCA (' + str(k) + ')':22s}{lr_a:10.4f}{kn_a:10.4f}")
print("\nNote: t-SNE is a visualisation tool (non-parametric, no transform() for new data) so it is not used as a classifier input.")
