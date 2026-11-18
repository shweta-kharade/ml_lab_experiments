"""
Program 15: Gaussian Mixture Model (decision boundaries) and Agglomerative Clustering (dendrograms)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ===================== Part A: GMM =====================
X, y = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=5)
X = X @ np.array([[0.8, -0.5], [0.3, 0.9]])                       # stretch -> elliptical clusters
# choose number of components with BIC
bics = [GaussianMixture(k, random_state=0, n_init=5).fit(X).bic(X) for k in range(1, 9)]
print("BIC by #components:", dict(zip(range(1, 9), np.round(bics, 1))))
best = int(np.argmin(bics)) + 1; print("Best #components (lowest BIC):", best)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0, n_init=5).fit(X)
labels = gmm.predict(X)
print(f"ARI vs true labels = {adjusted_rand_score(y, labels):.4f}")
print("Means:\n", gmm.means_.round(2), "\nWeights:", gmm.weights_.round(3))

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(range(1, 9), bics, "o-"); ax[0].set(title="Model selection: BIC", xlabel="# components", ylabel="BIC")
ax[1].contourf(xx, yy, gmm.predict(grid).reshape(xx.shape), alpha=.3, cmap="viridis")      # hard decision regions
ax[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12); ax[1].scatter(*gmm.means_.T, c="r", marker="X", s=200)
ax[1].set_title("GMM decision boundaries")
logp = -gmm.score_samples(grid).reshape(xx.shape)
cs = ax[2].contourf(xx, yy, logp, levels=np.logspace(0, 2, 15) if False else 20, cmap="magma_r"); fig.colorbar(cs, ax=ax[2], label="-log likelihood")
ax[2].scatter(X[:, 0], X[:, 1], c="w", s=5, edgecolor="k", linewidths=.2); ax[2].set_title("GMM density (negative log-likelihood)")
plt.tight_layout(); plt.show()

# ===================== Part B: Agglomerative =====================
iris = load_iris(); Xi = iris.data
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
print("\nAgglomerative clustering on Iris (3 clusters):")
for a, link in zip(ax, ["ward", "complete", "average"]):
    Z = linkage(Xi, method=link)
    dendrogram(Z, truncate_mode="lastp", p=30, ax=a, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
    a.set_title(f"Dendrogram ({link} linkage)"); a.set_xlabel("Cluster (size)"); a.set_ylabel("Distance")
    lab = fcluster(Z, 3, criterion="maxclust"); a.axhline(Z[-3, 2] + 1e-9, c="r", ls="--")
    ag = AgglomerativeClustering(n_clusters=3, linkage=link).fit(Xi)
    print(f"  {link:9s} ARI={adjusted_rand_score(iris.target, ag.labels_):.4f}  silhouette={silhouette_score(Xi, ag.labels_):.4f}")
plt.tight_layout(); plt.show()

# 2D visualisation of the Ward clusters
from sklearn.decomposition import PCA
P = PCA(2).fit_transform(Xi); ag = AgglomerativeClustering(3, linkage="ward").fit(Xi)
plt.figure(figsize=(6, 5)); plt.scatter(P[:, 0], P[:, 1], c=ag.labels_, cmap="viridis"); plt.title("Agglomerative (Ward) clusters - PCA projection"); plt.show()
