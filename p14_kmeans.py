"""
Program 14: K-Means clustering, visualisation, tuning K with elbow method & silhouette score
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=42)

# ---------- Elbow + silhouette ----------
Ks = range(2, 11); inertia, sil = [], []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(f"K={k:2d}  inertia(WCSS)={km.inertia_:10.2f}  silhouette={sil[-1]:.4f}")
best_k = list(Ks)[int(np.argmax(sil))]
print(f"\nBest K by silhouette = {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(Ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia (WCSS)"); ax[0].grid(alpha=.3)
ax[1].plot(Ks, sil, "s-g"); ax[1].axvline(best_k, ls="--", c="r"); ax[1].set(title="Silhouette score", xlabel="K", ylabel="Average silhouette"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

# ---------- Cluster visualisation for several K ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, k in zip(ax, [2, best_k, 7]):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    a.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=15); a.scatter(*km.cluster_centers_.T, c="k", marker="X", s=200)
    a.set_title(f"K={k}  silhouette={silhouette_score(X, km.labels_):.3f}")
plt.tight_layout(); plt.show()

# ---------- Silhouette plot for best K ----------
km = KMeans(best_k, n_init=10, random_state=0).fit(X)
s = silhouette_samples(X, km.labels_)
plt.figure(figsize=(7, 5)); y_low = 10
for c in range(best_k):
    v = np.sort(s[km.labels_ == c]); y_up = y_low + len(v)
    plt.fill_betweenx(np.arange(y_low, y_up), 0, v, color=cm.tab10(c / 10), alpha=.8); plt.text(-0.05, y_low + .5 * len(v), str(c)); y_low = y_up + 10
plt.axvline(s.mean(), c="r", ls="--"); plt.title(f"Silhouette plot (K={best_k})"); plt.xlabel("Silhouette coefficient"); plt.ylabel("Cluster"); plt.show()
