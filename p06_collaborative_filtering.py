"""
Program 6: User-based and item-based collaborative filtering
Similarity: cosine or Pearson. Recommend top-N items for a user.
"""
import numpy as np
import pandas as pd

# ---------- Ratings matrix (0 / NaN = not rated) ----------
users = ["Asha", "Bhavesh", "Chitra", "Dev", "Esha", "Farhan"]
items = ["Inception", "Titanic", "Avengers", "Interstellar", "Notebook", "Matrix", "Frozen"]
R = pd.DataFrame([
    [5, 1, 4, 5, 0, 5, 0],
    [4, 0, 5, 4, 1, 4, 0],
    [1, 5, 0, 1, 5, 0, 4],
    [0, 4, 1, 0, 5, 1, 5],
    [5, 0, 4, 0, 2, 5, 1],
    [2, 4, 0, 2, 4, 1, 0],
], index=users, columns=items, dtype=float)
R = R.replace(0, np.nan)
print("Ratings matrix:\n", R.fillna("-"), "\n")

# ---------- Similarity measures ----------
def cosine_sim(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)           # co-rated entries only
    if mask.sum() == 0: return 0.0
    a, b = a[mask], b[mask]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0

def pearson_sim(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2: return 0.0
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0

def sim_matrix(M, fn):
    n = M.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = 1.0 if i == j else fn(M[i], M[j])
    return S

A = R.values
user_sim = pd.DataFrame(sim_matrix(A, cosine_sim), index=users, columns=users)
item_sim = pd.DataFrame(sim_matrix(A.T, cosine_sim), index=items, columns=items)
user_sim_p = pd.DataFrame(sim_matrix(A, pearson_sim), index=users, columns=users)
print("User-user cosine similarity:\n", user_sim.round(2), "\n")
print("Item-item cosine similarity:\n", item_sim.round(2), "\n")
print("User-user Pearson similarity:\n", user_sim_p.round(2), "\n")

# ---------- Prediction ----------
def predict_user_based(u, i, S, k=3):
    """Mean-centred weighted average of ratings by the k most similar users who rated item i."""
    ui, ii = users.index(u), items.index(i)
    mu = np.nanmean(A[ui])
    cands = [(S[ui, v], A[v, ii] - np.nanmean(A[v])) for v in range(len(users))
             if v != ui and not np.isnan(A[v, ii]) and S[ui, v] > 0]
    cands = sorted(cands, reverse=True)[:k]
    if not cands: return mu
    num = sum(s * d for s, d in cands); den = sum(abs(s) for s, _ in cands)
    return float(np.clip(mu + num / den, 1, 5))

def predict_item_based(u, i, S, k=3):
    """Weighted average of the user's own ratings on the k items most similar to i."""
    ui, ii = users.index(u), items.index(i)
    cands = [(S[ii, j], A[ui, j]) for j in range(len(items))
             if j != ii and not np.isnan(A[ui, j]) and S[ii, j] > 0]
    cands = sorted(cands, reverse=True)[:k]
    if not cands: return np.nanmean(A[ui])
    return float(sum(s * r for s, r in cands) / sum(s for s, _ in cands))

def recommend(u, method="user", sim="cosine", N=3):
    ui = users.index(u)
    unrated = [it for it, r in zip(items, A[ui]) if np.isnan(r)]
    if method == "user":
        S = user_sim.values if sim == "cosine" else user_sim_p.values
        scores = {it: predict_user_based(u, it, S) for it in unrated}
    else:
        S = item_sim.values if sim == "cosine" else pd.DataFrame(sim_matrix(A.T, pearson_sim)).values
        scores = {it: predict_item_based(u, it, S) for it in unrated}
    return sorted(scores.items(), key=lambda t: -t[1])[:N]

for u in ["Asha", "Dev"]:
    print(f"--- Top-3 recommendations for {u} ---")
    for method, sim in [("user", "cosine"), ("user", "pearson"), ("item", "cosine"), ("item", "pearson")]:
        recs = ", ".join(f"{it} ({s:.2f})" for it, s in recommend(u, method, sim, 3))
        print(f"{method.title()}-based / {sim:7s}: {recs}")
    print()
