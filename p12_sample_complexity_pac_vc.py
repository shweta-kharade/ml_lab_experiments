"""
Program 12: Sample complexity, PAC learning (toy), VC-dimension of linear classifiers (conceptual lab)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import LinearSVC

# =============== Part A: sample complexity (learning curve) ===============
X, y = load_digits(return_X_y=True); X = X / 16.0
sizes, tr_sc, te_sc = learning_curve(LogisticRegression(max_iter=2000), X, y, train_sizes=np.linspace(0.03, 1, 12),
                                     cv=5, shuffle=True, random_state=0)
print("=== Part A: Performance vs training size ===")
for s, a, b in zip(sizes, tr_sc.mean(1), te_sc.mean(1)): print(f"  m={s:5d}  train acc={a:.3f}  validation acc={b:.3f}")

# =============== Part B: PAC learning, finite hypothesis class ===============
# Class H = conjunctions of literals over n boolean variables -> |H| = 3^n.
# Realizable-case bound: m >= (1/eps) * ( ln|H| + ln(1/delta) )
def pac_bound(H_size, eps, delta): return int(np.ceil((np.log(H_size) + np.log(1 / delta)) / eps))
print("\n=== Part B: PAC bound for conjunctions over n boolean vars (|H|=3^n) ===")
for n in [5, 10, 20]:
    for eps, delta in [(0.1, 0.05), (0.05, 0.05), (0.01, 0.01)]:
        print(f"  n={n:2d} eps={eps:<5} delta={delta:<5} -> m >= {pac_bound(3 ** n, eps, delta)}")

# Empirical check: learn a target conjunction with the "elimination" algorithm and measure error vs m
rng = np.random.default_rng(0); n = 10
target = {0: 1, 3: 0, 7: 1}              # x0 AND (NOT x3) AND x7
def label(x): return int(all(x[i] == v for i, v in target.items()))
def learn_conjunction(Xs, ys):
    lits = {(i, v) for i in range(n) for v in (0, 1)}
    for x, l in zip(Xs, ys):
        if l == 1: lits = {(i, v) for (i, v) in lits if x[i] == v}   # drop literals contradicted by positives
    return lits
def predict(lits, x): return int(all(x[i] == v for i, v in lits))
# distribution biased so positives appear reasonably often
def sample(m):
    Xs = rng.integers(0, 2, (m, n)); pos = rng.random(m) < 0.4
    Xs[pos, 0], Xs[pos, 3], Xs[pos, 7] = 1, 0, 1
    return Xs, np.array([label(x) for x in Xs])
Xtest, ytest = sample(5000)
ms = [2, 5, 10, 20, 40, 80, 160, 320]; errs = []
for m in ms:
    e = []
    for _ in range(100):
        Xs, ys = sample(m); h = learn_conjunction(Xs, ys)
        e.append(np.mean([predict(h, x) != t for x, t in zip(Xtest[:1000], ytest[:1000])]))
    errs.append(np.mean(e))
print("\nEmpirical error of conjunction learner vs sample size m:")
for m, e in zip(ms, errs): print(f"  m={m:4d} -> error={e:.4f}")

# =============== Part C: VC dimension of linear classifiers in R^d ===============
def linearly_separable(P, labels):
    if len(set(labels)) == 1: return True
    clf = LinearSVC(C=1e6, max_iter=200000, tol=1e-8, dual=True).fit(P, labels)
    return clf.score(P, labels) == 1.0

def can_shatter(P):
    return all(linearly_separable(P, np.array(l)) for l in product([0, 1], repeat=len(P)))

print("\n=== Part C: VC dimension of lines in R^2 (expected = 3) ===")
three = np.array([[0, 0], [1, 0], [0, 1]], float)
four  = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)           # XOR configuration (square)
print("3 points in general position shattered by a line? ", can_shatter(three))
print("4 points (square / XOR labelling)  shattered?      ", can_shatter(four))
for d in [1, 2, 3]:
    pts = np.vstack([np.zeros(d), np.eye(d)])                        # d+1 affinely independent points
    print(f"R^{d}: {d + 1} points shattered? {can_shatter(pts)}   -> VC dim of hyperplanes in R^{d} = {d + 1}")
rng2 = np.random.default_rng(1)
print("R^2: random 4-point sets shattered:", sum(can_shatter(rng2.random((4, 2))) for _ in range(30)), "/ 30  (never, VC=3)")

# plots
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(sizes, tr_sc.mean(1), "o-", label="train"); ax[0].plot(sizes, te_sc.mean(1), "s-", label="validation")
ax[0].set(title="Sample complexity (Digits, LogReg)", xlabel="Training size m", ylabel="Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].loglog(ms, errs, "o-"); ax[1].set(title="PAC: error of conjunction learner", xlabel="m", ylabel="Test error"); ax[1].grid(alpha=.3, which="both")
ax[2].scatter(*four.T, c=[0, 0, 1, 1], cmap="coolwarm", s=200, edgecolor="k")
ax[2].set(title="XOR labelling of 4 points: no line separates -> VC(lines in R²)=3", xlim=(-.5, 1.5), ylim=(-.5, 1.5))
plt.tight_layout(); plt.show()
