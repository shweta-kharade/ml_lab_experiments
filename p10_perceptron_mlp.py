"""
Program 10: Single-layer perceptron from scratch + multilayer perceptron (scikit-learn)
Compare activation functions. (Optional Keras version included at the bottom if TensorFlow is installed.)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- Single-layer perceptron (from scratch) ----------------
class Perceptron:
    """y_hat = step(w.x + b);  update: w <- w + lr*(y - y_hat)*x"""
    def __init__(self, lr=0.1, epochs=50): self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0; self.errors_ = []
        for _ in range(self.epochs):
            err = 0
            for xi, yi in zip(X, y):
                upd = self.lr * (yi - self.predict(xi.reshape(1, -1))[0])
                self.w += upd * xi; self.b += upd; err += int(upd != 0)
            self.errors_.append(err)
            if err == 0: break
        return self
    def predict(self, X): return (X @ self.w + self.b >= 0).astype(int)

X_ = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print("=== Perceptron on logic gates ===")
for name, y_ in {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}.items():
    p = Perceptron(0.1, 100).fit(X_, np.array(y_))
    print(f"{name:3s}: predictions={p.predict(X_)}  target={y_}  accuracy={np.mean(p.predict(X_) == y_):.2f}"
          + ("   <-- XOR is not linearly separable, single layer fails" if name == "XOR" else ""))

# Perceptron on Iris (setosa vs versicolor)
iris = load_iris(); mask = iris.target < 2
Xi, yi = iris.data[mask][:, [0, 2]], iris.target[mask]
pi = Perceptron(0.01, 100).fit(Xi, yi)
print(f"Iris (setosa vs versicolor) perceptron accuracy: {np.mean(pi.predict(Xi) == yi):.3f}, epochs to converge: {len(pi.errors_)}")

# ---------------- MLP with different activations ----------------
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0)
sc = StandardScaler().fit(X_tr); X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

acts = ["identity", "logistic", "tanh", "relu"]
fig, ax = plt.subplots(1, 5, figsize=(25, 4.5))
xx, yy = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-3, 3, 250))
print("\n=== MLP (2 hidden layers: 16, 8) - activation comparison ===")
print(f"{'Activation':10s}{'Train':>8s}{'Test':>8s}{'Iters':>8s}")
for a, act in zip(ax[:4], acts):
    m = MLPClassifier(hidden_layer_sizes=(16, 8), activation=act, max_iter=2000, random_state=0).fit(X_tr, y_tr)
    print(f"{act:10s}{m.score(X_tr, y_tr):8.3f}{m.score(X_te, y_te):8.3f}{m.n_iter_:8d}")
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, alpha=.3, cmap="coolwarm"); a.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolor="k", s=20)
    a.set_title(f"{act} (test={m.score(X_te, y_te):.3f})")
    ax[4].plot(m.loss_curve_, label=act)
ax[4].set(title="Training loss curves", xlabel="Iteration", ylabel="Loss"); ax[4].legend()
plt.tight_layout(); plt.show()

# ---------------- Optional: Keras version ----------------
try:
    from tensorflow import keras
    for act in ["linear", "sigmoid", "tanh", "relu"]:
        km = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(16, activation=act),
                               keras.layers.Dense(8, activation=act), keras.layers.Dense(1, activation="sigmoid")])
        km.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        km.fit(X_tr, y_tr, epochs=100, verbose=0)
        print(f"Keras {act:8s} test acc = {km.evaluate(X_te, y_te, verbose=0)[1]:.3f}")
except ImportError:
    print("\n(TensorFlow/Keras not installed - skipped optional Keras comparison. Install with: pip install tensorflow)")
