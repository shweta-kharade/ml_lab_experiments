"""
Program 11: 2-3 layer DNN for digit classification using Keras
Visualise layers, activations and loss.
Requires: pip install tensorflow   (falls back to scikit-learn MLP if TensorFlow is unavailable)
Dataset : scikit-learn digits (8x8 images). To use MNIST instead, see the comment in load_data().
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data():
    d = load_digits()
    X = d.data / 16.0                                   # scale pixels to [0,1]
    # --- For MNIST use: (X_tr,y_tr),(X_te,y_te)=keras.datasets.mnist.load_data(); reshape to (-1,784)/255 ---
    return train_test_split(X, d.target, test_size=0.2, random_state=0, stratify=d.target)

X_tr, X_te, y_tr, y_te = load_data()

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    keras = None

if keras is not None:
    keras.utils.set_random_seed(0)
    model = keras.Sequential([
        layers.Input((64,)),
        layers.Dense(128, activation="relu", name="hidden1"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", name="hidden2"),
        layers.Dense(10, activation="softmax", name="output"),
    ])
    model.summary()                                      # layer-wise architecture & parameter count
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(X_tr, y_tr, validation_split=0.2, epochs=40, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest loss={loss:.4f}  Test accuracy={acc:.4f}")

    # ---- Loss / accuracy curves ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val"); ax[0].set(title="Loss", xlabel="Epoch"); ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val"); ax[1].set(title="Accuracy", xlabel="Epoch"); ax[1].legend()
    plt.tight_layout(); plt.show()

    # ---- Layer activations for one test image ----
    sample = X_te[:1]
    probe = keras.Model(model.inputs, [model.get_layer(n).output for n in ["hidden1", "hidden2", "output"]])
    acts = probe(sample)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    ax[0].imshow(sample.reshape(8, 8), cmap="gray"); ax[0].set_title(f"Input (true={y_te[0]})")
    for a, name, o in zip(ax[1:], ["hidden1 (128)", "hidden2 (64)", "output softmax (10)"], acts):
        a.bar(range(o.shape[1]), o.numpy()[0]); a.set_title(name + " activations")
    plt.tight_layout(); plt.show()

    # ---- First-layer weights as 8x8 filters ----
    W = model.get_layer("hidden1").get_weights()[0]
    fig, ax = plt.subplots(2, 8, figsize=(14, 4))
    for i, a in enumerate(ax.ravel()): a.imshow(W[:, i].reshape(8, 8), cmap="seismic"); a.axis("off")
    plt.suptitle("Learned weights of 16 neurons in hidden1"); plt.show()

    # ---- Predictions ----
    pred = model.predict(X_te, verbose=0).argmax(1)
    fig, ax = plt.subplots(2, 8, figsize=(14, 4))
    for i, a in enumerate(ax.ravel()):
        a.imshow(X_te[i].reshape(8, 8), cmap="gray"); a.axis("off")
        a.set_title(f"{pred[i]}", color="g" if pred[i] == y_te[i] else "r")
    plt.suptitle("Predictions (green = correct)"); plt.show()
else:
    print("TensorFlow/Keras not installed -> running scikit-learn MLP fallback (pip install tensorflow for the Keras version).")
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier((128, 64), activation="relu", max_iter=300, random_state=0).fit(X_tr, y_tr)
    print(f"Test accuracy = {m.score(X_te, y_te):.4f}")
    plt.plot(m.loss_curve_); plt.title("Training loss"); plt.xlabel("Iteration"); plt.show()
