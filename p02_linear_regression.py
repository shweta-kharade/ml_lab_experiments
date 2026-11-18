"""
Program 2: Simple & multiple linear regression using NumPy (normal equation) and scikit-learn
Plots regression line + residuals; computes RMSE and R^2.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
def r2(y, yhat): return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

class LinearRegressionNumPy:
    """Ordinary least squares: w = (X^T X)^-1 X^T y  (solved with lstsq for stability)."""
    def fit(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]
        self.w_ = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.w_

data = load_diabetes()
X_all, y = data.data, data.target
bmi_idx = data.feature_names.index("bmi")

# ------------------ SIMPLE regression (BMI -> disease progression) ------------------
X1 = X_all[:, [bmi_idx]]
X1_tr, X1_te, y_tr, y_te = train_test_split(X1, y, test_size=0.2, random_state=0)
np_s = LinearRegressionNumPy().fit(X1_tr, y_tr)
sk_s = LinearRegression().fit(X1_tr, y_tr)
print("=== Simple Linear Regression (BMI) ===")
print(f"NumPy  : intercept={np_s.w_[0]:.3f} slope={np_s.w_[1]:.3f} RMSE={rmse(y_te, np_s.predict(X1_te)):.3f} R2={r2(y_te, np_s.predict(X1_te)):.4f}")
print(f"sklearn: intercept={sk_s.intercept_:.3f} slope={sk_s.coef_[0]:.3f} RMSE={np.sqrt(mean_squared_error(y_te, sk_s.predict(X1_te))):.3f} R2={r2_score(y_te, sk_s.predict(X1_te)):.4f}")

# ------------------ MULTIPLE regression (all 10 features) ------------------
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=0)
np_m = LinearRegressionNumPy().fit(X_tr, y_tr)
sk_m = LinearRegression().fit(X_tr, y_tr)
print("\n=== Multiple Linear Regression (10 features) ===")
print(f"NumPy  : RMSE={rmse(y_te, np_m.predict(X_te)):.3f} R2={r2(y_te, np_m.predict(X_te)):.4f}")
print(f"sklearn: RMSE={np.sqrt(mean_squared_error(y_te, sk_m.predict(X_te))):.3f} R2={r2_score(y_te, sk_m.predict(X_te)):.4f}")
print("Coefficients match:", np.allclose(np_m.w_[1:], sk_m.coef_), "| intercept match:", np.isclose(np_m.w_[0], sk_m.intercept_))

# ------------------ Plots ------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
# simple: regression line
xs = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
ax[0, 0].scatter(X1_te, y_te, alpha=.6, label="test data")
ax[0, 0].plot(xs, np_s.predict(xs), "r", lw=2, label="NumPy fit")
ax[0, 0].plot(xs, sk_s.predict(xs), "g--", lw=2, label="sklearn fit")
ax[0, 0].set(title="Simple regression line", xlabel="BMI (standardised)", ylabel="Disease progression"); ax[0, 0].legend()
# simple: residuals
res1 = y_te - np_s.predict(X1_te)
ax[0, 1].scatter(np_s.predict(X1_te), res1, alpha=.6); ax[0, 1].axhline(0, color="r")
ax[0, 1].set(title="Simple regression residuals", xlabel="Predicted", ylabel="Residual")
# multiple: actual vs predicted
pred_m = np_m.predict(X_te)
ax[1, 0].scatter(y_te, pred_m, alpha=.6); lim = [y.min(), y.max()]
ax[1, 0].plot(lim, lim, "r--"); ax[1, 0].set(title="Multiple regression: actual vs predicted", xlabel="Actual", ylabel="Predicted")
# multiple: residuals
ax[1, 1].scatter(pred_m, y_te - pred_m, alpha=.6); ax[1, 1].axhline(0, color="r")
ax[1, 1].set(title="Multiple regression residuals", xlabel="Predicted", ylabel="Residual")
plt.tight_layout(); plt.show()
