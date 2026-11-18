"""
Program 13: Bagging (Random Forest) vs Boosting (AdaBoost / XGBoost) + feature importance
(XGBoost is optional: pip install xgboost. Falls back to sklearn GradientBoosting if absent.)
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, eval_metric="logloss", random_state=0); xgb_name = "XGBoost"
except ImportError:
    xgb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0); xgb_name = "GradientBoosting (XGBoost fallback)"

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)

models = {
    "Single Decision Tree": DecisionTreeClassifier(random_state=0),
    "Bagging (trees)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=0),
    xgb_name: xgb,
}
print(f"{'Model':40s}{'Test Acc':>10s}{'F1':>8s}{'CV Acc(5)':>11s}{'Time(s)':>9s}")
fitted = {}
for name, m in models.items():
    t = time.time(); m.fit(X_tr, y_tr); dt = time.time() - t
    p = m.predict(X_te); cv = cross_val_score(m, X, y, cv=5).mean()
    print(f"{name:40s}{accuracy_score(y_te, p):10.4f}{f1_score(y_te, p):8.4f}{cv:11.4f}{dt:9.2f}")
    fitted[name] = m

# ---- Feature importance ----
fig, ax = plt.subplots(1, 3, figsize=(20, 7))
for a, name in zip(ax, ["Random Forest", "AdaBoost", xgb_name]):
    imp = fitted[name].feature_importances_; idx = np.argsort(imp)[-10:]
    a.barh(np.array(data.feature_names)[idx], imp[idx]); a.set_title(f"Top-10 features: {name}", fontsize=10)
plt.tight_layout(); plt.show()

# ---- Error vs number of estimators ----
rf_err, ada_err = [], []
ns = [1, 5, 10, 25, 50, 100, 200]
for n in ns:
    rf_err.append(1 - RandomForestClassifier(n_estimators=n, random_state=0).fit(X_tr, y_tr).score(X_te, y_te))
    ada_err.append(1 - AdaBoostClassifier(n_estimators=n, learning_rate=0.5, random_state=0).fit(X_tr, y_tr).score(X_te, y_te))
plt.figure(figsize=(7, 4)); plt.plot(ns, rf_err, "o-", label="Random Forest (bagging)"); plt.plot(ns, ada_err, "s-", label="AdaBoost (boosting)")
plt.xlabel("# estimators"); plt.ylabel("Test error"); plt.title("Bagging vs Boosting"); plt.legend(); plt.grid(alpha=.3); plt.show()
print("\nBagging reduces VARIANCE (parallel, independent deep trees); Boosting reduces BIAS (sequential, each learner fixes previous errors).")
