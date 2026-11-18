"""
Program 7: Gaussian Naive Bayes - manual implementation vs scikit-learn
Part A: Iris (verification of manual implementation)
Part B: Spam / ham classification using word-count features
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GaussianNBScratch:
    """P(c|x) ∝ P(c) * Π_j N(x_j | mu_cj, var_cj)   (computed in log-space)"""
    def __init__(self, var_smoothing=1e-9): self.eps = var_smoothing
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        self.mu_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        eps = self.eps * X.var(axis=0).max()
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + eps
        return self
    def _log_post(self, X):
        out = []
        for k in range(len(self.classes_)):
            ll = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k]) + (X - self.mu_[k]) ** 2 / self.var_[k], axis=1)
            out.append(np.log(self.prior_[k]) + ll)
        return np.array(out).T
    def predict(self, X): return self.classes_[np.argmax(self._log_post(X), axis=1)]

# ---------------- Part A: Iris ----------------
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
m1 = GaussianNBScratch().fit(X_tr, y_tr); m2 = GaussianNB().fit(X_tr, y_tr)
print("=== Part A: Iris ===")
print(f"Manual  accuracy: {accuracy_score(y_te, m1.predict(X_te)):.4f}")
print(f"sklearn accuracy: {accuracy_score(y_te, m2.predict(X_te)):.4f}")
print("Predictions identical:", np.array_equal(m1.predict(X_te), m2.predict(X_te)))

# ---------------- Part B: Spam / ham ----------------
spam = ["Win a free iPhone now click here", "Congratulations you won a lottery prize claim now", "Free entry in weekly contest text WIN",
        "Urgent your account is suspended verify now", "Cheap loans approved instantly apply now", "Claim your free gift card today",
        "You have been selected for a cash prize call now", "Limited offer buy now get discount free shipping", "Earn money fast work from home guaranteed",
        "Exclusive deal win cash prize click link", "Get free recharge offer limited time", "Final notice claim your reward now",
        "Lowest price pills order now free delivery", "Winner winner claim free vacation now", "Act now free trial no credit card needed"]
ham = ["Are we meeting for lunch today", "Please send me the assignment notes", "Can you call me when you reach home",
       "The lab session is shifted to Monday morning", "Happy birthday have a great day", "Did you finish the DBMS record",
       "Let us study together for the algorithms exam", "Mom said dinner is ready", "I will submit the project report tomorrow",
       "Can you share the lecture slides please", "Team meeting at three in the seminar hall", "Thanks for helping me with the code",
       "See you at the college fest tomorrow", "Remember to bring your laptop for practical", "The NSS camp starts next week"]
texts = spam + ham
labels = np.array([1] * len(spam) + [0] * len(ham))        # 1 = spam, 0 = ham

vec = CountVectorizer(lowercase=True)
Xt = vec.fit_transform(texts).toarray().astype(float)
Xt_tr, Xt_te, yt_tr, yt_te, txt_tr, txt_te = train_test_split(Xt, labels, texts, test_size=.3, random_state=3, stratify=labels)
g1 = GaussianNBScratch(var_smoothing=1e-2).fit(Xt_tr, yt_tr)
g2 = GaussianNB(var_smoothing=1e-2).fit(Xt_tr, yt_tr)
p1, p2 = g1.predict(Xt_te), g2.predict(Xt_te)
print("\n=== Part B: Spam/Ham ===")
print(f"Manual  accuracy: {accuracy_score(yt_te, p1):.4f}")
print(f"sklearn accuracy: {accuracy_score(yt_te, p2):.4f}")
print("Confusion matrix (manual) [rows=true ham/spam]:\n", confusion_matrix(yt_te, p1))
print(classification_report(yt_te, p1, target_names=["ham", "spam"], zero_division=0))

def classify(msg):
    v = vec.transform([msg]).toarray().astype(float)
    return "SPAM" if g1.predict(v)[0] == 1 else "HAM"
for msg in ["Free cash prize claim now", "Are you coming to the lab tomorrow", "Win free gift now"]:
    print(f"'{msg}' -> {classify(msg)}")
