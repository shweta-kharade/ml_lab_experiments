# Machine Learning Lab - 15 Programs

## Setup
    pip install -r requirements.txt      # xgboost & tensorflow are optional (fallbacks included)

## Run
    python p01_decision_boundary_cv.py   # any single program
    python run_all.py                    # all programs, one after another (close each plot to continue)

| # | File | Topic |
|---|------|-------|
| 1 | p01_decision_boundary_cv.py | Manual nearest-centroid classifier, manual train/test split & k-fold CV, Accuracy/Precision/Recall/F1 |
| 2 | p02_linear_regression.py | Simple & multiple linear regression (NumPy + sklearn), regression line, residuals, RMSE, R² |
| 3 | p03_decision_tree.py | Tree visualisation, depth control, cost-complexity pruning, overfitting on synthetic data |
| 4 | p04_knn.py | kNN from scratch, decision regions, tuning k, curse of dimensionality |
| 5 | p05_pca_tsne.py | PCA & t-SNE (Digits), 2D plots, accuracy before/after reduction |
| 6 | p06_collaborative_filtering.py | User-based & item-based CF, cosine / Pearson, top-N recommendations |
| 7 | p07_gaussian_naive_bayes.py | Gaussian NB manual vs sklearn; Iris + spam/ham |
| 8 | p08_logistic_regression.py | Sigmoid, decision boundary, ROC & AUC |
| 9 | p09_svm.py | Linear/RBF/poly SVM, margins, support vectors, tuning C & gamma |
| 10 | p10_perceptron_mlp.py | Perceptron from scratch (AND/OR/XOR), MLP with different activations (Keras optional) |
| 11 | p11_keras_dnn_digits.py | Keras 2-3 layer DNN for digits; layers, activations, loss plots |
| 12 | p12_sample_complexity_pac_vc.py | Learning curves, PAC bound, VC-dimension of linear classifiers |
| 13 | p13_bagging_vs_boosting.py | Random Forest vs AdaBoost vs XGBoost, feature importance |
| 14 | p14_kmeans.py | K-Means, elbow method, silhouette score |
| 15 | p15_gmm_agglomerative.py | GMM boundaries (BIC), agglomerative clustering, dendrograms |

Notes: all datasets are built into scikit-learn (no downloads needed). Random seeds are fixed for reproducible output.
