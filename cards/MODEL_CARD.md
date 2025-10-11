
# Model Card — Rice Classification
- **Algorithms**: LogReg, NB, DT, KNN, SVM(RBF), RF, GBDT, AdaBoost
- **Selection**: 10-fold CV baselines; GridSearchCV for SVM/RF/LogReg
- **Metrics**: accuracy, macro/weighted F1, ROC-AUC, PR-AUC
- **Reproducibility**: seeds fixed; deps pinned; preprocessing encapsulated
- **Intended use**: coursework → engineering-ready demo; not for safety-critical use
- **Notes**: For deployment, consider probability calibration and external validation.
