# Rice Classification (Group 44)
 
A compact, reproducible machine‑learning project turning a class assignment into an **engineering‑ready repo**.  
Task: **binary classification** of rice grains (class1 vs class2) using geometric features.
 
## 🧰 Tech stack
- Python, NumPy, Pandas, scikit‑learn
- Jupyter Notebook for exploration & reporting
 
## 📦 Dataset
- File: `data/rice-final2.csv` (8 columns, 1,400 rows).  
- Preprocessing: `? → NaN → mean impute → MinMaxScaler`; label map `class1→0`, `class2→1`.
- If this file comes from a public source (e.g., UCI “Rice (Cammeo and Osmancik)”), please add the proper citation in **Dataset Card** below.
 
## 🏗️ Project structure
```
.
├── notebooks/
│   └── a1-set2-group44.ipynb
├── data/
│   └── rice-final2.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```
 
## 🔁 Reproducibility
```bash
# 1) create a clean env (Conda recommended)
conda create -n rice python=3.10 -y
conda activate rice
# 2) install deps
pip install -r requirements.txt
# 3) run the notebook
jupyter notebook notebooks/a1-set2-group44.ipynb
```
 
## 📊 Results (this repo’s exact preprocessing & splits)
10‑fold Stratified CV for non‑tuned models; grid‑search CV + held‑out test for tuned models (stratified, `random_state=0`).  
 
### Cross‑validation (no tuning)
| Model | 10‑fold CV Accuracy |
|---|---:|
| Logistic Regression | **0.9386** |
| Naïve Bayes | 0.9264 |
| Decision Tree (entropy) | 0.9179 |
| Bagging (DT base) | **0.9386** |
| AdaBoost (DT base) | **0.9421** |
| Gradient Boosting | 0.9329 |
 
### With tuning (grid‑search)
| Model | Best Params | CV Acc | Test Acc | Test F1 (macro / weighted) |
|---|---|---:|---:|---:|
| KNN | k=5, p=1 | 0.9371 | 0.9257 | — |
| SVM (RBF) | C=5, γ=1 | **0.9457** | 0.9343 | — |
| Random Forest | n_estimators=30, max_leaf_nodes=12, criterion=entropy, max_features=sqrt | 0.9390 | **0.9371** | **0.9355 / 0.9370** |
 
> TL;DR — **SVM (RBF)** gives the strongest CV; **RandomForest** gives the best held‑out **test** F1.
 
## ✅ What’s “engineering‑ready” here
- Pinned dependencies and a one‑command setup (`requirements.txt`).
- Clean data pipeline inside the notebook with fixed random seeds.
- Small dataset committed for easy reproduction.
 
## 🛠️ Next steps (good first issues)
1. **Split code** from notebook into `src/` modules (preprocess, train, evaluate) + add a tiny CLI (e.g., `python -m src.train --model svm`).  
2. **Tests** with `pytest` (unit test `scale→split→fit→predict` for each model).  
3. **Pre-commit**: `ruff`, `black`, `isort`.  
4. **CI**: GitHub Actions — run tests on push; add a status badge here.  
5. **Dataset Card**: source/license/fields/collection process; add a citation.  
6. **Model Card**: what’s optimized, metrics, limitations, responsible AI notes.
 
## 👥 Credits
- Group 44 — please list each teammate’s role (e.g., data prep, modeling, evaluation, report).
 
## 📄 License
MIT — see `LICENSE`.
 
## 📚 Dataset Card (template)
- **Source**: (link)
- **License**: (license name)
- **Task**: Binary classification of rice varieties via geometric features
- **Features**: Area, Perimeter, Major/Minor Axis Length, Eccentricity, Convex Area, Extent
- **Preprocessing**: mean impute for missing, MinMax scaling; label mapping 0/1
- **Intended use**: academic demo, ML benchmarking
- **Limitations**: small dataset, simple features; distribution shift not studied
- **Citation**: (add here if applicable)


---

## 🔧 CLI quickstart

```bash
# train with SVM (GridSearchCV) and save to artifacts/
python -m src.train --model svm

# predict on a CSV (adds a 'pred' column)
python scripts/predict_csv.py --model artifacts/best_model_svm.joblib --input data/rice-final2.csv --output preds.csv
```
