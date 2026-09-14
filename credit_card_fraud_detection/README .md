# Credit Card Fraud Detection — Classification Project

Predicting fraudulent credit card transactions from anonymized (PCA-transformed)
transaction data, using a full ML pipeline: EDA → imbalance handling → model comparison →
hyperparameter tuning → evaluation with metrics suited to a highly imbalanced problem.

## Problem statement

Credit card fraud makes up a tiny fraction of all transactions (well under 1%). A model
that ignores this and optimizes for accuracy alone will look great on paper while catching
almost no fraud — so the entire project is built around imbalance-aware techniques and
metrics that actually reflect performance on the minority (fraud) class.

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  (ULB Machine Learning Group).
- **Features:** `Time`, `Amount`, and `V1`–`V28` (PCA-transformed, anonymized for privacy).
- **Target:** `Class` — `0` = legitimate, `1` = fraud.
- Not included in this repo (see [Setup](#setup) below) — download `creditcard.csv` from
  Kaggle and place it in the project root before running the notebook.

## Approach

1. **EDA** — class distribution, transaction amount by class, and feature correlation
   with fraud, to understand the imbalance and what signal is available.
2. **Train/test split before any resampling** — done first and stratified, so no
   synthetic or duplicated data can leak into the test set.
3. **Preprocessing** — a `ColumnTransformer` scales `Time`/`Amount` (the only features not
   already PCA-scaled) and passes `V1`–`V28` through unchanged.
4. **Imbalance handling** — `SMOTE` (synthetic oversampling of fraud) and, for the more
   compute-heavy models, `RandomUnderSampler` on the majority class first to keep training
   time practical. All resampling is wrapped inside `imblearn` pipelines so it only ever
   touches training folds.
5. **Models compared:**
   - Logistic Regression (baseline)
   - SVM (RBF kernel)
   - Random Forest, tuned via `RandomizedSearchCV` (scored on PR-AUC, not accuracy)
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, and PR-AUC (Average
   Precision), with an explanation of what each metric means and why PR-AUC/recall matter
   most for this problem specifically.

## Results

From the run recorded in the notebook (Random Forest after tuning):

| Metric | Score |
|---|---|
| Accuracy | 0.997 |
| Precision | 0.60 |
| Recall | 0.90 |
| F1-score | 0.72 |
| ROC-AUC | 0.998 |
| PR-AUC (Avg Precision) | 0.900 |

Random Forest outperformed both Logistic Regression and SVM on PR-AUC and F1, and was
selected as the best-performing model. Exact numbers will vary depending on the size of
the `creditcard.csv` you run it against (the full Kaggle dataset has ~285K rows; a smaller
sample will naturally shift the scores).

## Setup

```bash
# 1. Clone this repo, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download creditcard.csv from Kaggle and place it in the project root
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 4. Launch Jupyter and run the notebook top to bottom
jupyter notebook fraud_detection_classification_model.ipynb
```

## Project structure

```
.
├── fraud_detection_classification_model.ipynb   # main notebook — EDA, models, evaluation
├── requirements.txt                             # Python dependencies
├── README.md                                    # this file
└── creditcard.csv                               # dataset (not included — see Setup)
```

## Key design decisions worth knowing about

- **PR-AUC over ROC-AUC / accuracy** as the primary metric: with ~99.8% of transactions
  being legitimate, ROC-AUC can look deceptively good and accuracy is close to meaningless.
  PR-AUC focuses entirely on how well the model handles the fraud class.
- **Undersampling + SMOTE combined** for SVM and Random Forest: training these directly on
  the full dataset (or after a full SMOTE oversample) is computationally expensive at scale
  (SVM training cost grows roughly quadratically with row count), so the majority class is
  undersampled first to keep training tractable, then SMOTE tops up the minority class.
- **Resampling only ratio, never full 50/50 balance**: SMOTE is set to bring fraud up to
  10–30% of the majority class rather than a full balance, since fully balancing tends to
  hurt precision more than necessary.



## License

Dataset is provided by the ULB Machine Learning Group under Kaggle's terms; see the
[dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for details.
