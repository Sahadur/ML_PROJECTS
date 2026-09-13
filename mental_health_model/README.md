# Mental Health Score Predictor

A machine learning project that predicts a student's **Mental Health Score** based on their social media habits, lifestyle, and academic details. The project covers the full workflow: exploratory data analysis, data cleaning, feature engineering, and comparison of multiple regression models.

## Dataset

`Student_Social_Media_And_Mental_Health_Impact.csv` — ~5,000 student records with the following features:

| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Student's gender |
| Country | Student's country (111 unique values) |
| Academic_Level | High School / Undergraduate / Graduate |
| Most_Used_Platform | Primary social media platform used |
| Purpose_Of_Use | Reason for using social media (e.g. Networking, Entertainment, Education) |
| Avg_Daily_Usage_Hours | Average daily social media usage (hours) |
| Daily_Unlocks | Number of times the phone is unlocked per day |
| Study_Hours | Daily study hours |
| Physical_Activity_Hours | Daily physical activity (hours) |
| Sleep_Hours_Per_Night | Average sleep duration (hours) |
| Stress_Level | Low / Medium / High / Very High |
| **Mental_Health_Score** | Target variable (continuous score) |

## Approach

1. **EDA** — checked distributions, correlations, and relationships between usage/lifestyle features and the target score (e.g. stress level vs. mental health score, screen time vs. mental health score).
2. **Data Cleaning** — removed duplicates, checked for outliers using the IQR method, and clipped unrealistic values (e.g. negative physical activity hours).
3. **Feature Engineering** — the `Country` column had 111 unique values, so one-hot encoding directly would have caused a curse-of-dimensionality problem. The top 10 most frequent countries were kept as individual categories, and the rest were grouped into `Other`.
4. **Preprocessing pipeline** (via `sklearn.compose.ColumnTransformer`):
   - Log-transform + scale skewed numeric features (`Study_Hours`)
   - Scale other numeric features (`Age`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`)
   - Ordinal-encode `Stress_Level`
   - One-hot encode nominal categorical features (`Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_Country`)
5. **Modeling** — trained and compared four regression models:
   - Linear Regression
   - Random Forest Regressor
   - Random Forest Regressor (hyperparameter-tuned via `RandomizedSearchCV`)
   - XGBoost Regressor

## Results

| Model | Training R² | Testing R² | MSE | MAE |
|---|---|---|---|---|
| Linear Regression | 0.726 | 0.743 | 0.459 | 0.534 |
| Random Forest | 0.983 | 0.887 | 0.201 | 0.330 |
| Random Forest (tuned) | 0.982 | 0.887 | 0.201 | 0.332 |
| **XGBoost** | 0.966 | 0.882 | 0.211 | 0.339 |

Tree-based models (Random Forest, XGBoost) significantly outperformed Linear Regression, capturing non-linear relationships between lifestyle factors and mental health scores. The Random Forest models showed some overfitting (high train R² vs. lower test R²), which hyperparameter tuning only marginally improved. XGBoost was selected as the final model and saved for reuse.

**Best hyperparameters found (Random Forest, via RandomizedSearchCV):**
```
n_estimators: 200
max_depth: 20
min_samples_split: 2
min_samples_leaf: 1
```

## Project Structure

```
├── Mental_Health_Score_Predictor.ipynb   # Full notebook: EDA, cleaning, feature engineering, modeling
├── Student_Social_Media_And_Mental_Health_Impact.csv   # Dataset
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```






