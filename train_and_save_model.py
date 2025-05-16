import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/hard_theft_classification_dataset.csv")

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5,6]).astype(int)

# Clean and encode
df['Crime Severity'] = df['Crime Severity'].astype(str).str.strip().map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Crime Severity'] = df['Crime Severity'].fillna(df['Crime Severity'].mode()[0]).astype(int)
df['Reported'] = df['Reported'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0).astype(int)
df['Police Response Time'] = pd.to_numeric(df['Police Response Time'], errors='coerce')
df['Police Response Time'] = df['Police Response Time'].fillna(df['Police Response Time'].median())

categorical_cols = ['Time of Day', 'Socioeconomic Zone', 'Crime Type', 'Area']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Unknown')
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

drop_cols = ['Date', 'Latitude', 'Longitude', 'Is_Theft']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Is_Theft']
X = X.select_dtypes(include=['number', 'bool'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for class balancing
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# XGBoost with hyperparameter tuning
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [5, 7, 9, 12],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, 2, 3, 5, 10]
}
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, scoring='accuracy', n_jobs=-1, cv=cv, random_state=42, verbose=1)
search.fit(X_res, y_res)
xgb_best = search.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print("XGBoost Results:")
print(classification_report(y_test, xgb_pred))
print(f"XGBoost Test accuracy: {xgb_acc:.2%}")

# CatBoost
cat = CatBoostClassifier(verbose=0, random_state=42)
cat.fit(X_res, y_res)
cat_pred = cat.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)
print("CatBoost Results:")
print(classification_report(y_test, cat_pred))
print(f"CatBoost Test accuracy: {cat_acc:.2%}")

# Pick the best model
if cat_acc > xgb_acc:
    best_model = cat
    best_acc = cat_acc
    best_name = "CatBoost"
else:
    best_model = xgb_best
    best_acc = xgb_acc
    best_name = "XGBoost"

# Save the best model and feature names
dump(best_model, "xgboost_model.pkl")
X_train.columns.to_series().to_csv("model_features.csv", index=False, header=False)
with open("model_accuracy.txt", "w") as f:
    f.write(f"{best_acc:.4f}")

print(f"Best model: {best_name}. Test accuracy: {best_acc:.2%}")
print("Model and feature list saved.")
