import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("hard_theft_classification_dataset.csv")

# Clean and encode
df['Crime Severity'] = df['Crime Severity'].astype(str).str.strip().map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Crime Severity'] = df['Crime Severity'].fillna(df['Crime Severity'].mode()[0]).astype(int)
df['Reported'] = df['Reported'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0).astype(int)
df['Police Response Time'] = pd.to_numeric(df['Police Response Time'], errors='coerce')
df['Police Response Time'] = df['Police Response Time'].fillna(df['Police Response Time'].median())

# One-hot encode categoricals
categorical_cols = ['Time of Day', 'Socioeconomic Zone', 'Crime Type', 'Area']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Unknown')
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop columns not used for modeling
drop_cols = ['Date', 'Latitude', 'Longitude', 'Is_Theft']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Is_Theft']
X = X.select_dtypes(include=['number', 'bool'])

# Train/test split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for balancing classes
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# XGBoost with balanced class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and feature names
dump(model, "xgboost_model.pkl")
X_train.columns.to_series().to_csv("model_features.csv", index=False, header=False)

# Save accuracy for the Streamlit app
accuracy = accuracy_score(y_test, y_pred)
with open("model_accuracy.txt", "w") as f:
    f.write(f"{accuracy:.4f}")

print(f"Model and feature list saved. Test accuracy: {accuracy:.2%}")
