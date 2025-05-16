import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

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

# Save feature names for Streamlit
pd.Series(X.columns).to_csv("model_features.csv", index=False, header=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.save")

# Reshape for Conv1D: (samples, timesteps, features)
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

# Build CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(16, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[es],
    verbose=2
)

# Evaluate
loss, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"Test accuracy: {acc:.2%}")

# Save model and accuracy
model.save("cnn_model.h5")
with open("model_accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")
