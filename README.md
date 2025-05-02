# Bengaluru Theft Classification Dashboard

This project is a machine learning-powered dashboard for predicting and analyzing theft crimes in Bengaluru using real-world-inspired datasets. It includes model training, interactive analytics, and a Streamlit web app with an interactive map.

---

## 🚀 Features

- **Predicts whether a crime is a theft or not using XGBoost**
- **Interactive Streamlit dashboard**
- **Visualizes crime trends, distributions, and area-wise analysis**
- **Interactive Folium map of recent crimes**
- **Handles class imbalance with SMOTE and class weights**
- **Easy to retrain with your own data**

---

## 📁 Project Structure

├── app.py
├── train_and_save_model.py
├── requirements.txt
├── hard_theft_classification_dataset.csv # Main dataset (sample or anonymized for GitHub)
├── model_accuracy.txt # (auto-generated, optional for GitHub)
├── model_features.csv # (auto-generated, optional for GitHub)
├── xgboost_model.pkl # (auto-generated, optional for GitHub)
├── .gitignore
└── README.md

---

## 🏁 Getting Started

### 1. **Install dependencies**

pip install -r requirements.txt

### 2. **Train the model**

python train_and_save_model.py

- This will train the XGBoost model, handle class imbalance, and save the model and feature list.

### 3. **Run the Streamlit app**

streamlit run app.py

---

## 🗂️ Files

- **app.py**: Streamlit dashboard for prediction and visualization.
- **train_and_save_model.py**: Model training script (XGBoost + SMOTE).
- **requirements.txt**: List of Python dependencies.
- **hard_theft_classification_dataset.csv**: Main data file (keep a small sample if pushing to GitHub).
- **model_accuracy.txt, model_features.csv, xgboost_model.pkl**: Auto-generated after training (optional to push).
- **.gitignore**: Ignore `.venv/`, large datasets, and model files.

---

## 📊 Dashboard Features

- **Data Overview**: See sample data and basic statistics.
- **Crime Analysis**: Visualize trends, types, and area-wise crime.
- **Map**: Interactive map of recent crimes by location and type.
- **Prediction**: Try out different scenarios and see theft probability.
- **About**: Project details and credits.

---

## 📝 Notes

- For privacy and size, only push a **small sample** of your dataset to GitHub.
- Do **not** push your `.venv/` folder or large `.pkl` files.
- You can retrain the model any time by running `train_and_save_model.py`.

---

## 🙏 Credits

- Data: Synthetic and anonymized for project use.
- Model: XGBoost, SMOTE (imbalanced-learn)
- Dashboard: Streamlit, Folium, Matplotlib, Seaborn

---

## 📬 Contact

For questions or feedback, please open an issue or contact Ashmith.






