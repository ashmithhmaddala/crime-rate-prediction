import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import os
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Theft Classification Dashboard", layout="wide")

# Load data and model
raw_df = pd.read_csv("hard_theft_classification_dataset.csv")
model = load("xgboost_model.pkl")
model_features = pd.read_csv("model_features.csv", header=None).squeeze().tolist()
if isinstance(model_features, str):
    model_features = [model_features]

# Load accuracy if available
if os.path.exists("model_accuracy.txt"):
    with open("model_accuracy.txt", "r") as f:
        accuracy = float(f.read())
    accuracy_str = f"Model Accuracy on Test Data: {accuracy:.2%}"
else:
    accuracy_str = "Model Accuracy: Not available. Please retrain the model."

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Overview", "Crime Analysis", "Map", "Prediction", "About"])

if section == "Data Overview":
    st.title("Dataset Overview")
    with st.expander("See a sample of the dataset"):
        st.dataframe(raw_df.head())
    st.subheader("Basic Statistics")
    st.write(f"Total records: {len(raw_df)}")
    st.write(f"Date range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")
    st.write(f"Number of crime types: {raw_df['Crime Type'].nunique()}")

elif section == "Crime Analysis":
    st.title("Crime Analysis")
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    st.subheader("Crime Trend Over Time")
    monthly_counts = raw_df.groupby(raw_df['Date'].dt.to_period('M')).size()
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_counts.plot(ax=ax)
    plt.xlabel("Month")
    plt.ylabel("Number of Crimes")
    plt.title("Monthly Crime Count in Bengaluru")
    st.pyplot(fig)
    st.subheader("Crime Type Distribution")
    crime_counts = raw_df['Crime Type'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=crime_counts.values, y=crime_counts.index, ax=ax)
    plt.xlabel("Number of Cases")
    plt.title("Crime Types in Bengaluru")
    st.pyplot(fig)
    st.subheader("Crimes by Area")
    area_counts = raw_df['Area'].value_counts()
    st.bar_chart(area_counts)

elif section == "Map":
    st.title("Crime Map of Bengaluru")
    st.markdown("Showing locations of recent crimes (red: Theft, blue: Not Theft).")
    # Filter to last 200 records for speed
    map_df = raw_df.tail(200).copy()
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    for _, row in map_df.iterrows():
        color = "red" if row["Is_Theft"] == 1 else "blue"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Crime Type']} ({row['Area']})",
        ).add_to(m)
    st_folium(m, width=800, height=500)

elif section == "Prediction":
    st.title("Theft Prediction")
    st.info(accuracy_str)
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", float(raw_df['Temperature'].min()), float(raw_df['Temperature'].max()), float(raw_df['Temperature'].mean()))
        rainfall = st.slider("Rainfall (mm)", float(raw_df['Rainfall'].min()), float(raw_df['Rainfall'].max()), float(raw_df['Rainfall'].mean()))
        severity = st.selectbox("Crime Severity", sorted(raw_df['Crime Severity'].dropna().unique()))
        reported = st.selectbox("Reported", sorted(raw_df['Reported'].dropna().unique()))
        if 'HighSeverityCrime' in raw_df.columns:
            high_sev = st.selectbox("High Severity Crime", sorted(raw_df['HighSeverityCrime'].dropna().unique()))
        else:
            high_sev = 0
    with col2:
        response_time = st.slider("Police Response Time (minutes)", float(raw_df['Police Response Time'].min()), float(raw_df['Police Response Time'].max()), float(raw_df['Police Response Time'].mean()))
        time_of_day = st.selectbox("Time of Day", sorted(raw_df['Time of Day'].dropna().unique()))
        socio_zone = st.selectbox("Socioeconomic Zone", sorted(raw_df['Socioeconomic Zone'].dropna().unique()))
        area = st.selectbox("Area", sorted(raw_df['Area'].dropna().unique()))

    input_dict = {
        'Temperature': temperature,
        'Rainfall': rainfall,
        'Crime Severity': {'Low': 0, 'Moderate': 1, 'High': 2}[severity] if isinstance(severity, str) else severity,
        'Reported': 1 if (reported == 'Yes' or reported == 1) else 0,
        'Police Response Time': response_time,
        'HighSeverityCrime': high_sev
    }
    # One-hot for Time of Day
    for tod in sorted(raw_df['Time of Day'].dropna().unique()):
        col_name = f'Time of Day_{tod}'
        if col_name in model_features:
            input_dict[col_name] = 1 if time_of_day == tod else 0
    # Socioeconomic Zone
    for zone in sorted(raw_df['Socioeconomic Zone'].dropna().unique()):
        col_name = f'Socioeconomic Zone_{zone}'
        if col_name in model_features:
            input_dict[col_name] = 1 if socio_zone == zone else 0
    # Area
    for a in sorted(raw_df['Area'].dropna().unique()):
        col_name = f'Area_{a}'
        if col_name in model_features:
            input_dict[col_name] = 1 if area == a else 0
    # Crime Type (all 0, since we're predicting Theft)
    for crime in raw_df['Crime Type'].dropna().unique():
        col_name = f'Crime Type_{crime}'
        if col_name in model_features:
            input_dict[col_name] = 0

    input_df = pd.DataFrame([input_dict])
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[model_features]

    if st.button("Predict Theft"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        if prediction == 1:
            st.success("✅ Prediction: THEFT")
            st.markdown(f"<span style='color:green;font-size:22px;font-weight:bold;'>Probability of theft: {proba[1]*100:.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:gray;'>Probability of not theft: {proba[0]*100:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.error("🚨 Prediction: NOT THEFT")
            st.markdown(f"<span style='color:red;font-size:22px;font-weight:bold;'>Probability of theft: {proba[1]*100:.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:green;'>Probability of not theft: {proba[0]*100:.2f}%</span>", unsafe_allow_html=True)

        with st.expander("ℹ️ What does this mean?"):
            st.write("""
            - **THEFT:** The model predicts this case is likely a theft.
            - **NOT THEFT:** The model predicts this case is likely another crime type.
            - Probabilities show the model's confidence in its prediction.
            """)

elif section == "About":
    st.title("About This Project")
    st.write("""
    This project demonstrates how machine learning can be used to predict thefts and analyze crime patterns.
    - Interactive crime trend visualization
    - Machine learning model to classify theft crimes
    - Area-based analysis and prediction
    - Interactive crime map of Bengaluru
    - Created by: Your Name
    """)
    st.markdown("---")
    st.markdown("© 2025 Theft Classification System | Created for educational purposes")
