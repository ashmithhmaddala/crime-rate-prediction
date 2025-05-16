from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os


app = Flask(__name__)
app.secret_key = "osmoCrime@2024_!secure"

# Load ML model and scaler
model = load_model("cnn_model.h5")
scaler = joblib.load("scaler.save")
model_features = pd.read_csv("model_features.csv", header=None).squeeze().tolist()
if isinstance(model_features, str):
    model_features = [model_features]


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_dict = {
        'Temperature': data['temperature'],
        'Rainfall': data['rainfall'],
        'Crime Severity': {'Low': 0, 'Moderate': 1, 'High': 2}[data['severity']],
        'Reported': 1 if data['reported'] == 'Yes' else 0,
        'Police Response Time': data['response_time'],
    }

    # Add one-hot encoded features
    for prefix, value in [
        ('Time of Day', data['time_of_day']),
        ('Socioeconomic Zone', data['socio_zone']),
        ('Area', data['area'])
    ]:
        for feature in model_features:
            if feature.startswith(f"{prefix}_"):
                input_dict[feature] = 1 if feature == f"{prefix}_{value}" else 0

    # Ensure all features are present
    for feature in model_features:
        if feature not in input_dict:
            input_dict[feature] = 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_features]

    # Scale and reshape input
    scaled = scaler.transform(input_df)
    reshaped = np.expand_dims(scaled, axis=2)

    # Predict
    prob = model.predict(reshaped)[0][0]
    prediction = "High Risk of Crime" if prob > 0.5 else "Relatively Safe"

    return jsonify({
        "prediction": prediction,
        "probability": round(prob * 100, 2)
    })

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = cur.fetchone()
        conn.close()

        if user:
            session["user_id"] = user[0]
            session["role"] = user[3]
            if user[3] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("user_dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/dashboard")
def user_dashboard():
    if "user_id" not in session or session.get("role") != "user":
        return redirect(url_for("login"))
    return render_template("dashboard_user.html", role="user")


@app.route("/admin")
def admin_dashboard():
    if "user_id" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))
    return render_template("dashboard_admin.html", role="admin")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
