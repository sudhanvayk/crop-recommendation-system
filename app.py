from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

df = pd.read_csv("Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    inputs = {
        "N": float(request.form["N"]),
        "P": float(request.form["P"]),
        "K": float(request.form["K"]),
        "Temperature": float(request.form["temperature"]),
        "Humidity": float(request.form["humidity"]),
        "pH": float(request.form["ph"]),
        "Rainfall": float(request.form["rainfall"])
    }

    values = list(inputs.values())
    prediction = model.predict([values])[0]

    return render_template(
        "index.html",
        result=prediction,
        inputs=inputs
    )

if __name__ == "__main__":
    app.run(debug=True)
