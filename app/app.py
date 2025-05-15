
import os
import pickle
import numpy as np
from flask import Flask, render_template, request, abort
import google.generativeai as genai  # legacy Gemini SDK import

GEMINI_API_KEY = "AIzaSyDYsHWL8IUYdLT-1HIYXOEJhR09lM0Oxsc"
# Configure Gemini API key (expects env var GEMINI_API_KEY)
genai.configure(api_key="AIzaSyDYsHWL8IUYdLT-1HIYXOEJhR09lM0Oxsc")

# Instantiate the Gemini chat model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load the life-expectancy ML model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    lr_model = pickle.load(f)

app = Flask(__name__)

# Generate lifestyle advice for normal users
def get_normal_advice(features: dict, predicted: float) -> str:
    prompt = (
        "I have a user with these health & demographic features:\n"
        + "\n".join(f"- {k}: {v}" for k, v in features.items())
        + f"\nTheir predicted life expectancy is {predicted:.1f} years.\n\n"
        "Please provide detailed lifestyle, dietary, and exercise recommendations "
        "to help increase their lifespan."
    )
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text.strip()

# Generate HIV-specific advice
def get_hiv_advice(stage: str, cd4: float, viral_load: float) -> str:
    prompt = (
        f"I have an HIV patient at {stage}. Their lab values are:\n"
        f"- CD4 count: {cd4}\n"
        f"- Viral load: {viral_load}\n\n"
        "Please:\n"
        "1) Recommend appropriate antiretroviral medications.\n"
        "2) Suggest monitoring and lifestyle advice.\n"
        "3) Offer motivating messages to support adherence."
    )
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    normal_result = None
    normal_advice = None
    hiv_advice = None

    if request.method == "POST":
        user_type = request.form.get("user_type", "")

        if user_type == "normal":
            keys = [
                "year","status","adult_mortality","alcohol","hepatitis_b",
                "measles","bmi","under_five_deaths","polio",
                "total_expenditure","diphtheria","hiv_aids","gdp",
                "population","thinness_1_19_years","income_composition","schooling"
            ]
            data = {}
            try:
                for k in keys:
                    val = request.form[k]
                    data[k] = int(val) if k in ["year","status","measles","under_five_deaths"] else float(val)
            except (KeyError, ValueError):
                abort(400, "Invalid form data for normal user")

            arr = np.array([[data[k] for k in keys]])
            predicted = float(lr_model.predict(arr)[0])
            normal_result = predicted
            normal_advice = get_normal_advice(data, predicted)

        elif user_type == "hiv":
            try:
                stage      = request.form["hiv_stage"]
                cd4        = float(request.form["cd4_count"])
                viral_load = float(request.form["viral_load"])
            except (KeyError, ValueError):
                abort(400, "Invalid form data for HIV patient")

            hiv_advice = get_hiv_advice(stage, cd4, viral_load)

    return render_template(
        "index.html",
        normal_result=normal_result,
        normal_advice=normal_advice,
        hiv_advice=hiv_advice
    )

if __name__ == "__main__":
    app.run(debug=True)
