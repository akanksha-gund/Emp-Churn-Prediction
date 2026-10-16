import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naiveB.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template (Tumcha purva sarkhaich advanced template ithe rahu dya)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root { --primary: #6366f1; --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); --card-bg: rgba(255, 255, 255, 0.95); --text-main: #0f172a; --border: #e2e8f0; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: var(--bg-gradient); color: var(--text-main); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 40px 20px; }
        .container { width: 100%; max-width: 750px; background: var(--card-bg); border-radius: 24px; box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.1); overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.8); }
        .header-banner { background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white; padding: 40px 30px; text-align: center; }
        .header-banner img { width: 70px; height: 70px; margin-bottom: 15px; }
        .header-banner h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
        form { padding: 40px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #475569; }
        input, select { padding: 12px 16px; border: 2px solid var(--border); border-radius: 12px; font-size: 15px; outline: none; background-color: #f8fafc; }
        .btn-submit { grid-column: span 2; background: var(--primary); color: white; border: none; padding: 16px; font-size: 16px; font-weight: 600; border-radius: 12px; cursor: pointer; }
        .result-card { grid-column: span 2; background: #e0e7ff; border-radius: 14px; padding: 20px; text-align: center; margin-top: 10px; }
        .result-card h3 { color: #4338ca; font-size: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-banner">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="Icon">
            <h1>Employee Intelligence Portal</h1>
            <p>Advanced Naive Bayes Machine Learning System</p>
        </div>
        <form method="POST">
            <div class="form-group">
                <label>Education Level</label>
                <select name="Education" required>
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>
            <div class="form-group">
                <label>Joining Year</label>
                <input type="number" name="JoiningYear" placeholder="e.g. 2018" required>
            </div>
            <div class="form-group">
                <label>City Location</label>
                <select name="City" required>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>
            <div class="form-group">
                <label>Payment Tier</label>
                <select name="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="Age" placeholder="e.g. 28" required>
            </div>
            <div class="form-group">
                <label>Gender</label>
                <select name="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Ever Benched?</label>
                <select name="EverBenched" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Experience in Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
            </div>
            <button type="submit" class="btn-submit">Run AI Prediction</button>
            {% if prediction_text %}
            <div class="result-card">
                <h3>{{ prediction_text }}</h3>
            </div>
            {% endif %}
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # 1. Input values ghene
            raw_education = request.form['Education']
            joining_year = float(request.form['JoiningYear'])
            raw_city = request.form['City']
            payment_tier = float(request.form['PaymentTier'])
            age = float(request.form['Age'])
            raw_gender = request.form['Gender']
            raw_ever_benched = request.form['EverBenched']
            experience = float(request.form['ExperienceInCurrentDomain'])

            # 2. Categorical values la numeric madhe map karne (Label Encoding sathi)
            # Jar tumhi training madhe wegli mapping keli asel, tar ithe tya nusar values badla.
            
            # Education Mapping (Eg: Bachelors: 0, Masters: 1, PHD: 2)
            edu_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
            education = edu_map.get(raw_education, 0)

            # City Mapping (Eg: Bangalore: 0, New Delhi: 1, Pune: 2 - Alphabetical order standard LabelEncoder)
            city_map = {'Bangalore': 0, 'New Delhi': 1, 'Pune': 2}
            city = city_map.get(raw_city, 0)

            # Gender Mapping (Female: 0, Male: 1)
            gender_map = {'Female': 0, 'Male': 1}
            gender = gender_map.get(raw_gender, 0)

            # EverBenched Mapping (No: 0, Yes: 1)
            benched_map = {'No': 0, 'Yes': 1}
            ever_benched = benched_map.get(raw_ever_benched, 0)

            # 3. DataFrame tayar karne numeric values sobat
            input_data = pd.DataFrame([[
                education, joining_year, city, payment_tier, 
                age, gender, ever_benched, experience
            ]], columns=[
                'Education', 'JoiningYear', 'City', 'PaymentTier', 
                'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
            ])

            # 4. Prediction run karne
            prediction = model.predict(input_data)
            
            # Output result (0 or 1 la readable format madhe dakhavne, optional)
            res = prediction[0]
            prediction_text = f"Prediction Result: Employee will leave (Class {res})" if res == 1 else f"Prediction Result: Employee will stay (Class {res})"

        except Exception as e:
            prediction_text = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
