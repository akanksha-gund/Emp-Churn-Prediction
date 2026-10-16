import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naiveB.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML template with modern styling and responsive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #cbd5e1;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }

        .header-banner {
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header-banner h1 {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .header-banner p {
            font-size: 14px;
            opacity: 0.9;
        }

        form {
            padding: 30px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }

        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
            background-color: #fff;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-card {
            grid-column: span 2;
            background: #eef2ff;
            border: 1px: solid #c7d2fe;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            margin-top: 10px;
        }

        .result-card h3 {
            color: var(--primary);
            font-size: 18px;
            font-weight: 700;
        }

        @media (max-width: 600px) {
            form {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, .btn-submit, .result-card {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header-banner">
            <h1>Employee Analytics Portal</h1>
            <p>Powered by Naive Bayes Machine Learning Model</p>
        </div>

        <form method="POST">
            <div class="form-group">
                <label for="Education">Education</label>
                <select name="Education" id="Education" required>
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>

            <div class="form-group">
                <label for="JoiningYear">Joining Year</label>
                <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g. 2018" required>
            </div>

            <div class="form-group">
                <label for="City">City</label>
                <select name="City" id="City" required>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>

            <div class="form-group">
                <label for="PaymentTier">Payment Tier</label>
                <select name="PaymentTier" id="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <div class="form-group">
                <label for="Age">Age</label>
                <input type="number" name="Age" id="Age" placeholder="e.g. 28" required>
            </div>

            <div class="form-group">
                <label for="Gender">Gender</label>
                <select name="Gender" id="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="EverBenched">Ever Benched</label>
                <select name="EverBenched" id="EverBenched" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="ExperienceInCurrentDomain">Experience in Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>

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
            # Extracting input values matching model features[cite: 1]
            education = request.form['Education']
            joining_year = float(request.form['JoiningYear'])
            city = request.form['City']
            payment_tier = float(request.form['PaymentTier'])
            age = float(request.form['Age'])
            gender = request.form['Gender']
            ever_benched = request.form['EverBenched']
            experience = float(request.form['ExperienceInCurrentDomain'])

            # Create DataFrame matching feature names schema[cite: 1]
            input_data = pd.DataFrame([[
                education, joining_year, city, payment_tier, 
                age, gender, ever_benched, experience
            ]], columns=[
                'Education', 'JoiningYear', 'City', 'PaymentTier', 
                'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
            ])

            # Perform prediction
            prediction = model.predict(input_data)
            prediction_text = f"Prediction Result: {prediction[0]}"
        except Exception as e:
            prediction_text = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
