import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naiveB.pkl', 'rb') as f:
    model = pickle.load(f)

# Advanced HTML Template with Modern Glassmorphism & Icons
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Employee Analytics Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #06b6d4;
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-main: #0f172a;
            --text-muted: #475569;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 750px;
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.8);
        }

        .header-banner {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }

        .header-banner img {
            width: 70px;
            height: 70px;
            margin-bottom: 15px;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
        }

        .header-banner h1 {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .header-banner p {
            font-size: 15px;
            opacity: 0.9;
            font-weight: 300;
        }

        form {
            padding: 40px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
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
            margin-bottom: 8px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        label i {
            color: var(--primary);
        }

        input, select {
            padding: 12px 16px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 15px;
            outline: none;
            transition: all 0.3s ease;
            background-color: #f8fafc;
            color: var(--text-main);
        }

        input:focus, select:focus {
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        .btn-submit {
            grid-column: span 2;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 16px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.5);
        }

        .result-card {
            grid-column: span 2;
            background: linear-gradient(135deg, #e0e7ff 0%, #fae8ff 100%);
            border: 1px solid #c7d2fe;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-top: 10px;
            animation: fadeIn 0.5s ease-in-out;
        }

        .result-card h3 {
            color: #4338ca;
            font-size: 20px;
            font-weight: 700;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            form {
                grid-template-columns: 1fr;
                padding: 20px;
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
            <!-- Attractive Vector Illustration / Image -->
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="Employee Analytics Icon">
            <h1>Employee Intelligence Portal</h1>
            <p>Advanced Naive Bayes Machine Learning System</p>
        </div>

        <form method="POST">
            <div class="form-group">
                <label for="Education"><i class="fa-solid fa-graduation-cap"></i> Education Level</label>
                <select name="Education" id="Education" required>
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>

            <div class="form-group">
                <label for="JoiningYear"><i class="fa-solid fa-calendar-days"></i> Joining Year</label>
                <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g. 2018" required>
            </div>

            <div class="form-group">
                <label for="City"><i class="fa-solid fa-city"></i> City Location</label>
                <select name="City" id="City" required>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>

            <div class="form-group">
                <label for="PaymentTier"><i class="fa-solid fa-layer-group"></i> Payment Tier</label>
                <select name="PaymentTier" id="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <div class="form-group">
                <label for="Age"><i class="fa-solid fa-user"></i> Age</label>
                <input type="number" name="Age" id="Age" placeholder="e.g. 28" required>
            </div>

            <div class="form-group">
                <label for="Gender"><i class="fa-solid fa-venus-mars"></i> Gender</label>
                <select name="Gender" id="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="EverBenched"><i class="fa-solid fa-briefcase"></i> Ever Benched?</label>
                <select name="EverBenched" id="EverBenched" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="ExperienceInCurrentDomain"><i class="fa-solid fa-clock-rotate-left"></i> Experience in Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
            </div>

            <button type="submit" class="btn-submit">
                <i class="fa-solid fa-wand-magic-sparkles"></i> Run AI Prediction
            </button>

            {% if prediction_text %}
            <div class="result-card">
                <h3><i class="fa-solid fa-circle-check"></i> {{ prediction_text }}</h3>
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
            # Extracting input values matching model features
            education = request.form['Education']
            joining_year = float(request.form['JoiningYear'])
            city = request.form['City']
            payment_tier = float(request.form['PaymentTier'])
            age = float(request.form['Age'])
            gender = request.form['Gender']
            ever_benched = request.form['EverBenched']
            experience = float(request.form['ExperienceInCurrentDomain'])

            # Create DataFrame matching feature names schema
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
    app.run(host='0.0.0.0', port=5000, debug=True)
