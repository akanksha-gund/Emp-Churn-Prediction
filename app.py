import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naiveB.pkl', 'rb') as f:
  model = pickle.load(f)

# Modern, attractive responsive layout with embedded CSS (Tailwind via CDN)
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl overflow-hidden p-8">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                Employee Performance Predictor
            </h1>
            <p class="text-slate-400 text-sm mt-2">Naive Bayes Classification Model</p>
        </div>

        <!-- Prediction Form -->
        <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Education -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Education[cite: 1]</label>
                <select name="Education" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>

            <!-- JoiningYear -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Joining Year[cite: 1]</label>
                <input type="number" name="JoiningYear" required min="2012" max="2026" placeholder="e.g. 2018" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- City -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">City[cite: 1]</label>
                <select name="City" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>

            <!-- PaymentTier -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Payment Tier[cite: 1]</label>
                <select name="PaymentTier" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <!-- Age -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age[cite: 1]</label>
                <input type="number" name="Age" required min="18" max="70" placeholder="e.g. 28" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- Gender -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Gender[cite: 1]</label>
                <select name="Gender" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <!-- EverBenched -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Ever Benched[cite: 1]</label>
                <select name="EverBenched" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <!-- ExperienceInCurrentDomain -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Experience in Domain[cite: 1]</label>
                <input type="number" name="ExperienceInCurrentDomain" required min="0" max="20" placeholder="e.g. 3" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- Submit Button -->
            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full py-3 px-4 bg-gradient-to-r from-indigo-500 to-cyan-500 text-white font-semibold rounded-lg shadow-lg hover:from-indigo-600 hover:to-cyan-600 transition-all duration-200">
                    Predict Result
                </button>
            </div>
        </form>

        <!-- Prediction Result Display -->
        {% if prediction is not none %}
        <div class="mt-8 p-5 bg-slate-800/80 border border-slate-700 rounded-xl text-center shadow-inner">
            <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Prediction Outcome</h2>
            <p class="text-2xl font-bold mt-2 {% if prediction == 1 %}text-rose-400{% else %}text-emerald-400{% endif %}">
                {% if prediction == 1 %}
                    ⚠️ High Risk / Will Leave
                {% else %}
                    ✅ Low Risk / Will Stay
                {% endif %}
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction = None
  if request.method == 'POST':
    try:
      # Retrieve form variables matching feature names[cite: 1]
      education = request.form['Education']
      joining_year = float(request.form['JoiningYear'])
      city = request.form['City']
      payment_tier = float(request.form['PaymentTier'])
      age = float(request.form['Age'])
      gender = request.form['Gender']
      ever_benched = request.form['EverBenched']
      exp_domain = float(request.form['ExperienceInCurrentDomain'])

      # Categorical mappings (adjust these numbers if your training encoder used a different mapping)
      edu_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
      city_map = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
      gender_map = {'Male': 0, 'Female': 1}
      benched_map = {'No': 0, 'Yes': 1}

      features = np.array([[
          edu_map.get(education, 0),
          joining_year,
          city_map.get(city, 0),
          payment_tier,
          age,
          gender_map.get(gender, 0),
          benched_map.get(ever_benched, 0),
          exp_domain,
      ]])

      pred = model.predict(features)
      prediction = int(pred[0])
    except Exception as e:
      print(f'Error: {e}')

  return render_template_string(TEMPLATE, prediction=prediction)


if __name__ == '__main__':
  app.run(debug=True)
