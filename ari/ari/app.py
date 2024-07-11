from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Dummy SVR model and encoders for demonstration
model = SVR(kernel='linear')

# Assume the model was trained with scaled numerical data and encoded categorical data
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Fit the label encoder with possible categories (example)
label_encoder.fit(['gaming', 'education', 'entertainment'])  # Update with actual categories

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        views_str = request.form.get('views', '')
        category_str = request.form.get('category', '')

        if not views_str:
            flash('Views field cannot be empty')
            return redirect(url_for('home'))

        try:
            views = float(views_str)
        except ValueError:
            flash('Views must be a number')
            return redirect(url_for('home'))

        if not category_str:
            flash('Category field cannot be empty')
            return redirect(url_for('home'))

        # Transform views and category
        views_scaled = scaler.transform(np.array([[views]]))
        category_encoded = label_encoder.transform([category_str])

        # Combine scaled views and encoded category
        features = np.hstack((views_scaled, category_encoded.reshape(-1, 1)))

        # Predict using the model
        prediction = model.predict(features)

        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        flash(f'An error occurred: {e}')
        return redirect(url_for('home'))

if __name__ == "_main_":
    app.run(debug=True)