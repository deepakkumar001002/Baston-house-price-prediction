from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from model import train_and_save_model

# Train model if the pickle file doesn't exist
if not os.path.exists('house_model.pkl'):
    train_and_save_model()

app = Flask(__name__)
with open('house_model.pkl', 'rb') as f:
    model, feature_names = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = np.array([float(request.form[name]) for name in feature_names]).reshape(1, -1)
        pred = model.predict(features)[0]
        return render_template('index.html', prediction=round(pred, 2), feature_names=feature_names)
    return render_template('index.html', prediction=None, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
