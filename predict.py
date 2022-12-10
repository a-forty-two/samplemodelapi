from flask import Flask
from joblib import load
import pandas as pd
app = Flask(__name__)
model = load('model.joblib') 
@app.route('/')
def welcome():
  return "Welcome to our model!"

@app.route('/predict/<r>/<a>')
def predict(r,a):
  r = float(r)
  a = float(a)
  df = pd.DataFrame({
      "radius_mean": [r],
      "area_mean":[a]
  })
  p = str(model.predict(df)[0])
  p = 'Malignant' if p=='1' else 'Benign'
  return {'result':p}

app.run()
