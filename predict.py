from flask import Flask
from joblib import load
import pandas as pd
app = Flask(__app__)
model = load('model.joblib') 
@app.route('/')
def welcome():
  return "Welcome to our model!"

@app.route('/<r>/<a>')
def predict(r,a):
  df = pd.DataFrame({
      "radius_mean": [r],
      "area_mean":[a]
  })
  p = model.predict(df)
  return p




