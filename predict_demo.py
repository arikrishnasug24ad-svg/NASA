import joblib
import pandas as pd
import os

model_path = os.path.join('models','model.joblib')
print('Loading model from', model_path)
model = joblib.load(model_path)
sample = {
    'Num Stars': 1,
    'Num Planets': 1,
    'Discovery Year': 2020,
    'Orbital Period Days': 10.0,
    'Orbit Semi-Major Axis': 0.05,
    'Eccentricity': 0.0,
    'Insolation Flux': None,
    'Equilibrium Temperature': None,
    'Stellar Effective Temperature': None,
    'Stellar Radius': None,
    'Stellar Mass': 1.0,
    'Stellar Surface Gravity': None,
    'Distance': None,
    'Gaia Magnitude': None,
    'Discovery Method': 'Transit',
    'Spectral Type': 'G2 V'
}

df = pd.DataFrame([sample])
print('Input dataframe:\n', df.T)
pred = model.predict(df)
print('Predicted Mass:', float(pred[0]))
