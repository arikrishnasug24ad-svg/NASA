# Exoplanets ML example

This small project trains a RandomForest regressor to predict exoplanet `Mass` using features from `all_exoplanets_2021.csv`.

Files:
- `src/train.py` - the training script (reads CSV, preprocesses, trains, evaluates, saves a model).
- `src/app.py` - a small Flask app that loads the saved pipeline and exposes a `/predict` endpoint and static web UI.
- `web/index.html`, `web/style.css`, `web/app.js` - static client to call the API.
- `requirements.txt` - Python deps.

Quick start (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
python src\train.py --data "d:\\NASA\\all_exoplanets_2021.csv" --outdir "d:\\NASA\\models"
```

After training, run the web UI:

```powershell
python src\app.py
# open http://127.0.0.1:5000/ in a browser
```

Notes:
- The script imputes missing numeric features, encodes categorical features, and trains a RandomForestRegressor.
- It saves a `model.joblib` and prints evaluation metrics (MAE, RMSE, R^2).
- The web UI posts a small JSON of feature values to `/predict` and shows the predicted mass.

Next steps:
- hyperparameter tuning, cross-validation, feature importance analysis, model registry.

Public access / deployment
-------------------------
You have two simple options to make the app public:

1) Quick tunnel (ngrok) — good for demos
 - Install ngrok and run (after starting the local Flask app):
	 - `ngrok http 5000`
 - ngrok prints a public URL (https) which forwards to your local server. Share that URL.

2) Deploy to Render (recommended for a persistent public URL)
 - Push this repository to GitHub.
 - On Render.com create a new Web Service, connect your GitHub repo, choose the `Dockerfile` or use the `Procfile` option.
 - Render builds and provides a public HTTPS URL.

Notes:
- If you deploy to a public host, make sure `models/model.joblib` is included in the repo or stored in a persistent file store (S3, Render disk) and the app can access it. For larger models, use object storage and load from URL.
- For production, disable Flask debug mode and use a proper WSGI server.
