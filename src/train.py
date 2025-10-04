import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    return model


def main(args):
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    # Target: Mass (assumed in Earth mass? Original file may have varying units; use as-is)
    target_col = 'Mass'
    if target_col not in df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in the data")

    # Select initial features - drop identifying columns and highly sparse ones
    drop_cols = ['No.', 'Planet Name', 'Planet Host', 'Discovery Facility']
    df = df.copy()
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Keep rows where Mass is present
    df = df[~df[target_col].isna()].reset_index(drop=True)
    print(f"Using {len(df)} rows with target present")

    # Reasonable numeric features to try
    numeric_candidates = ['Num Stars', 'Num Planets', 'Discovery Year', 'Orbital Period Days',
                          'Orbit Semi-Major Axis', 'Eccentricity', 'Insolation Flux', 'Equilibrium Temperature',
                          'Stellar Effective Temperature', 'Stellar Radius', 'Stellar Mass', 'Stellar Surface Gravity', 'Distance', 'Gaia Magnitude']

    numeric_features = [c for c in numeric_candidates if c in df.columns]

    # Categorical features
    categorical_candidates = ['Discovery Method', 'Spectral Type']
    categorical_features = [c for c in categorical_candidates if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(numeric_features, categorical_features)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, 'model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a regressor on exoplanet dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to all_exoplanets_2021.csv')
    parser.add_argument('--outdir', type=str, default='models', help='Output directory for model')
    args = parser.parse_args()
    main(args)
