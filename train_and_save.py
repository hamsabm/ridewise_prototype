# RideWise Bike Demand Prediction - Model Training Script
# Based on the actual bike sharing dataset from ridewise.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================
# Custom Feature Engineer Class
# ==============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer matching the notebook preprocessing
    """
    def __init__(self, peak_hours=None):
        if peak_hours is None:
            peak_hours = (7, 8, 9, 17, 18, 19)
        self.peak_hours = peak_hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Handle datetime conversion and feature extraction
        if 'dteday' in X.columns:
            X['dteday'] = pd.to_datetime(X['dteday'], errors='coerce')
            # Only create year/month if they don't already exist
            if 'year' not in X.columns and 'yr' not in X.columns:
                X['year'] = X['dteday'].dt.year
            if 'month' not in X.columns and 'mnth' not in X.columns:
                X['month'] = X['dteday'].dt.month
            if 'day' not in X.columns:
                X['day'] = X['dteday'].dt.day
            if 'weekday' not in X.columns:
                X['weekday'] = X['dteday'].dt.weekday

        # Normalize column names to match notebook format
        if 'yr' in X.columns:
            X['year'] = X['yr'] + 2011  # Convert 0,1 to 2011,2012
        if 'mnth' in X.columns:
            X['month'] = X['mnth']

        # Ensure required columns exist with defaults
        for col, default in [('year', 2024), ('month', 1), ('day', 1), ('weekday', 0)]:
            if col not in X.columns:
                X[col] = default

        # Weekend indicator (Saturday=5, Sunday=6)
        X['is_weekend'] = X['weekday'].apply(lambda d: 1 if int(d) >= 5 else 0)

        # Peak hour indicator (commuting hours from notebook: 7-9 AM, 5-7 PM)
        if 'hr' in X.columns:
            peak_set = set(self.peak_hours)
            X['is_peak_hour'] = X['hr'].apply(lambda h: 1 if int(h) in peak_set else 0)
        else:
            X['is_peak_hour'] = 0

        # Handle missing values
        if 'holiday' in X.columns:
            X['holiday'] = X['holiday'].fillna(0).astype(int)
        if 'workingday' in X.columns:
            X['workingday'] = X['workingday'].fillna(0).astype(int)

        return X

# ==============================
# Data Loading (Real Dataset)
# ==============================
def load_bike_sharing_data():
    """
    Load the real bike sharing dataset (hour.csv) exactly as in the notebook
    """
    try:
        # Try to load the actual dataset
        print("🔄 Loading bike sharing dataset (hour.csv)...")
        df = pd.read_csv("hour.csv")
        print("✅ Real dataset loaded successfully!")
        
    except FileNotFoundError:
        print("⚠️ hour.csv not found. Please download it from:")
        print("   https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        print("🔄 Creating sample dataset based on the real structure...")
        
        # Create sample data with the exact same structure as the real dataset
        np.random.seed(42)
        n_samples = 17379  # Same size as original dataset
        
        # Generate date range (2 years: 2011-2012)
        date_range = pd.date_range('2011-01-01', periods=n_samples, freq='h')
        
        df = pd.DataFrame({
            'instant': range(1, n_samples + 1),
            'dteday': date_range.date,
            'season': np.random.choice([1,2,3,4], n_samples),
            'yr': date_range.year - 2011,  # 0 for 2011, 1 for 2012
            'mnth': date_range.month,
            'hr': date_range.hour,
            'holiday': np.random.choice([0,1], n_samples, p=[0.97, 0.03]),
            'weekday': date_range.weekday,
            'workingday': ((date_range.weekday < 5) & 
                          (np.random.choice([0,1], n_samples, p=[0.03, 0.97]))).astype(int),
            'weathersit': np.random.choice([1,2,3,4], n_samples, p=[0.627, 0.263, 0.104, 0.006]),
            'temp': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),
            'atemp': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),
            'hum': np.random.normal(0.6, 0.15, n_samples).clip(0, 1),
            'windspeed': np.random.exponential(0.15, n_samples).clip(0, 1),
        })
        
        # Create realistic bike counts based on patterns from real data
        # Hour patterns (rush hours have higher demand)
        hour_multiplier = df['hr'].map({
            7: 1.8, 8: 2.2, 9: 1.6, 17: 2.0, 18: 2.4, 19: 1.8,
            12: 1.4, 13: 1.3, 16: 1.5, 6: 1.2, 10: 1.1, 11: 1.2,
            14: 1.2, 15: 1.3, 20: 1.4, 21: 1.2, 22: 0.9, 23: 0.7,
            0: 0.4, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.6
        })
        
        # Season patterns (summer/fall higher than winter/spring)
        season_multiplier = df['season'].map({1: 0.8, 2: 1.2, 3: 1.1, 4: 0.6})
        
        # Weather patterns
        weather_multiplier = df['weathersit'].map({1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2})
        
        # Temperature effect (optimal around 0.6-0.7)
        temp_effect = 1 + 0.5 * (1 - (df['temp'] - 0.65)**2 * 4)
        
        # Working day effect
        workingday_effect = df['workingday'].map({1: 1.1, 0: 0.9})
        
        # Base demand calculation
        base_cnt = 100 * hour_multiplier * season_multiplier * weather_multiplier * temp_effect * workingday_effect
        
        # Add noise and ensure realistic range
        df['cnt'] = (base_cnt + np.random.normal(0, 30, n_samples)).clip(1, 977)
        
        # Create casual and registered split (casual is typically 20-30% of total)
        casual_ratio = np.random.uniform(0.15, 0.35, n_samples)
        df['casual'] = (df['cnt'] * casual_ratio).round().astype(int)
        df['registered'] = df['cnt'] - df['casual']
        
        # Ensure non-negative values
        df['casual'] = df['casual'].clip(0)
        df['registered'] = df['registered'].clip(0)
        df['cnt'] = df['casual'] + df['registered']
    
    print(f"📊 Dataset summary:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Date range: {df['dteday'].min()} to {df['dteday'].max()}")
    print(f"   - Demand range: {df['cnt'].min()} to {df['cnt'].max()}")
    print(f"   - Mean demand: {df['cnt'].mean():.1f}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    return df

# ==============================
# Model Training Pipeline
# ==============================
def train_ridewise_model():
    """
    Train the RideWise bike demand prediction model
    """
    print("🚴‍♂️ Training RideWise Bike Demand Prediction Model...")
    
    # Load real training data
    df = load_bike_sharing_data()
    
    # Prepare features and target exactly as in the notebook
    y = df['cnt']
    
    # Drop target and leakage columns (casual, registered are components of cnt)
    X = df.drop(['cnt', 'casual', 'registered'], axis=1, errors='ignore')
    
    # Also drop instant (just row index) if present
    if 'instant' in X.columns:
        X = X.drop('instant', axis=1)
    
    # Define feature groups matching the notebook preprocessing
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'year', 'month', 'day', 'weekday']
    categorical_features = ['season', 'weathersit']
    
    # Add yr and mnth if they exist instead of year/month
    if 'yr' in X.columns and 'year' not in X.columns:
        numeric_features = [f if f != 'year' else 'yr' for f in numeric_features]
    if 'mnth' in X.columns and 'month' not in X.columns:
        numeric_features = [f if f != 'month' else 'mnth' for f in numeric_features]
    
    # Filter to only existing columns
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')
    
    # Create full pipeline with XGBoost model (best performer from notebook)
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Train the pipeline
    print("🔄 Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Performance:")
    print(f"   - MAE:  {mae:.2f}")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - R²:   {r2:.4f}")
    
    # Cross-validation (skip to avoid cloning issues for now)
    print("\n🔄 Skipping cross-validation to avoid cloning issues...")
    print(f"   - Single train/test split R² Score: {r2:.4f}")
    
    # Save the pipeline
    pipeline_filename = 'ridewise_pipeline.pkl'
    joblib.dump(pipeline, pipeline_filename)
    
    print(f"\n✅ RideWise pipeline saved as '{pipeline_filename}'")
    
    # Also save the legacy filename for compatibility
    legacy_filename = 'demand_prediction_pipeline.pkl'
    joblib.dump(pipeline, legacy_filename)
    print(f"✅ Legacy pipeline saved as '{legacy_filename}'")
    
    # Sample predictions
    print("\n🔮 Sample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        print(f"   Sample {i+1}: Actual={actual:.0f}, Predicted={predicted:.0f}")
    
    return pipeline, {'mae': mae, 'rmse': rmse, 'r2': r2}

# ==============================
# Alternative Model Training (Random Forest)
# ==============================
def train_alternative_model():
    """
    Train an alternative Random Forest model for comparison
    """
    print("\n🌲 Training alternative Random Forest model...")
    
    # Load real data
    df = load_bike_sharing_data()
    y = df['cnt']
    X = df.drop(['cnt'], axis=1)
    
    # Simpler pipeline for Random Forest
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'year', 'month', 'day', 'weekday']
    categorical_features = ['season', 'weathersit']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ], remainder='drop')
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)
    
    y_pred_rf = rf_pipeline.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"   - Random Forest MAE:  {mae_rf:.2f}")
    print(f"   - Random Forest RMSE: {rmse_rf:.2f}")
    print(f"   - Random Forest R²:   {r2_rf:.4f}")
    
    # Save alternative model
    joblib.dump(rf_pipeline, 'ridewise_rf_pipeline.pkl')
    print("✅ Random Forest model saved as 'ridewise_rf_pipeline.pkl'")
    
    return rf_pipeline, {'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("🚴‍♂️ RideWise Bike Demand Prediction - Model Training")
    print("=" * 60)
    
    # Train main XGBoost model
    main_pipeline, main_metrics = train_ridewise_model()
    
    # Train alternative Random Forest model
    alt_pipeline, alt_metrics = train_alternative_model()
    
    # Model comparison
    print("\n" + "=" * 60)
    print("📊 Model Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<15} {'XGBoost':<15} {'Random Forest':<15}")
    print("-" * 45)
    print(f"{'MAE':<15} {main_metrics['mae']:<15.2f} {alt_metrics['mae']:<15.2f}")
    print(f"{'RMSE':<15} {main_metrics['rmse']:<15.2f} {alt_metrics['rmse']:<15.2f}")
    print(f"{'R²':<15} {main_metrics['r2']:<15.4f} {alt_metrics['r2']:<15.4f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Files saved:")
    print(f"   - ridewise_pipeline.pkl (XGBoost - Main)")
    print(f"   - demand_prediction_pipeline.pkl (XGBoost - Legacy)")
    print(f"   - ridewise_rf_pipeline.pkl (Random Forest - Alternative)")
    
    print(f"\n🚀 To start the web application, run:")
    print(f"   python app.py")
    print("=" * 60)