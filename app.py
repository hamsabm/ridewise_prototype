# RideWise Bike Demand Prediction Backend
# Flask application for bike sharing demand prediction

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import datetime
import os

# ==============================
# Custom Feature Engineer Class (based on ridewise.ipynb)
# ==============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that creates derived features
    from raw bike sharing data, matching the notebook preprocessing.
    """
    def __init__(self, peak_hours=None):
        if peak_hours is None:
            peak_hours = (7, 8, 9, 17, 18, 19)
        self.peak_hours = peak_hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert dteday to datetime if present and extract features
        if 'dteday' in X.columns:
            X['dteday'] = pd.to_datetime(X['dteday'], errors='coerce')
            X['year'] = X['dteday'].dt.year
            X['month'] = X['dteday'].dt.month
            X['day'] = X['dteday'].dt.day
            X['weekday'] = X['dteday'].dt.weekday  # Monday=0, Sunday=6
        else:
            # Fallback for missing date columns
            X['year'] = X.get('yr', 2024)
            X['month'] = X.get('mnth', 1)
            X['day'] = X.get('day', 1)
            X['weekday'] = X.get('weekday', 0)

        # Weekend indicator (Saturday=5, Sunday=6)
        X['is_weekend'] = X['weekday'].apply(lambda d: 1 if int(d) >= 5 else 0)

        # Peak hour indicator (commuting hours)
        if 'hr' in X.columns:
            peak_set = set(self.peak_hours)
            X['is_peak_hour'] = X['hr'].apply(lambda h: 1 if int(h) in peak_set else 0)
        else:
            X['is_peak_hour'] = 0

        # Handle holiday and workingday columns
        if 'holiday' in X.columns:
            X['holiday'] = X['holiday'].fillna(0).astype(int)
        if 'workingday' in X.columns:
            X['workingday'] = X['workingday'].fillna(0).astype(int)

        return X

# ==============================
# Load or Create Model Pipeline
# ==============================
pipeline = None

def load_or_create_pipeline():
    """Load existing pipeline or create a new one"""
    global pipeline
    
    try:
        # Try to load existing pipeline
        pipeline = joblib.load("ridewise_pipeline.pkl")
        print("✅ RideWise pipeline loaded successfully.")
        return True
    except FileNotFoundError:
        print("⚠️ Pipeline not found.")
        print("💡 Please run: python train_and_save.py")
        print("   Or download real data with: python download_dataset.py")
        return False

def create_sample_pipeline():
    """Create and train a new pipeline with sample data"""
    global pipeline
    
    try:
        # Generate sample training data (mimicking bike sharing dataset)
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample datetime range
        date_range = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        sample_data = pd.DataFrame({
            'dteday': date_range,
            'season': np.random.choice([1,2,3,4], n_samples),
            'hr': date_range.hour,
            'holiday': np.random.choice([0,1], n_samples, p=[0.95, 0.05]),
            'weekday': date_range.weekday,
            'workingday': np.random.choice([0,1], n_samples, p=[0.3, 0.7]),
            'weathersit': np.random.choice([1,2,3,4], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'temp': np.random.uniform(0.1, 0.9, n_samples),
            'atemp': np.random.uniform(0.1, 0.9, n_samples),
            'hum': np.random.uniform(0.3, 0.9, n_samples),
            'windspeed': np.random.uniform(0.1, 0.6, n_samples),
        })
        
        # Create realistic target based on features
        # Higher demand during peak hours, good weather, weekdays
        base_demand = 50
        peak_boost = sample_data['hr'].apply(lambda h: 100 if h in [7,8,9,17,18,19] else 0)
        weather_penalty = sample_data['weathersit'].apply(lambda w: 0 if w<=2 else -30*(w-2))
        weekend_boost = sample_data['weekday'].apply(lambda d: 50 if d >= 5 else 0)
        temp_boost = sample_data['temp'] * 200
        
        sample_data['cnt'] = (base_demand + peak_boost + weather_penalty + 
                             weekend_boost + temp_boost + 
                             np.random.normal(0, 30, n_samples)).clip(0, 1000)
        
        # Prepare features and target
        y = sample_data['cnt']
        X = sample_data.drop(['cnt'], axis=1)
        
        # Define feature groups for preprocessing
        numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'year', 'month', 'day', 'weekday']
        categorical_features = ['season', 'weathersit']
        
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
        
        # Create full pipeline with XGBoost model
        xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        
        pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('model', xgb_model)
        ])
        
        # Train the pipeline
        pipeline.fit(X, y)
        
        # Save the trained pipeline
        joblib.dump(pipeline, "ridewise_pipeline.pkl")
        print("✅ New RideWise pipeline created and saved successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return False

# ==============================
# Flask App Setup
# ==============================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load pipeline on startup
load_or_create_pipeline()

# ==============================
# HTML Template for Web Interface
# ==============================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚴‍♂️ RideWise - Bike Demand Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin: 2rem auto;
            max-width: 800px;
        }
        .header {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 2rem;
            border-radius: 20px 20px 0 0;
            text-align: center;
        }
        .form-section {
            padding: 2rem;
        }
        .prediction-result {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 1rem;
        }
        .prediction-number {
            font-size: 3rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .input-group {
            margin-bottom: 1.5rem;
        }
        .btn-predict {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            border: none;
            padding: 1rem 2rem;
            border-radius: 50px;
            font-weight: bold;
            font-size: 1.1rem;
            transition: transform 0.2s;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .weather-icons {
            display: flex;
            justify-content: space-around;
            margin: 1rem 0;
        }
        .weather-option {
            text-align: center;
            padding: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
            margin: 0 0.5rem;
        }
        .weather-option:hover, .weather-option.active {
            border-color: #2196F3;
            background: #f0f8ff;
        }
        .loading {
            display: none;
        }
        .loading.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <div class="header">
                <h1><i class="fas fa-bicycle"></i> RideWise</h1>
                <p class="mb-0">AI-Powered Bike Sharing Demand Prediction</p>
            </div>
            
            <div class="form-section">
                <form id="predictionForm">
                    <!-- Date and Time Section -->
                    <div class="row input-group">
                        <div class="col-md-6">
                            <label class="form-label"><i class="fas fa-calendar"></i> Date</label>
                            <input type="date" class="form-control" id="date" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label"><i class="fas fa-clock"></i> Hour (0-23)</label>
                            <input type="number" class="form-control" id="hour" min="0" max="23" value="12" required>
                        </div>
                    </div>
                    
                    <!-- Season Selection -->
                    <div class="input-group">
                        <label class="form-label"><i class="fas fa-leaf"></i> Season</label>
                        <select class="form-select" id="season" required>
                            <option value="1">🌸 Spring</option>
                            <option value="2" selected>☀️ Summer</option>
                            <option value="3">🍂 Fall</option>
                            <option value="4">❄️ Winter</option>
                        </select>
                    </div>
                    
                    <!-- Weather Situation -->
                    <div class="input-group">
                        <label class="form-label"><i class="fas fa-cloud-sun"></i> Weather Condition</label>
                        <div class="weather-icons">
                            <div class="weather-option active" data-weather="1">
                                <i class="fas fa-sun" style="font-size: 2rem; color: #FFD700;"></i>
                                <div>Clear</div>
                            </div>
                            <div class="weather-option" data-weather="2">
                                <i class="fas fa-cloud" style="font-size: 2rem; color: #87CEEB;"></i>
                                <div>Cloudy</div>
                            </div>
                            <div class="weather-option" data-weather="3">
                                <i class="fas fa-cloud-rain" style="font-size: 2rem; color: #4682B4;"></i>
                                <div>Light Rain</div>
                            </div>
                            <div class="weather-option" data-weather="4">
                                <i class="fas fa-cloud-showers-heavy" style="font-size: 2rem; color: #2F4F4F;"></i>
                                <div>Heavy Rain</div>
                            </div>
                        </div>
                        <input type="hidden" id="weathersit" value="1">
                    </div>
                    
                    <!-- Environmental Conditions -->
                    <div class="row input-group">
                        <div class="col-md-4">
                            <label class="form-label"><i class="fas fa-thermometer-half"></i> Temperature</label>
                            <input type="range" class="form-range" id="temp" min="0" max="1" step="0.01" value="0.5">
                            <div class="text-center" id="tempValue">0.5</div>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label"><i class="fas fa-tint"></i> Humidity</label>
                            <input type="range" class="form-range" id="humidity" min="0" max="1" step="0.01" value="0.6">
                            <div class="text-center" id="humidityValue">0.6</div>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label"><i class="fas fa-wind"></i> Wind Speed</label>
                            <input type="range" class="form-range" id="windspeed" min="0" max="1" step="0.01" value="0.2">
                            <div class="text-center" id="windspeedValue">0.2</div>
                        </div>
                    </div>
                    
                    <!-- Special Days -->
                    <div class="row input-group">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input type="checkbox" class="form-check-input" id="holiday">
                                <label class="form-check-label">
                                    <i class="fas fa-star"></i> Holiday
                                </label>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input type="checkbox" class="form-check-input" id="workingday" checked>
                                <label class="form-check-label">
                                    <i class="fas fa-briefcase"></i> Working Day
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Predict Button -->
                    <div class="text-center">
                        <button type="submit" class="btn btn-predict text-white">
                            <i class="fas fa-chart-line"></i> Predict Demand
                        </button>
                        <div class="loading">
                            <i class="fas fa-spinner fa-spin"></i> Calculating...
                        </div>
                    </div>
                </form>
                
                <!-- Results Section -->
                <div id="results" style="display: none;">
                    <div class="prediction-result">
                        <h4><i class="fas fa-bicycle"></i> Predicted Bike Demand</h4>
                        <div class="prediction-number" id="predictionValue">0</div>
                        <p>bikes expected to be rented</p>
                    </div>
                </div>
                
                <!-- Error Section -->
                <div id="error" class="alert alert-danger" style="display: none;">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span id="errorMessage"></span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Set today's date as default
        document.getElementById('date').value = new Date().toISOString().split('T')[0];
        
        // Weather selection
        document.querySelectorAll('.weather-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.weather-option').forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                document.getElementById('weathersit').value = this.dataset.weather;
            });
        });
        
        // Range sliders with live values
        ['temp', 'humidity', 'windspeed'].forEach(id => {
            const slider = document.getElementById(id);
            const display = document.getElementById(id + 'Value');
            slider.addEventListener('input', function() {
                display.textContent = this.value;
            });
        });
        
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const loading = document.querySelector('.loading');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            
            // Show loading
            loading.classList.add('show');
            results.style.display = 'none';
            error.style.display = 'none';
            
            // Collect form data
            const formData = {
                dteday: document.getElementById('date').value,
                hr: parseInt(document.getElementById('hour').value),
                season: parseInt(document.getElementById('season').value),
                weathersit: parseInt(document.getElementById('weathersit').value),
                temp: parseFloat(document.getElementById('temp').value),
                atemp: parseFloat(document.getElementById('temp').value), // Use same as temp
                hum: parseFloat(document.getElementById('humidity').value),
                windspeed: parseFloat(document.getElementById('windspeed').value),
                holiday: document.getElementById('holiday').checked ? 1 : 0,
                workingday: document.getElementById('workingday').checked ? 1 : 0,
                weekday: new Date(document.getElementById('date').value).getDay()
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('predictionValue').textContent = Math.round(data.prediction);
                    results.style.display = 'block';
                } else {
                    throw new Error(data.error || 'Prediction failed');
                }
                
            } catch (err) {
                document.getElementById('errorMessage').textContent = err.message;
                error.style.display = 'block';
            } finally {
                loading.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

# ==============================
# Utility Functions
# ==============================
def build_input_dataframe(payload):
    """Convert API payload to DataFrame for prediction"""
    # Parse date
    dteday = pd.to_datetime(payload.get('dteday', '2024-01-01'))
    
    # Create DataFrame with all required features matching the real dataset format
    data = {
        'dteday': [dteday],
        'season': [payload.get('season', 1)],
        'yr': [dteday.year - 2011],  # Real dataset uses years as 0,1 (2011,2012)
        'mnth': [dteday.month],      # Real dataset uses mnth not month
        'hr': [payload.get('hr', 12)],
        'holiday': [payload.get('holiday', 0)],
        'weekday': [payload.get('weekday', dteday.weekday())],
        'workingday': [payload.get('workingday', 1)],
        'weathersit': [payload.get('weathersit', 1)],
        'temp': [payload.get('temp', 0.5)],
        'atemp': [payload.get('atemp', payload.get('temp', 0.5))],
        'hum': [payload.get('hum', 0.6)],
        'windspeed': [payload.get('windspeed', 0.2)]
    }
    
    return pd.DataFrame(data)

# ==============================
# API Routes
# ==============================
@app.route('/')
def index():
    """Serve the main web interface"""
    # Read the HTML file directly to get the latest changes
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for bike demand prediction"""
    if pipeline is None:
        return jsonify({
            "success": False, 
            "error": "Model pipeline not loaded. Check server logs."
        }), 500

    try:
        # Get JSON payload
        payload = request.get_json()
        if not payload:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        # Build input DataFrame
        input_df = build_input_dataframe(payload)
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        
        # Ensure non-negative integer result
        prediction = max(0, int(round(prediction)))
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "message": f"Predicted bike demand: {prediction} bikes"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "service": "RideWise Bike Demand Prediction API"
    })

# ==============================
# Development Server
# ==============================
if __name__ == '__main__':
    print("🚴‍♂️ Starting RideWise Bike Demand Prediction Server...")
    print("📊 Access the web interface at: http://127.0.0.1:5000")
    print("🔗 API endpoint: http://127.0.0.1:5000/predict")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Prevent double initialization
    )