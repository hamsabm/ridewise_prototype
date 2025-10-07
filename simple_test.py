import requests
import json

# Test with minimal data
test_data = {
    "dteday": "2012-06-15",
    "hr": 8,
    "season": 2,
    "weathersit": 1,
    "temp": 0.7,
    "hum": 0.6,
    "windspeed": 0.2,
    "holiday": 0,
    "workingday": 1
}

try:
    response = requests.post("http://127.0.0.1:5000/predict", json=test_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")