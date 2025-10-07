# Test script for RideWise API
import requests
import json

def test_api():
    """Test the RideWise prediction API"""
    
    # API endpoint
    url = "http://127.0.0.1:5000/predict"
    
    # Test data - typical summer weekday morning (matching real dataset format)
    test_data = {
        "dteday": "2012-06-15",  # Use 2012 (within dataset range)
        "hr": 8,                 # Morning rush hour
        "season": 2,             # Summer
        "weathersit": 1,         # Clear weather
        "temp": 0.7,             # Warm temperature
        "atemp": 0.7,            # Feels like temperature
        "hum": 0.6,              # Moderate humidity
        "windspeed": 0.2,        # Light wind
        "holiday": 0,            # Not a holiday
        "workingday": 0,         # Saturday (not working day)
        "weekday": 5             # Saturday
    }
    
    try:
        # Make API request
        print("🔄 Testing RideWise API...")
        response = requests.post(url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                prediction = result.get('prediction')
                print(f"✅ API Test Successful!")
                print(f"📊 Predicted bike demand: {prediction} bikes")
                print(f"📝 Message: {result.get('message', 'N/A')}")
            else:
                print(f"❌ API returned error: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure Flask server is running on port 5000")
    except requests.exceptions.Timeout:
        print("❌ Request timeout - server may be slow")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"🏥 Health Check: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🚴‍♂️ RideWise API Testing")
    print("=" * 40)
    
    # Test health endpoint
    test_health_endpoint()
    print()
    
    # Test prediction endpoint
    test_api()
    
    print("\n" + "=" * 40)
    print("🌐 You can also test the web interface at: http://127.0.0.1:5000")