"""
Download the bike sharing dataset used in the ridewise notebook.
This downloads the actual UCI Machine Learning Repository dataset.
"""

import urllib.request
import zipfile
import os
import pandas as pd

def download_bike_sharing_dataset():
    """Download and extract the bike sharing dataset"""
    
    print("🔄 Downloading bike sharing dataset from UCI ML Repository...")
    
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    zip_filename = "Bike-Sharing-Dataset.zip"
    
    try:
        # Download the dataset
        urllib.request.urlretrieve(url, zip_filename)
        print("✅ Dataset downloaded successfully!")
        
        # Extract the zip file
        print("🔄 Extracting files...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Check if hour.csv exists
        if os.path.exists('hour.csv'):
            print("✅ hour.csv extracted successfully!")
            
            # Load and show basic info
            df = pd.read_csv('hour.csv')
            print(f"\n📊 Dataset Information:")
            print(f"   - Shape: {df.shape}")
            print(f"   - Columns: {list(df.columns)}")
            print(f"   - Date range: {df['dteday'].min()} to {df['dteday'].max()}")
            print(f"   - Total bike rentals: {df['cnt'].min()} to {df['cnt'].max()}")
            print(f"   - Mean daily rentals: {df['cnt'].mean():.1f}")
            
            print(f"\n✅ Ready to train models with real data!")
            
        else:
            print("❌ hour.csv not found in extracted files")
            
        # Clean up zip file
        os.remove(zip_filename)
        print("🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 You can manually download from:")
        print("   https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        print("   Extract and place hour.csv in the current directory")

if __name__ == "__main__":
    download_bike_sharing_dataset()