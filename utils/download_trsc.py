import requests
from pathlib import Path
import zipfile
import os

# Define output paths
output_dir = Path("data")
zip_path = output_dir / "turkish-speech-command-dataset.zip"
extract_dir = output_dir / "turkish-speech-command-dataset"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Kaggle dataset URL (requires authentication)
url = "https://www.kaggle.com/api/v1/datasets/download/muratkurtkaya/turkish-speech-command-dataset"

# Download the file
response = requests.get(url, stream=True)
response.raise_for_status()

# Save zip file
with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print(f"Downloaded ZIP to: {zip_path}")

# Unzip the archive
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted to: {extract_dir}")

# Remove the ZIP file
os.remove(zip_path)
print(f"Removed ZIP file: {zip_path}")