import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

MODELS = {
    "face_landmark.tflite": "https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite"
}

ASSETS_DIR = os.path.join("assets", "models")

def main():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        
    for filename, url in MODELS.items():
        filepath = os.path.join(ASSETS_DIR, filename)
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"Successfully downloaded {filename} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
