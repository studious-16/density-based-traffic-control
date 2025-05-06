import os
import requests

# Create directory
folder_name = "yolo_files"
os.makedirs(folder_name, exist_ok=True)

# File URLs
files = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

# Set browser-like headers
headers = {'User-Agent': 'Mozilla/5.0'}

# Download files
for filename, url in files.items():
    print(f"Downloading {filename}...")
    filepath = os.path.join(folder_name, filename)

    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Saved {filename} to {filepath}")
        else:
            print(f"❌ Failed to download {filename}. HTTP Status: {r.status_code}")

print("\n🚀 All downloads attempted.")
