
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load model with latest weights
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Detect ambulance-like vehicles
def detect_ambulance(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    print("🔍 Detected objects:")
    ambulance_found = False
    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()
        name = COCO_INSTANCE_CATEGORY_NAMES[class_id]

        print(f"- {name} ({class_id}): {score:.2f}")

        # Detect "ambulance-like" vehicles (truck, bus, car)
        if class_id in [6, 8, 3] and score > 0.6:
            ambulance_found = True
            box = boxes[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            print(f"🟩 Possible Ambulance ({name}) detected at {box} with score {score:.2f}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Ambulance-like ({name}): {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not ambulance_found:
        print("❌ No ambulance-like vehicle detected.")

    cv2.imshow("Ambulance Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# File picker GUI
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image")

if file_path:
    detect_ambulance(file_path)
else:
    print("⚠️ No file selected.")







Custom Classifier on Top
After detecting a "truck" or "car", crop the detected region and run a secondary classifier (custom CNN) trained specifically to distinguish ambulances from other vehicles.

2. Fine-Tune Faster R-CNN
Fine-tune the model on a custom dataset that includes ambulance as a separate class.

This means collecting or downloading a labeled dataset with images of ambulances.






2222222222222222222222222222222222222222222222222222222222222








import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load model with latest weights
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Detect vehicles with high confidence
def detect_ambulance(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    print("🔍 Detected objects:")
    ambulance_found = False
    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        # Check if class_id is within the valid range
        if class_id < len(COCO_INSTANCE_CATEGORY_NAMES):
            name = COCO_INSTANCE_CATEGORY_NAMES[class_id]
        else:
            name = f"Unknown (ID: {class_id})"
            print(f"⚠️ Warning: Invalid class ID {class_id} detected.")
        
        print(f"- {name} ({class_id}): {score:.2f}")

        # Refine detection: Only consider vehicles with high confidence
        # Ambulance-like vehicles typically have larger bounding boxes
        # Use the size of bounding boxes as a filter
        if class_id in [6, 8, 3, 4] and score > 0.85:  # Check for car, truck, bus, motorcycle
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            box_width = x2 - x1
            box_height = y2 - y1

            # Filter for large bounding boxes (vehicles are typically large)
            if box_width > 100 and box_height > 100:  # Arbitrary size thresholds
                ambulance_found = True
                print(f"🟩 Possible Ambulance-like ({name}) detected at {x1}, {y1}, {x2}, {y2} with score {score:.2f}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{name} ({score:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not ambulance_found:
        print("❌ No ambulance-like vehicle detected.")

    cv2.imshow("Ambulance Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# File picker GUI
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image")

if file_path:
    detect_ambulance(file_path)
else:
    print("⚠️ No file selected.")























??????????????????????????????????/
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load model with latest weights
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Detect ambulance-like vehicles
def detect_ambulance(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    print("🔍 Detected objects:")
    ambulance_found = False
    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id < len(COCO_INSTANCE_CATEGORY_NAMES):
            name = COCO_INSTANCE_CATEGORY_NAMES[class_id]
        else:
            name = f"Unknown (ID: {class_id})"
            print(f"⚠️ Warning: Invalid class ID {class_id} detected.")
        
        print(f"- {name} ({class_id}): {score:.2f}")

        # Only consider vehicles with high confidence
        if class_id in [3, 4, 6, 8] and score > 0.85:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > 100 and box_height > 100:
                ambulance_found = True

                # Heuristic substitution: treat truck or bus as ambulance
                if name in ["truck", "bus"]:
                    label_name = "Ambulance"
                else:
                    label_name = name

                print(f"🟩 Possible Ambulance-like ({label_name}) detected at {x1}, {y1}, {x2}, {y2} with score {score:.2f}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label_name} ({score:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not ambulance_found:
        print("❌ No ambulance-like vehicle detected.")

    cv2.imshow("Ambulance Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# File picker GUI
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image")

if file_path:
    detect_ambulance(file_path)
else:
    print("⚠️ No file selected.")
