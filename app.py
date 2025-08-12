#import spoof
import os
import time
import pickle
import datetime 
import sys
import cv2
import face_recognition
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import nn

sys.path.append(os.path.join('SilentFaceAntiSpoofing'))
from SilentFaceAntiSpoofing.src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

def parse_model_name(model_name):
    info = model_name.split('_'); dim_part = [p for p in info if 'x' in p][0]; dim_index = info.index(dim_part)
    scale = float(info[dim_index - 1]); h, w = dim_part.split('x'); model_type = info[dim_index + 1].split('.')[0]
    return int(h), int(w), model_type, scale

def get_kernel(height, width):
    return ((height + 15) // 16, (width + 15) // 16)

class ToTensor(object):
    def __call__(self, img): 
        return torch.from_numpy(img.transpose((2, 0, 1))).float()

class Compose(object):
    def __init__(self, transforms): 
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms: 
            img = t(img)
        return img

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1, 
    'MiniFASNetV2': MiniFASNetV2, 
    'MiniFASNetV1SE': MiniFASNetV1SE, 
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class CropImage:
    def crop(self, org_img, bbox, scale, out_w, out_h):
        face_w, face_h = bbox[2], bbox[3]; x_center, y_center = bbox[0] + face_w / 2, bbox[1] + face_h / 2
        box_w, box_h = face_w * scale, face_h * scale; x1, y1 = x_center - box_w / 2, y_center - box_h / 2
        x1, y1, x2, y2 = map(int, [x1, y1, x1 + box_w, y1 + box_h]); h, w, _ = org_img.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cropped = org_img[y1:y2, x1:x2]
        if cropped.size == 0: return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        return cv2.resize(cropped, (out_w, out_h))

class Detection:
    def __init__(self):
        caffemodel = os.path.join('SilentFaceAntiSpoofing','resources','detection_model','Widerface-RetinaFace.caffemodel')
        deploy = os.path.join('SilentFaceAntiSpoofing','resources','detection_model','deploy.prototxt')
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6 
    def get_bbox(self, img):
        height, width, _ = img.shape; aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192*192:
            img_resized = cv2.resize(img, (int(192*math.sqrt(aspect_ratio)), int(192/math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
        else: img_resized = img
        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123)); self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        if out.ndim == 1 or len(out) == 0: return None
        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < self.detector_confidence: return None
        left,top,right,bottom = out[max_conf_index,3]*width, out[max_conf_index,4]*height, out[max_conf_index,5]*width, out[max_conf_index,6]*height
        return [int(left), int(top), int(right-left+1), int(bottom-top+1)]

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super().__init__()
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def _load_model(self, model_path):
        if model_path in self.models: 
            self.model = self.models[model_path]; return
        model_name = os.path.basename(model_path); h, w, model_type, scale = parse_model_name(model_name)
        kernel_size = get_kernel(h, w)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if next(iter(state_dict)).startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            self.model.load_state_dict(new_state_dict)
        else: 
            self.model.load_state_dict(state_dict)
        self.models[model_path] = self.model

    def predict(self, img, model_path):
        test_transform = Compose([ToTensor()]); img = test_transform(img).unsqueeze(0).to(self.device)
        self._load_model(model_path); self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img); result = F.softmax(result, dim=1).cpu().numpy()
        return result
        

ENCODINGS_PATH = 'encodings.pickle'
LOG_FILE = 'log.txt'
MODEL_DIR = os.path.join('SilentFaceAntiSpoofing', 'resources', 'anti_spoof_models')
DEVICE_ID = 0

print("Initializing models...")
anti_spoof_model = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()
print("Models initialized.")

if not os.path.exists(ENCODINGS_PATH):
    print(f"[ERROR] Encodings file not found at '{ENCODINGS_PATH}'.")
    encodeDict = None
else:
    print("[INFO] Loading encodings...")
    with open(ENCODINGS_PATH, "rb") as f:
        encodeDict = pickle.load(f)
    print(f"[INFO] Loaded {len(encodeDict)} known user encodings.")


def recognize_face(rgb_frame, known_encodings_dict):
    if known_encodings_dict is None:
        return "Unknown"
    boxes = face_recognition.face_locations(rgb_frame, model='hog')
    current_encodings = face_recognition.face_encodings(rgb_frame, boxes)
    if not current_encodings:
        return "Unknown" 
    face_enc_to_check = current_encodings[0]
    for name, known_encoding in known_encodings_dict.items():
        try:
            match = face_recognition.compare_faces([known_encoding], face_enc_to_check)
            if match[0]: 
                return name
        except Exception as e:
            continue
    return "Unknown"

def is_real_face(image_frame):
    image_bbox = anti_spoof_model.get_bbox(image_frame)
    if image_bbox is None: return 0, None
    prediction = np.zeros((1, 3))
    model_filenames = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.pth', '.onnx'))]
    for model_name in model_filenames:
        try: h, w, model_type, scale = parse_model_name(model_name)
        except Exception as e: 
            print(f"Warning: Could not parse model name '{model_name}'. Skipping."); continue
        param = { "org_img": image_frame, "bbox": image_bbox, "scale": scale, "out_w": w, "out_h": h }
        img = image_cropper.crop(**param)
        prediction += anti_spoof_model.predict(img, os.path.join(MODEL_DIR, model_name))
    if np.sum(prediction) == 0: return 0, image_bbox
    label = np.argmax(prediction)
    return label, image_bbox

def recognize_uploaded_image(image_path):
    if encodeDict is None:
        print("[ERROR] Encodings file not loaded.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label, bbox = is_real_face(img)
    if label != 1:
        print("[WARNING] Spoof attempt detected or no face found.")
        return
    name = recognize_face(rgb_img, encodeDict)
    print(f"[RESULT] Recognized as: {name}")

def start_live_check():
    if encodeDict is None:
        print("[ERROR] Cannot start live check, encodings not loaded.")
        return
    video_capture = cv2.VideoCapture()
    if not video_capture.isOpened():
        print("[ERROR] Could not open video stream.")
        return
    print("[INFO] Starting live camera feed. Press 'q' to quit.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        live_label, bbox = is_real_face(frame)
        name = "Unknown"
        text_color = (0, 0, 255)
        if live_label == 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            name = recognize_face(rgb_frame, encodeDict)
            if name != "Unknown":
                text_color = (0, 255, 0)
        status = "Spoof" if live_label != 1 else name
        if bbox is not None:
            x, y, w, h = bbox
            box_color = (0, 255, 0) if live_label == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            y_text = y - 10 if y - 10 > 10 else y + h + 25
            cv2.putText(frame, status, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    print("Select an operation:")
    print("1: Recognize faces")
    print("2: Add a new face")
    print("3: Delete a face encoding")
    choice = input("Enter the number of the operation you want to perform: ")
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    prev_frame_time = 0
    new_frame_time = 0
    if choice == '1':
        print("Face recognition mode selected.")
    elif choice == '2':
        print("Add face mode selected.")
        name = input("Enter the name of the person to add: ")
        while True:
            success, img = cap.read()
            add_face_encoding_indi(name, img)
            time.sleep(2)
            break
    elif choice == '3':
        print("Delete face mode selected.")
        name = input("Enter the name of the person to delete: ")
        delete_face_encoding(name)
    elif choice == '4':
        start_live_check()
    elif choice == '5':
        start_live_check()
    print("Invalid option selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

