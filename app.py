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
from collections import OrderedDict 
from threading import Thread, Lock 
from queue import Queue 
import concurrent.futures 
import warnings
import sys
import threading

stop_live_flag = threading.Event()
last_detection = {"label": 0, "bbox": None, "name": "Unknown", "time": None}

sys.path.append(os.path.join('SilentFaceAntiSpoofing'))
from SilentFaceAntiSpoofing.src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

def parse_model_name(model_name):
    info = model_name.split('_')
    dim_part = [p for p in info if 'x' in p][0]
    dim_index = info.index(dim_part)
    scale = float(info[dim_index - 1])
    h, w = dim_part.split('x')
    model_type = info[dim_index + 1].split('.')[0]
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
        face_w, face_h = bbox[2], bbox[3]
        x_center, y_center = bbox[0] + face_w / 2, bbox[1] + face_h / 2
        box_w, box_h = face_w * scale, face_h * scale
        x1, y1 = x_center - box_w / 2, y_center - box_h / 2
        x1, y1, x2, y2 = map(int, [x1, y1, x1 + box_w, y1 + box_h])
        h, w, _ = org_img.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cropped = org_img[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        return cv2.resize(cropped, (out_w, out_h))

class Detection:
    def __init__(self):
        caffemodel = os.path.join('SilentFaceAntiSpoofing', 'resources', 'detection_model', 'Widerface-RetinaFace.caffemodel')
        deploy = os.path.join('SilentFaceAntiSpoofing', 'resources', 'detection_model', 'deploy.prototxt')
        
        if not os.path.exists(caffemodel) or not os.path.exists(deploy): 
            raise FileNotFoundError; warnings.warn("[WARNING] Detection model files not found")
        
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6 
        self.lock = Lock()
    
    def get_bbox(self, img):
        height, width, _ = img.shape
        aspect_ratio = width / height
        
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(
                img, 
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))), 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            img_resized = img
        
        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
        
        with self.lock:  # Thread-safe inference
            self.detector.setInput(blob, 'data')
            out = self.detector.forward('detection_out').squeeze()
        
        if out.ndim == 1 or len(out) == 0:
            return None
        
        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < self.detector_confidence:
            return None
        
        left = out[max_conf_index, 3] * width
        top = out[max_conf_index, 4] * height
        right = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height
        
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super().__init__()
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_lock = Lock() 
        print(f"[INFO] Using device: {self.device}")

    def _load_model(self, model_path):
        with self.model_lock:
            if model_path in self.models: 
                return self.models[model_path]
            
            model_name = os.path.basename(model_path)
            h, w, model_type, scale = parse_model_name(model_name)
            kernel_size = get_kernel(h, w)
            
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            
            if next(iter(state_dict)).startswith('module.'):
                new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
                model.load_state_dict(new_state_dict)
            else: 
                model.load_state_dict(state_dict)
            
            model.eval()
            self.models[model_path] = model
            return model

    def predict_single(self, img, model_path):
        test_transform = Compose([ToTensor()])
        img_tensor = test_transform(img).unsqueeze(0).to(self.device)
        model = self._load_model(model_path)
        
        with torch.no_grad():
            result = model.forward(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

ENCODINGS_PATH = 'encodings.pickle'
LOG_FILE = 'log.txt'
MODEL_DIR = os.path.join('SilentFaceAntiSpoofing', 'resources', 'anti_spoof_models')
DEVICE_ID = 0

print("[INFO] Initializing models...")
try:
    anti_spoof_model = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()
    print("[INFO] models initialized successfully ")
except Exception as e:
    print(f"[ERROR] Failed to initialize models: {e}", file=sys.stderr)
    sys.exit(1)

if not os.path.exists(ENCODINGS_PATH):
    warnings.warn(f"Encodings file not found at '{ENCODINGS_PATH}'.")
    print("[INFO] You'll need to add faces before recognition can work.")
    encodeDict = {}
else:
    print("[INFO] Loading encodings...")
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            encodeDict = pickle.load(f)
        print(f"[INFO] Loaded {len(encodeDict)} known user encodings.")
    except Exception as e:
        print(f"[ERROR] Failed to load encodings: {e}", file=sys.stderr)
        encodeDict = {}

known_names = list(encodeDict.keys())
known_encodings = list(encodeDict.values())

def save_encodings():
    global known_names, known_encodings
    try:
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(encodeDict, f)
        known_names = list(encodeDict.keys())
        known_encodings = list(encodeDict.values())
        print(f"[INFO] Encodings saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {e}", file=sys.stderr)

def log_entry(name, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now()
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {name}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to log: {e}", file=sys.stderr)

def recognize_face_fast(rgb_frame, tolerance=0.6):
    if not known_encodings:
        return "Unknown"
    
    try:
        boxes = face_recognition.face_locations(rgb_frame, model='cnn' if torch.cuda.is_available() else 'hog', number_of_times_to_upsample=1)
        
        if not boxes:
            return "Unknown"
        
        current_encodings = face_recognition.face_encodings(rgb_frame, boxes, num_jitters=1)
        
        if not current_encodings:
            return "Unknown"
        
        face_enc_to_check = current_encodings[0]
        
        matches = face_recognition.compare_faces(known_encodings, face_enc_to_check, tolerance=tolerance)
        
        if True in matches:
            first_match_index = matches.index(True)
            return known_names[first_match_index]
        
        return "Unknown"
    except Exception as e:
        print(f"[ERROR] Error in face recognition: {e}", file=sys.stderr)
        return "Unknown"

def is_real_face_parallel(image_frame):
    try:
        image_bbox = anti_spoof_model.get_bbox(image_frame)
        if image_bbox is None:
            return 0, None
        
        model_filenames = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.pth', '.onnx'))]
        
        if not model_filenames:
            warnings.warn(f"[WARNING] No model files found in {MODEL_DIR}")
            return 0, image_bbox
    
        crops_and_models = []
        for model_name in model_filenames:
            try:
                h, w, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image_frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w,
                    "out_h": h
                }
                img = image_cropper.crop(**param)
                crops_and_models.append((img, os.path.join(MODEL_DIR, model_name)))
            except Exception as e:
                warnings.warn(f"[WARNING] Could not parse model name '{model_name}': {e}")
                continue
        
        predictions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(crops_and_models))) as executor:
            futures = [executor.submit(anti_spoof_model.predict_single, img, model_path) 
                      for img, model_path in crops_and_models]
            for future in concurrent.futures.as_completed(futures):
                predictions.append(future.result())
        
        if not predictions:
            return 0, image_bbox
        
        prediction = np.sum(predictions, axis=0)
        label = np.argmax(prediction)
        return label, image_bbox
        
    except Exception as e:
        print(f"[ERROR] Error in spoof detection: {e}",file=sys.stderr)
        return 0, None


class FrameProcessor(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.frame_queue = Queue(maxsize=2) 
        self.result_queue = Queue(maxsize=2)
        self.running = True
        
    def run(self):
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    live_label, bbox = is_real_face_parallel(frame)
                    name = "Unknown"
                    
                    if live_label == 1:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        name = recognize_face_fast(rgb_frame)
                    
                    self.result_queue.put((live_label, bbox, name))
                else:
                    time.sleep(0.01)  
            except Exception as e:
                print(f"[ERROR] Frame processing error: {e}",file=sys.stderr)
    
    def stop(self):
        self.running = False

def add_face_encoding_indi(name, img):
    global encodeDict
    
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label, bbox = is_real_face_parallel(img)
        if label != 1:
            warnings.warn("[WARNING] Spoof detected or no face found. Cannot add encoding.")
            return False
        
        boxes = face_recognition.face_locations(rgb_img, model='hog')
        encodings = face_recognition.face_encodings(rgb_img, boxes, num_jitters=1)
        
        if not encodings:
            warnings.warn("[WARNING] No face encoding found in image.")
            return False
        
        encodeDict[name] = encodings[0]
        save_encodings()
        print(f"[INFO] Face encoding added for {name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to add face encoding: {e}",file=sys.stderr)
        return False

def delete_face_encoding(name):
    """Delete a face encoding by name."""
    global encodeDict
    
    if name in encodeDict:
        del encodeDict[name]
        save_encodings()
        print(f"[INFO] Deleted encoding for {name}")
    else:
        warnings.warn(f"[WARNING] No encoding found for {name}")
        print(f"[INFO] Available names: {', '.join(encodeDict.keys())}")

def recognize_uploaded_image(image_path):
    """Recognize face from uploaded image file."""
    if not encodeDict:
        print("[ERROR] No encodings loaded. Please add faces first.", file=sys.stderr)
        return
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}", file=sys.stderr)
        return
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}", file=sys.stderr)
            return
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label, bbox = is_real_face_parallel(img)
        
        if label != 1:
            warnings.warn("[WARNING] Spoof attempt detected or no face found.")
            return
        
        name = recognize_face_fast(rgb_img)
        print(f"[INFO] Recognized as: {name}")
        
        if name != "Unknown":
            log_entry(name)
            
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}", file=sys.stderr)


def start_live_check(show_window=True, shared_last_detection=None, stop_flag=None):

    import warnings, datetime, cv2

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not open video stream.", file=sys.stderr)
        return

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    processor = FrameProcessor()
    processor.start()
    current_result = (0, None, "Unknown")

    try:
        while not stop_flag.is_set():
            ret, frame = video_capture.read()
            if not ret:
                continue

            if processor.frame_queue.qsize() < 2:
                processor.frame_queue.put(frame.copy())

            if not processor.result_queue.empty():
                current_result = processor.result_queue.get()

            live_label, bbox, name = current_result

            if shared_last_detection is not None:
                shared_last_detection.update({
                    "label": int(live_label),
                    "bbox": bbox,
                    "name": name if live_label == 1 else "Spoof",
                    "time": datetime.datetime.now().isoformat()
                })

            if show_window:
                display_text = shared_last_detection.get("name", "Unknown")
                color = (0, 255, 0) if live_label == 1 else (0, 0, 255)

                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    y_text = y - 10 if y - 10 > 10 else y + h + 25
                    cv2.putText(frame, display_text, (x, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag.set()
                    break

    finally:
        processor.stop()
        processor.join(timeout=2)
        video_capture.release()
        if show_window:
            cv2.destroyAllWindows()
        print("[INFO] Camera released")


def get_last_detection():

    global last_detection
    return last_detection


def list_faces():
    if not encodeDict:
        warnings.warn("[WARNING] No faces stored yet.")
    else:
        print(f"\n[INFO] Stored faces ({len(encodeDict)}):")
        for i, name in enumerate(encodeDict.keys(), 1):
            print(f"  {i}. {name}")

def main():
    """"
    while True:
        print("\n" + "="*50)
        print("FACE RECOGNITION SYSTEM (OPTIMIZED)")
        print("="*50)
        print("1: Recognize faces (live camera)")
        print("2: Add a new face")
        print("3: Delete a face encoding")
        print("4: Recognize from uploaded image")
        print("5: List all stored faces")
        print("6: Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n[INFO] Starting face recognition...")
            start_live_check()
        
        elif choice == '2':
            print("\n[INFO] Add face mode")
            name = input("Enter the name of the person: ").strip()
            
            if not name:
                print("[ERROR] Name cannot be empty.")
                continue
            
            if name in encodeDict:
                response = input(f"[WARNING] {name} already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    continue
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Could not open camera.")
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("[INFO] Position face in frame. Capturing in 3 seconds...")
            
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, f"Capturing in {i}...", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Add Face', frame)
                    cv2.waitKey(1000)
            
            ret, img = cap.read()
            if ret:
                add_face_encoding_indi(name, img)
            else:
                print("[ERROR] Failed to capture image.")
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            print("\n[INFO] Delete face mode")
            list_faces()
            name = input("Enter the name to delete: ").strip()
            delete_face_encoding(name)
        
        elif choice == '4':
            print("\n[INFO] Recognize from image file")
            image_path = input("Enter the path to the image: ").strip()
            recognize_uploaded_image(image_path)
        
        elif choice == '5':
            list_faces()
        
        elif choice == '6':
            print("\n[INFO] Exiting...")
            break
        
        else:
            print("\n[ERROR] Invalid option. Please enter 1-6.")
            """

if __name__ == "__main__":
    main()