# new.py  (patched)
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

# threading + helpers
import threading, queue
from collections import deque, Counter

# --- CameraReader & majority_vote (background reader + debouncing) ---
class CameraReader:
    """Background camera reader that keeps only the latest frames (no backlog)."""
    def __init__(self, src=0, queue_size=2):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            # keep only latest frames (drop oldest) to prevent backlog
            try:
                if self.q.full():
                    _ = self.q.get_nowait()
                self.q.put_nowait((time.time(), frame))
            except queue.Full:
                pass

    def read(self):
        try:
            return self.q.get_nowait()  # (timestamp, frame)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        # join briefly; thread is daemon so program will exit anyway
        try:
            self.thread.join(timeout=0.5)
        except:
            pass
        try:
            self.cap.release()
        except:
            pass

def majority_vote(history_deque, min_count=3):
    """
    Return most frequent name in history_deque if it appears >= min_count, else "Unknown".
    history_deque holds names or None.
    """
    if not history_deque:
        return "Unknown"
    c = Counter(history_deque)
    most, cnt = c.most_common(1)[0]
    if most is None:
        return "Unknown"
    return most if cnt >= min_count else "Unknown"
# -------------------------------------------------------------------

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


def recognize_face(rgb_frame, known_encodings_dict, tolerance=0.45):
    """Original, simple compare_faces-based recognizer (kept for backward compatibility)."""
    if known_encodings_dict is None:
        return "Unknown"

    boxes = face_recognition.face_locations(rgb_frame, model='hog')
    current_encodings = face_recognition.face_encodings(rgb_frame, boxes)

    if not current_encodings:
        return "Unknown"

    face_enc_to_check = current_encodings[0]

    for name, enc_list in known_encodings_dict.items():
        # Ensure enc_list is iterable (some may be single array, some list)
        if not isinstance(enc_list, (list, tuple, np.ndarray)):
            enc_list = [enc_list]

        # Compare against all known encodings for this person
        matches = face_recognition.compare_faces(enc_list, face_enc_to_check, tolerance=tolerance)
        if True in matches:
            return name

    return "Unknown"


def recognize_face_strict(rgb_frame, known_encodings_dict, tolerance=0.48):
    """
    Distance-aware recognizer. Returns (name, min_distance).
    Use this in the main loop for more robust decisions.
    """
    if known_encodings_dict is None:
        return "Unknown", None

    boxes = face_recognition.face_locations(rgb_frame, model='hog')
    encs = face_recognition.face_encodings(rgb_frame, boxes, num_jitters=1)
    if not encs:
        return "Unknown", None

    query = encs[0]
    best_name = "Unknown"
    best_dist = float('inf')

    for name, enc_list in known_encodings_dict.items():
        if not isinstance(enc_list, (list, tuple, np.ndarray)):
            enc_list = [enc_list]
        encs_arr = np.asarray(enc_list)
        try:
            dists = face_recognition.face_distance(encs_arr, query)
            min_d = float(np.min(dists))
            if min_d < best_dist:
                best_dist = min_d
                best_name = name
        except Exception:
            continue

    if best_dist <= tolerance:
        return best_name, best_dist
    return "Unknown", best_dist


# ---------- Replaced start_live_check with stabilized threaded approach ----------
def start_live_check(camera_indices=(0,1,2,3,4,5), process_every_n=6, queue_size=2,
                     fps_window_seconds=1.0, history_len=7, required_consensus=3,
                     tolerance=0.48, consec_required=3, ema_alpha=0.3):
    """
    Threaded camera reader + periodic processing + debounced identity confirmation.

    Tuning params:
     - process_every_n: how often to run heavy models (every N frames)
     - queue_size: CameraReader queue size (1 = lowest latency)
     - fps_window_seconds: sliding window for FPS smoothing
     - history_len & required_consensus: majority voting parameters
     - tolerance: face distance threshold for matching
     - consec_required: number of consecutive low-distance frames required to accept quickly
     - ema_alpha: smoothing factor for distance EMA (0..1)
    """
    if encodeDict is None:
        print("[ERROR] Cannot start live check, encodings not loaded.")
        return

    # try to open a working camera index using CameraReader
    reader = None
    opened_idx = None
    for idx in camera_indices:
        cand = CameraReader(idx, queue_size=queue_size)
        if cand.cap is not None and cand.cap.isOpened():
            reader = cand.start()
            opened_idx = idx
            break
        else:
            try:
                cand.cap.release()
            except:
                pass

    if reader is None:
        print(f"[ERROR] Could not open any camera from indices {camera_indices}.")
        print(" -> Run 'ls /dev/video*' or 'v4l2-ctl --list-devices' to inspect devices.")
        return

    print(f"[INFO] Starting live camera feed on index {opened_idx}. Press 'q' to quit.")
    fps_timestamps = deque()
    frame_counter = 0

    # state for debouncing / smoothing
    identity_history = deque(maxlen=history_len)
    last_name = None
    last_bbox = None
    last_live_label = 0
    consec_count = 0
    dist_ema = None

    try:
        while True:
            item = reader.read()
            if item is None:
                # no frame available right now; avoid busy-loop
                time.sleep(0.005)
                continue

            frame_ts, frame = item
            frame_counter += 1
            do_process = (frame_counter % process_every_n) == 0

            name = "Unknown"
            dist = None

            if do_process:
                try:
                    live_label, bbox = is_real_face(frame)
                    last_live_label = live_label
                    last_bbox = bbox
                    if live_label == 1 and bbox is not None:
                        x, y, w, h = bbox
                        x, y = max(0, x), max(0, y)
                        roi = frame[y:y+h, x:x+w]
                        if roi.size != 0:
                            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            name, dist = recognize_face_strict(rgb_roi, encodeDict, tolerance=tolerance)
                        else:
                            name, dist = "Unknown", None
                    else:
                        name, dist = "Unknown", None
                except Exception as e:
                    print(f"[ERROR] Exception during processing: {e}")
                    name, dist = "Unknown", None
            else:
                # not processing this frame - preserve previous last_name (do not expand history)
                pass

            # update smoothing and histories only if we processed in this iteration
            if do_process:
                identity_history.append(name if name != "Unknown" else None)

                # update consecutive count logic
                if name != "Unknown" and dist is not None and dist <= tolerance:
                    if last_name == name:
                        consec_count += 1
                    else:
                        consec_count = 1
                    last_name = name
                else:
                    consec_count = 0
                    if name != "Unknown":
                        last_name = name
                    else:
                        last_name = None

                # EMA for distance smoothing
                if dist is not None:
                    if dist_ema is None:
                        dist_ema = dist
                    else:
                        dist_ema = ema_alpha * dist + (1.0 - ema_alpha) * dist_ema

            # Confirm identity via majority vote
            confirmed_name = majority_vote(identity_history, min_count=required_consensus)

            # Accept quick if we have consecutive low-distance frames
            if consec_count >= consec_required and last_name is not None:
                confirmed_name = last_name

            # Draw bbox/label: prefer last_bbox; if none, show "No face detected"
            if last_bbox is not None:
                x, y, w, h = last_bbox
                box_color = (0, 255, 0) if last_live_label == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                # label selection: show confirmed_name, and mark Spoof if spoof detected
                if confirmed_name != "Unknown":
                    label_text = f"{confirmed_name}"
                else:
                    label_text = "Spoof" if last_live_label != 1 else "Unknown"
                y_text = y - 10 if y - 10 > 10 else y + h + 25
                text_color = (0,255,0) if confirmed_name != "Unknown" else ((0,0,255) if last_live_label != 1 else (0,255,255))
                cv2.putText(frame, label_text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
                # optionally show distance EMA for debugging
                if dist_ema is not None:
                    cv2.putText(frame, f"dEMA:{dist_ema:.2f}", (x, y_text + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

            # Stable FPS using sliding timestamp window
            now = time.time()
            fps_timestamps.append(now)
            while fps_timestamps and (now - fps_timestamps[0]) > fps_window_seconds:
                fps_timestamps.popleft()
            fps_display = len(fps_timestamps) / fps_window_seconds
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow('Live Anti-Spoof', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()

def capture_once():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        return

    print("[INFO] Camera started. Looking for face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        live_label, bbox = is_real_face(frame)

        if bbox is not None:  # Face detected
            print("[INFO] Face detected, processing...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            name = recognize_face_strict(rgb_frame, encodeDict)

            if live_label == 1:
                print(f"[RESULT] Real face detected: {name}")
            else:
                print("[RESULT] Spoof detected!")

            # Save snapshot if you want
            cv2.imwrite("last_snapshot.jpg", frame)

            break  # ✅ Exit loop after detection

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit manually
            break

    cap.release()
    cv2.destroyAllWindows()


    
# --------------------------------------------------------------------


def main():
    # Directly start the stabilized live anti-spoof + recognition.
    # You can tune parameters here if needed.
    print("Press ENTER to start detection...")
    input()  # Wait for user "button click" (keyboard enter)
    capture_once()
    start_live_check(
        camera_indices=(0, 1, 2, 3, 4, 5),  # camera priorities to try
        process_every_n=6,                  # run heavy models every N frames
        queue_size=2,                       # CameraReader queue size (1 = lowest latency)
        fps_window_seconds=1.0,             # FPS smoothing window (seconds)
        history_len=7,                      # number of recent predictions to keep
        required_consensus=3,               # majority-vote threshold
        tolerance=0.48,                     # face distance matching threshold
        consec_required=3,                  # consecutive low-distance frames to accept quickly
        ema_alpha=0.3                       # smoothing factor for distance EMA
    )
    # start_live_check cleans up; keep this call blocking until user quits.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
