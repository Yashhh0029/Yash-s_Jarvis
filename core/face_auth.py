# core/face_auth.py
"""
Simple & Accurate face authentication using OpenCV LBPH + Haar cascade.
Usage:
    python -m core.face_auth enroll --name yash config/face_data/yash_reference.jpg
    python -m core.face_auth train
    python -m core.face_auth verify_image --name yash config/face_data/yash_reference.jpg
    python -m core.face_auth live_verify --name yash
"""

from __future__ import annotations
import os
import sys
import pathlib
import json
import time
import shutil
from typing import Tuple, List, Optional

# OpenCV + numpy (required)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception as e:
    print("⚠️ OpenCV not available:", e)
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# -------------------- Paths & config --------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # project root (one level up from core/)
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

LBPH_MODEL_FILE = MODEL_DIR / "lbph_model.yml"
LABELS_JSON = MODEL_DIR / "labels.json"
CASCADE_FILE = MODEL_DIR / "haarcascade_frontalface_default.xml"

# Standard face size
STANDARD_FACE_SIZE = (200, 200)

# default LBPH threshold (lower = stricter). Tweak if necessary.
DEFAULT_THRESHOLD = 70.0

# -------------------- Helpers --------------------
def normalize_path(p: str) -> str:
    if not p:
        return p
    return os.path.normpath(p)

def save_json(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: pathlib.Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Ensure cascade --------------------
def ensure_cascade() -> bool:
    """
    Make sure there's a local copy of haarcascade in MODEL_DIR.
    Tries to copy from cv2.data.haarcascades if available.
    """
    try:
        if CASCADE_FILE.exists():
            return True
        if not OPENCV_AVAILABLE:
            return False
        cascade_src = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_src):
            shutil.copyfile(cascade_src, str(CASCADE_FILE))
            return True
    except Exception:
        pass
    return False

# -------------------- Image reading --------------------
def read_image(path: str):
    path = normalize_path(path)
    if not path:
        return None
    # handle unicode / Windows long paths by using numpy + imdecode
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        if arr is not None and arr.size:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    # fallback
    try:
        return cv2.imread(path)
    except Exception:
        return None

def to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------- Face detection & preprocessing --------------------
def detect_faces_in_image(img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) -> List[Tuple[int,int,int,int]]:
    if not OPENCV_AVAILABLE:
        return []
    if not ensure_cascade():
        return []
    gray = to_gray(img)
    if gray is None:
        return []
    cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
    faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    try:
        return faces.tolist() if hasattr(faces, "tolist") else list(faces)
    except Exception:
        return list(faces)

def crop_face(img, box, margin=0.25):
    if img is None:
        return None
    x, y, w, h = box
    h_img, w_img = img.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    return img[y1:y2, x1:x2]

def prepare_face_for_training(img):
    """
    Return grayscale (STANDARD_FACE_SIZE) face or None.
    Uses largest detected face; falls back to center crop if no face found.
    """
    if img is None:
        return None
    boxes = detect_faces_in_image(img, scaleFactor=1.22, minNeighbors=7, minSize=(120,120))
    if boxes:
        boxes_sorted = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        face_crop = crop_face(img, boxes_sorted[0], margin=0.25)
    else:
        # center crop fallback
        h, w = img.shape[:2]
        side = min(h, w)
        cx = w // 2
        cy = h // 2
        x1 = max(0, cx - side//2)
        y1 = max(0, cy - side//2)
        face_crop = img[y1:y1+side, x1:x1+side]

    if face_crop is None:
        return None

    gray = to_gray(face_crop)
    if gray is None:
        return None
    try:
        resized = cv2.resize(gray, STANDARD_FACE_SIZE, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return None

# -------------------- Dataset builder --------------------
def load_face_dataset():
    """
    Reads data/faces/<person>/*(.jpg|.png) and returns:
        images: list of numpy arrays (grayscale)
        labels: list of ints
        label_names: list mapping index -> person name
    """
    images = []
    labels = []
    label_names = []
    if not FACES_DIR.exists():
        return images, labels, label_names
    people = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])
    for idx, person_dir in enumerate(people):
        person_name = person_dir.name
        label_names.append(person_name)
        # accept jpg/png
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_file in sorted(person_dir.glob(pattern)):
                img = read_image(str(img_file))
                face = prepare_face_for_training(img)
                if face is not None:
                    images.append(face)
                    labels.append(idx)
    return images, labels, label_names

# -------------------- Train LBPH --------------------
def train_lbph_model() -> bool:
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        print("❌ Cannot train — OpenCV or numpy missing.")
        return False

    images, labels, label_names = load_face_dataset()
    if len(images) == 0:
        print("❌ No face images available in data/faces/. Enroll first.")
        return False

    # ensure face module exists (opencv-contrib)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("❌ LBPH recognizer unavailable. Ensure 'opencv-contrib-python' is installed.", e)
        return False

    try:
        recognizer.train(images, np.array(labels))
        recognizer.write(str(LBPH_MODEL_FILE))
        mapping = {str(i): name for i, name in enumerate(label_names)}
        save_json(LABELS_JSON, mapping)
        print("✓ LBPH model trained and saved.")
        return True
    except Exception as e:
        print("❌ Training failed:", e)
        return False

# -------------------- Load LBPH --------------------
def load_lbph():
    if not LBPH_MODEL_FILE.exists():
        return None, None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(LBPH_MODEL_FILE))
    except Exception as e:
        print("❌ Could not load LBPH model:", e)
        return None, None
    labels = load_json(LABELS_JSON, {})
    return recognizer, labels

# -------------------- Enrollment --------------------
def enroll_image(name: str, image_path: str) -> bool:
    """
    Enrolls face by saving BOTH:
    - raw image (full resolution)
    - processed 200x200 cropped face image
    """
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available.")
        return False

    image_path = normalize_path(image_path)

    if not os.path.exists(image_path):
        print(f"❌ enroll_image: file not found: {image_path}")
        return False

    img = read_image(image_path)
    if img is None:
        print("❌ Could not read image.")
        return False

    face = prepare_face_for_training(img)
    if face is None:
        print("❌ No detectable face — try a clearer image.")
        return False

    save_dir = FACES_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    # Save RAW original image (helps model learn real variations)
    raw_path = save_dir / f"{name}_RAW_{timestamp}.jpg"
    cv2.imwrite(str(raw_path), img)

    # Save PROCESSED cropped face for LBPH training
    proc_path = save_dir / f"{name}_PROC_{timestamp}.jpg"
    cv2.imwrite(str(proc_path), face)

    print(f"✓ Enrolled RAW: {raw_path}")
    print(f"✓ Enrolled PROCESSED: {proc_path}")

    return True



# -------------------- Verify static image --------------------
def verify_face_image(name: str, img_path: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    """
    Returns (ok, confidence). Lower confidence = better match for LBPH.
    """
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available.")
        return False, 0.0
    recognizer, labels = load_lbph()
    if recognizer is None:
        print("❌ LBPH model not loaded. run: python -m core.face_auth train")
        return False, 0.0
    img = read_image(img_path)
    if img is None:
        print("❌ Could not read image:", img_path)
        return False, 0.0
    face = prepare_face_for_training(img)
    if face is None:
        print("❌ Could not detect face in image.")
        return False, 0.0
    try:
        predicted_label, confidence = recognizer.predict(face)
    except Exception as e:
        print("❌ Prediction failed:", e)
        return False, 0.0
    # check label mapping
    if str(predicted_label) in labels and labels[str(predicted_label)] == name:
        ok = confidence <= threshold
        return ok, float(confidence)
    return False, float(confidence)

# -------------------- Live verification (webcam) --------------------
def live_verify(name: str, attempts: int = 20, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available — live verify disabled.")
        return False, 0.0
    recognizer, labels = load_lbph()
    if recognizer is None:
        print("❌ LBPH model not loaded.")
        return False, 0.0
    # find target label
    target_label = None
    for k, v in (labels or {}).items():
        if v == name:
            try:
                target_label = int(k)
            except Exception:
                continue
            break
    if target_label is None:
        print(f"❌ No trained images found for {name}")
        return False, 0.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return False, 0.0
    print("📷 Live verification started. Look at the camera...")
    best_conf = 9999.0
    try:
        for i in range(attempts):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ensure cascade exists
            if not ensure_cascade():
                print("❌ cascade not found.")
                break
            cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
            if len(faces) == 0:
                # no face, continue
                time.sleep(0.05)
                continue
            # largest face
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, STANDARD_FACE_SIZE)
            except Exception:
                continue
            try:
                predicted, confidence = recognizer.predict(face)
            except Exception as e:
                print("❌ predict error:", e)
                continue
            best_conf = min(best_conf, float(confidence))
            if predicted == target_label and confidence <= threshold:
                print(f"✓ VERIFIED: {name} (confidence={confidence:.2f})")
                cap.release()
                return True, float(confidence)
            # small sleep
            time.sleep(0.06)
    except KeyboardInterrupt:
        print("\n❌ Verification aborted by user.")
    finally:
        cap.release()
    print(f"FAILED — best confidence={best_conf:.2f}")
    return False, best_conf

# -------------------- CLI glue --------------------
def enroll_image_cli(name: str, image: str) -> bool:
    return enroll_image(name, image)

def train_lbph() -> bool:
    return train_lbph_model()

def verify_image_cli(name: str, image: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    return verify_face_image(name, image, threshold)

def live_verify_cli(name: str, attempts: int = 20, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    return live_verify(name, attempts=attempts, threshold=threshold)

# -------------------- Main --------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(description="Simple LBPH face authentication")
    sub = parser.add_subparsers(dest="command")

    enroll_p = sub.add_parser("enroll", help="Enroll a new face")
    enroll_p.add_argument("--name", required=True, help="User name to enroll")
    enroll_p.add_argument("image", help="Path to face image")

    train_p = sub.add_parser("train", help="Train LBPH model from enrolled images")

    verify_p = sub.add_parser("verify_image", help="Verify using a static image")
    verify_p.add_argument("--name", required=True, help="User name")
    verify_p.add_argument("image", help="Path to image")
    verify_p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    live_p = sub.add_parser("live_verify", help="Verify using webcam")
    live_p.add_argument("--name", required=True, help="User name")
    live_p.add_argument("--attempts", type=int, default=20)
    live_p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    args = parser.parse_args(argv)

    # basic pre-check
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        print("ERROR: OpenCV and numpy required. Install: pip install opencv-contrib-python numpy")
        return

    # ensure cascade present (best-effort)
    if not ensure_cascade():
        print("⚠️ Warning: Haar cascade not found and couldn't be copied. Some face detection may fail.")
    else:
        print("✓ Haar cascade ready.")

    if args.command == "enroll":
        ok = enroll_image_cli(args.name, args.image)
        print(f"ENROLL: {ok}")
        return
    elif args.command == "train":
        ok = train_lbph()
        print(f"TRAIN: {ok}")
        return
    elif args.command == "verify_image":
        ok, score = verify_image_cli(args.name, args.image, threshold=args.threshold)
        print(f"VERIFY_IMAGE: ok={ok} score={score}")
        return
    elif args.command == "live_verify":
        ok, score = live_verify_cli(args.name, attempts=args.attempts, threshold=args.threshold)
        print(f"LIVE_VERIFY: ok={ok} score={score}")
        return
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
