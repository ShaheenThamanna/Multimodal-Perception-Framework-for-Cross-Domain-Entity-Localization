"""
Multimodal Perception Framework for Cross-Domain Entity Localization
=====================================================================
Multi-model detection pipeline:
  Model 1 — YOLOv8x      : COCO 80 classes  (best accuracy, general objects)
  Model 2 — YOLOv8x-oiv7 : Open Images v7, 601 classes (fruits, food, fine-grained)
  Model 3 — YOLOv8x-world: YOLO-World open-vocab, can detect ANY text prompt directly

All three run in parallel on every image. Detections are merged and
deduplicated via IoU-based NMS so the same object is not counted twice.
"""

from flask import Flask, render_template, request, url_for
from ultralytics import YOLO, YOLOWorld
import spacy
import os
import cv2
import uuid
import numpy as np

# ── Optional: WordNet synonyms ───────────────────────────────────────────────
try:
    from nltk.corpus import wordnet
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS  = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16 MB

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Confidence / NMS settings ────────────────────────────────────────────────
CONF_THRESHOLD  = 0.25   # lower = more detections; raise if too noisy
IOU_NMS         = 0.45   # IoU threshold for cross-model deduplication

# ── Load all models once at startup ─────────────────────────────────────────
print("[MPFCEL] Loading detection models …")

# Model 1: YOLOv8x — best COCO model (80 classes)
model_coco = YOLO("yolov8x.pt")
model_coco.to("cpu")
print("  ✓ YOLOv8x  (COCO 80 classes)")

# Model 2: YOLOv8x Open Images v7 — 601 classes including fruits, food, body parts
try:
    model_oiv7 = YOLO("yolov8x-oiv7.pt")
    model_oiv7.to("cpu")
    print("  ✓ YOLOv8x-oiv7 (Open Images 601 classes)")
except Exception as e:
    print(f"  ✗ OIv7 unavailable ({e}), falling back to yolov8n-oiv7")
    try:
        model_oiv7 = YOLO("yolov8n-oiv7.pt")
        model_oiv7.to("cpu")
    except Exception:
        model_oiv7 = None

# Model 3: YOLO-World — open-vocabulary (set classes from query at runtime)
try:
    model_world = YOLOWorld("yolov8x-worldv2.pt")
    model_world.to("cpu")
    YOLO_WORLD_AVAILABLE = True
    print("  ✓ YOLO-World v2 (open-vocabulary)")
except Exception as e:
    print(f"  ✗ YOLO-World unavailable ({e})")
    model_world = None
    YOLO_WORLD_AVAILABLE = False

print("[MPFCEL] All models ready.\n")

# ── NLP ──────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")


# ── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").lower().strip()


def get_synonyms(word: str) -> set:
    synonyms = {word}
    if not WORDNET_AVAILABLE:
        return synonyms
    for syn in wordnet.synsets(word, pos=wordnet.NOUN):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").lower())
    return synonyms


def extract_query_terms(text: str) -> set:
    """
    Extracts nouns, noun chunks, and WordNet synonyms from a natural language query.
    Returns a flat set of strings used for label matching.
    """
    doc = nlp(text)
    terms = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            word = token.text.lower()
            terms.add(word)
            terms.update(get_synonyms(word))
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        terms.add(phrase)
        terms.add(chunk.root.text.lower())
        terms.update(get_synonyms(chunk.root.text.lower()))
    return terms


def extract_noun_list(text: str) -> list[str]:
    """Return a clean list of nouns for YOLO-World class setting."""
    doc = nlp(text)
    nouns = []
    seen = set()
    for chunk in doc.noun_chunks:
        n = chunk.root.text.lower()
        if n not in seen:
            nouns.append(n)
            seen.add(n)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.text.lower() not in seen:
            nouns.append(token.text.lower())
            seen.add(token.text.lower())
    return nouns if nouns else [text.strip().lower()]


def iou(boxA, boxB) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def nms_merge(detections: list[dict], iou_thresh: float = IOU_NMS) -> list[dict]:
    """
    Cross-model NMS: if two detections from different models overlap heavily
    (IoU > iou_thresh), keep the one with higher confidence.
    """
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in detections:
        suppressed = False
        for k in kept:
            if iou(det["box"], k["box"]) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


def run_model(model_obj, image_path: str, source_tag: str) -> list[dict]:
    """Run a single YOLO model and return a list of raw detection dicts."""
    if model_obj is None:
        return []
    try:
        results = model_obj(image_path, device="cpu", conf=CONF_THRESHOLD, verbose=False)
        dets = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            label  = normalize_label(model_obj.names[cls_id])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                "label":      label,
                "confidence": conf,
                "box":        [x1, y1, x2, y2],
                "source":     source_tag,
                "matched":    False,
            })
        return dets
    except Exception as e:
        print(f"  [warn] {source_tag} inference failed: {e}")
        return []


def run_yolo_world(image_path: str, query_nouns: list[str]) -> list[dict]:
    """Run YOLO-World with the query nouns as the class vocabulary."""
    if not YOLO_WORLD_AVAILABLE or model_world is None:
        return []
    try:
        model_world.set_classes(query_nouns)
        results = model_world(image_path, device="cpu", conf=CONF_THRESHOLD, verbose=False)
        dets = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            label  = normalize_label(model_world.names[cls_id])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                "label":      label,
                "confidence": conf,
                "box":        [x1, y1, x2, y2],
                "source":     "yolo-world",
                "matched":    True,   # world model only returns queried classes
            })
        return dets
    except Exception as e:
        print(f"  [warn] YOLO-World inference failed: {e}")
        return []


# ── Source tag colours for bounding boxes (BGR) ──────────────────────────────
SOURCE_COLOURS = {
    "coco":       (56,  182, 255),   # blue
    "oiv7":       (0,   200, 100),   # green
    "yolo-world": (255, 140,  0),    # orange
}
MATCH_COLOUR    = (34,  197, 94)     # bright green for matched
UNMATCH_COLOUR  = (160, 160, 160)    # grey for unmatched


def draw_box(image, box, label, conf, matched: bool, source: str):
    x1, y1, x2, y2 = box
    colour = MATCH_COLOUR if matched else SOURCE_COLOURS.get(source, UNMATCH_COLOUR)
    thickness = 3 if matched else 1

    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

    if matched:
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 8, y1), colour, -1)
        cv2.putText(image, text, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    else:
        cv2.putText(image, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


# ── Main processing function ─────────────────────────────────────────────────

def process_image(image_path: str, text: str):
    query_terms = extract_query_terms(text)
    query_nouns = extract_noun_list(text)

    # ── Run all three models ──────────────────────────────────────────────────
    dets_coco  = run_model(model_coco, image_path, "coco")
    dets_oiv7  = run_model(model_oiv7, image_path, "oiv7") if model_oiv7 else []
    dets_world = run_yolo_world(image_path, query_nouns)

    # ── Merge and deduplicate via NMS ─────────────────────────────────────────
    all_raw = dets_coco + dets_oiv7 + dets_world
    merged  = nms_merge(all_raw)

    # ── Match detections against query terms ──────────────────────────────────
    for det in merged:
        if det["source"] == "yolo-world":
            det["matched"] = True   # already queried directly
            continue
        label_words = set(det["label"].split())
        det["matched"] = bool(label_words & query_terms) or det["label"] in query_terms

    # ── Draw on image ─────────────────────────────────────────────────────────
    image = cv2.imread(image_path)

    # Draw unmatched first (so matched render on top)
    for det in merged:
        if not det["matched"]:
            draw_box(image, det["box"], det["label"], det["confidence"],
                     matched=False, source=det["source"])
    for det in merged:
        if det["matched"]:
            draw_box(image, det["box"], det["label"], det["confidence"],
                     matched=True, source=det["source"])

    output_filename = f"{uuid.uuid4()}.jpg"
    output_path     = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    matched_objects = [d["label"] for d in merged if d["matched"]]

    all_objects = [
        {
            "label":      d["label"],
            "confidence": round(d["confidence"] * 100, 1),
            "matched":    d["matched"],
            "source":     d["source"],
        }
        for d in sorted(merged, key=lambda x: x["confidence"], reverse=True)
    ]

    return output_filename, matched_objects, all_objects


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    output_image_url = None
    input_image_url  = None
    warning          = None
    matched_objects  = []
    all_objects      = []
    query_text       = ""
    models_used      = []

    # Build model info for template
    models_info = [
        {"name": "YOLOv8x",       "classes": "80",  "dataset": "COCO",         "active": True},
        {"name": "YOLOv8x-OIv7",  "classes": "601", "dataset": "Open Images",  "active": model_oiv7 is not None},
        {"name": "YOLO-World v2", "classes": "∞",   "dataset": "Open-vocab",   "active": YOLO_WORLD_AVAILABLE},
    ]

    if request.method == "POST":
        image      = request.files.get("image")
        text       = request.form.get("text", "").strip()
        query_text = text

        if not image or image.filename == "":
            return render_template("index.html", error="No image selected.",
                                   models_info=models_info, query_text=query_text)

        if not allowed_file(image.filename):
            return render_template("index.html",
                error="Invalid file type. Allowed: JPG, PNG, WEBP, BMP.",
                models_info=models_info, query_text=query_text)

        if not text:
            return render_template("index.html", error="Please enter a description.",
                                   models_info=models_info, query_text=query_text)

        filename   = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        input_image_url = url_for("static", filename=f"uploads/{filename}")

        try:
            output_filename, matched_objects, all_objects = process_image(image_path, text)
            output_image_url = url_for("static", filename=f"outputs/{output_filename}")

            if not matched_objects:
                detected_labels = list({o["label"] for o in all_objects})
                detected_str    = ", ".join(detected_labels) if detected_labels else "nothing recognisable"
                warning = (
                    f"No entities matching your query were localised. "
                    f"Detected in scene: {detected_str}. "
                    f"Try rephrasing or using a more general term."
                )
        except Exception as e:
            return render_template("index.html", error=str(e),
                                   models_info=models_info, query_text=query_text)

    return render_template(
        "index.html",
        output_image    = output_image_url,
        input_image     = input_image_url,
        warning         = warning,
        matched_objects = matched_objects,
        all_objects     = all_objects,
        query_text      = query_text,
        models_info     = models_info,
    )


if __name__ == "__main__":
    app.run(debug=True)