# Multimodal Perception Framework for Cross-Domain Entity Localization

A web app that finds specific objects in a photo based on a plain-English description — powered by multiple YOLO models and NLP for smarter, query-aware detection.

Upload any image, type something like `"find the banana"`, and the app highlights only the object you asked for — everything else in the photo is shown faded out, so the result is easy to read at a glance.

![Detection demo](docs/demo_result.png)

---

## What it does (in simple terms)

Most object detection tools show you *everything* they can find in a photo — every person, every chair, every random object — whether you care about it or not.

This app is different: you tell it what you're looking for in normal English, and it shows you only that. Behind the scenes, it runs **three different AI detection models at once** so it doesn't miss things, understands your query using **basic NLP (language processing)**, and then combines all the results into one clean, readable image.

## Features

- **Plain-English search** — no need to know exact object category names, just describe what you want
- **Three detection models running together** — covers a much wider range of objects than any single model alone
- **Smart word matching** — understands synonyms (e.g. "auto" also matches "car") using WordNet
- **Duplicate removal** — if two models detect the same object, only one clean box is kept, not two overlapping ones
- **Visual highlighting** — matched objects get a bold green box with a confidence score; everything else fades into the background
- **Simple web interface** — upload an image, type a query, click a button, see the result

## Keyboard shortcuts

This app is a standard web form, so it doesn't have custom keyboard shortcuts yet — just normal browser behavior (e.g. pressing **Enter** while typing in the query box submits the form, and **Tab** moves between the upload field and text box). Adding real shortcuts (like a hotkey to trigger the search) is listed below as a possible improvement.

## Technology used

| Tool | What it's used for |
|---|---|
| **Python** | The main programming language |
| **Flask** | Runs the web app and handles the upload/search page |
| **YOLOv8x** | Detects common, everyday objects (80 categories) |
| **YOLOv8x-oiv7** | Detects a much wider range of objects (601 categories) |
| **YOLO-World v2** | Detects *any* object you describe, even ones the other two models don't know |
| **spaCy** | Reads your text query and picks out the important words (nouns) |
| **NLTK WordNet** | Expands those words into similar/synonym words, so more matches are caught |
| **OpenCV** | Draws the boxes and labels on the final image |
| **NumPy** | Helps with number crunching behind the scenes |

## How it works (step by step)

1. **You upload a photo and type a description**, like "find the banana."
2. **The text is read by spaCy**, which picks out the important nouns from your sentence.
3. **WordNet expands those words** into similar words, so if you typed "auto," it also checks for "car."
4. **Three detection models scan the photo at the same time:**
   - One trained on common objects
   - One trained on a much bigger object list
   - One that can detect literally anything you typed, even unusual objects
5. **All the results are combined**, and duplicate detections (same object found by two models) are cleaned up automatically.
6. **The final image is drawn** — objects that match your query are shown in green with confidence scores, everything else is faded out.
7. **The result is shown back to you** in the browser, along with a list of everything detected.

## How to run this project

**1. Clone the repository**
```bash
git clone https://github.com/ShaheenThamanna/Multimodal-Perception-Framework-for-Cross-Domain-Entity-Localization.git
cd Multimodal-Perception-Framework-for-Cross-Domain-Entity-Localization
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install the required libraries**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Run the app**
```bash
python app.py
```

**5. Open your browser** and go to:
```
http://127.0.0.1:5000
```

> The first time you run it, the three YOLO model files will download automatically (they're large, around 130MB each) — this may take a few minutes depending on your internet speed.

## What I learned building this

- How to combine **multiple AI models** together instead of relying on just one, and why that actually improves results in the real world
- How **basic NLP** (spaCy + WordNet) can be used to understand human language well enough to guide a computer vision task
- How to remove duplicate detections across different models using **IoU (Intersection over Union)** — a core computer vision concept
- How to build and structure a full **Flask web application** from scratch, connecting a frontend form to a Python backend
- Practical **Git and GitHub skills** — setting up `.gitignore` properly, keeping large model files out of version control, and pushing a clean project to GitHub

## Overall growth

This project pushed me beyond just following a tutorial — I had to make real design decisions: which models to combine, how to merge their outputs without conflicts, how to make the results actually readable for a user instead of just a wall of bounding boxes. It also gave me hands-on experience with the full lifecycle of a project — from writing code, to testing it, to properly packaging and publishing it on GitHub in a way that's usable by someone else.

## What can be improved

- **Speed** — right now all three models run one after another on the CPU, which is slow. Using a GPU or running the models in parallel would make it much faster.
- **Better confidence comparison** — the three models don't score confidence in exactly the same way, so results aren't perfectly comparable yet.
- **Smarter matching** — instead of just matching words and synonyms, using sentence embeddings (a more advanced NLP technique) could catch even more relevant matches.
- **Keyboard shortcuts and UI polish** — small usability additions like a hotkey to trigger search, drag-and-drop upload, or a loading spinner.
- **Automated testing** — adding proper test cases so future changes don't accidentally break the detection logic.
- **Deployment** — currently this only runs on your own computer; hosting it online (e.g. on Render or Railway) would let anyone try it without installing anything.

## Project structure

```
├── app.py                  # Main Flask app and detection pipeline
├── requirements.txt         # Python libraries needed to run the project
├── templates/
│   └── index.html           # The web page (upload form + results)
└── static/
    ├── uploads/              # Stores uploaded images (not tracked in Git)
    └── outputs/              # Stores result images (not tracked in Git)
```

## Authors

Built as a final-year B.Tech project (AI & Data Science) by Shaheen Thamanna B, Helsiyah Salomi B, and Muskaan Fathima S.
