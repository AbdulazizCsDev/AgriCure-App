"""
AgroCure v4 — Central Configuration
Hierarchical model: Stage 1 (plant) → Stage 2 (disease per plant)
"""

import os

# ── Separators ────────────────────────────────────────────────────────────────
FOLDER_SEP  = "--"
DISPLAY_SEP = " | "

def folder_to_display(name): return name.replace(FOLDER_SEP, DISPLAY_SEP)
def display_to_folder(name): return name.replace(DISPLAY_SEP, FOLDER_SEP)

# ── Class Merges (applied before training) ────────────────────────────────────
# Format: "Source Class" → "Target Class"
CLASS_MERGES = {
    "Tomato--Brown Spots":              "Tomato--Blight",
    "Corn-Maize--Leaf Spot":            "Corn-Maize--Leaf Disease",
    "Corn-Maize--Brown Spots":          "Corn-Maize--Leaf Disease",
    "Corn-Maize--Northern Leaf Blight": "Corn-Maize--Leaf Disease",
    "Potato--Early Blight":             "Potato--Blight",
    "Potato--Late Blight":              "Potato--Blight",
}

# ── Plant → Diseases mapping ──────────────────────────────────────────────────
# Plants with only 1 disease don't need a Stage 2 model
PLANT_DISEASES = {
    "Apple Tree":     ["Mosaic Disease", "Rust", "Scab", "Spider Mite"],
    "Bell Pepper":    ["Leaf Spot"],
    "Cassava":        ["Brown Leaf Spot", "Mosaic", "Root Rot"],
    "Chinese Cabbage":["Downy Mildew"],
    "Chinese Rose":   ["Gray Mold", "Powdery Mildew"],
    "Citrus Tree":    ["Citrus Leaf Miner"],
    "Corn-Maize":     ["Leaf Disease", "Rust", "Streak", "Stripe", "Yellowing"],
    "Grape":          ["Black Rot"],
    "Mango Tree":     ["Anthracnose"],
    "Pea":            ["Downy Mildew"],
    "Peach Tree":     ["Shot Hole Disease"],
    "Pear Tree":      ["Blister Mites", "Powdery Mildew", "Rust"],
    "Potato":         ["Blight"],
    "Squash":         ["Powdery Mildew"],
    "Tomato":         ["Blight", "Leaf Mold", "Viral Disease"],
    "Wheat":          ["Tan Spot"],
}

# Plants that need a Stage 2 model (more than 1 disease)
MULTI_DISEASE_PLANTS = [p for p, d in PLANT_DISEASES.items() if len(d) > 1]
# Plants with only 1 disease — Stage 1 is enough
SINGLE_DISEASE_PLANTS = {p: d[0] for p, d in PLANT_DISEASES.items() if len(d) == 1}

# ── Data sources (for drift tracking) ────────────────────────────────────────
DATA_SOURCES = {
    "plantdoc":  {"added": "2026-01", "notes": "PlantDoc public dataset"},
    "inat":      {"added": "2026-01", "notes": "iNaturalist scrape"},
    "client":    {"added": "2026-01", "notes": "Client-provided images"},
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "dataset")
SPLITS_DIR      = os.path.join(BASE_DIR, "splits")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORT_DIR      = os.path.join(BASE_DIR, "report")

# ── Dataset ───────────────────────────────────────────────────────────────────
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUG_TARGET      = 200

SPLIT_TRAIN     = 0.60
SPLIT_VAL       = 0.10
SPLIT_TEST      = 0.30
SPLIT_SEED      = 42

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 4

# ── Stage 1 Training ──────────────────────────────────────────────────────────
S1_PHASE_A_EPOCHS = 3
S1_PHASE_B_EPOCHS = 30
S1_PHASE_A_LR     = 1e-3
S1_PHASE_B_LR     = 1e-4
S1_EARLY_STOP_PAT = 8

# ── Stage 2 Training ──────────────────────────────────────────────────────────
S2_PHASE_A_EPOCHS = 3
S2_PHASE_B_EPOCHS = 30
S2_PHASE_A_LR     = 1e-3
S2_PHASE_B_LR     = 1e-4
S2_EARLY_STOP_PAT = 8

# ── Common ────────────────────────────────────────────────────────────────────
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.3
MIXUP_ALPHA     = 0.2
LABEL_SMOOTH    = 0.0
LR_MIN          = 1e-6

# ── Saved file names ──────────────────────────────────────────────────────────
STAGE1_MODEL    = os.path.join(MODELS_DIR, "stage1_plant_classifier.pth")
STAGE1_CLASSES  = os.path.join(MODELS_DIR, "stage1_class_names.json")
TEMPERATURE_S1  = os.path.join(MODELS_DIR, "stage1_temperature.json")
