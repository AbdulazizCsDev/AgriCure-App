"""
AgroCure v4 — Inference
Run on a single image or folder. Returns plant + disease + confidence.
"""

import os, json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import config_v4 as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_model(model_path, num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(p=config.DROPOUT),
        nn.Linear(m.fc.in_features, num_classes)
    )
    ckpt = torch.load(model_path, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state"])
    return m.to(DEVICE).eval()

class AgroCureV4:
    def __init__(self):
        print("Loading AgroCure v4 models...")

        # Stage 1
        with open(config.STAGE1_CLASSES) as f:
            s1 = json.load(f)
        self.plant_names  = s1["plant_names"]
        self.stage1_model = load_model(config.STAGE1_MODEL, len(self.plant_names))
        with open(config.TEMPERATURE_S1) as f:
            self.T1 = json.load(f)["temperature"]
        print(f"  Stage 1 loaded ✓  ({len(self.plant_names)} plants)")

        # Stage 2
        self.stage2_models = {}
        self.stage2_infos  = {}
        for plant in config.MULTI_DISEASE_PLANTS:
            plant_key  = plant.replace(" ", "_").replace("-", "_")
            model_path = os.path.join(config.MODELS_DIR, f"stage2_{plant_key}.pth")
            info_path  = os.path.join(config.MODELS_DIR, f"stage2_{plant_key}_info.json")
            temp_path  = os.path.join(config.MODELS_DIR, f"stage2_{plant_key}_temperature.json")
            if not os.path.exists(model_path):
                continue
            with open(info_path) as f:
                info = json.load(f)
            if os.path.exists(temp_path):
                with open(temp_path) as f:
                    info["temperature"] = json.load(f)["temperature"]
            num_dis = len(info["disease_labels"])
            self.stage2_models[plant_key] = load_model(model_path, num_dis)
            self.stage2_infos[plant_key]  = info
        print(f"  Stage 2 loaded ✓  ({len(self.stage2_models)} plant models)")
        print("Ready ✓\n")

    def predict(self, img_path: str) -> dict:
        img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Stage 1
            logit1 = self.stage1_model(img) / self.T1
            prob1  = torch.softmax(logit1, dim=1)
            plant_idx  = prob1.argmax(dim=1).item()
            plant_conf = prob1.max().item()
            plant_name = self.plant_names[plant_idx]

            # ── Open-set signals on Stage 1 (for unknown-plant warnings) ──
            k = min(2, prob1.shape[1])
            top = torch.topk(prob1, k=k, dim=1)
            p1 = top.values[0, 0].item()
            p2 = top.values[0, 1].item() if k > 1 else 0.0
            s1_extra = {
                "stage1_margin":  round(p1 - p2, 4),
                "stage1_entropy": round(float(-(prob1 * torch.log(prob1 + 1e-9)).sum().item()), 4),
                "stage1_energy":  round(float(torch.logsumexp(logit1, dim=1).item()), 4),
                "alt_plant":      self.plant_names[int(top.indices[0, 1].item())] if k > 1 else None,
            }

            # Single-disease plant
            if plant_name in config.SINGLE_DISEASE_PLANTS:
                disease = config.SINGLE_DISEASE_PLANTS[plant_name]
                return {
                    "plant":      plant_name,
                    "disease":    disease,
                    "label":      f"{plant_name}{config.DISPLAY_SEP}{disease}",
                    "confidence": round(plant_conf, 4),
                    "stage1_conf": round(plant_conf, 4),
                    "stage2_conf": None,
                    **s1_extra,
                }

            # Stage 2
            plant_key = plant_name.replace(" ", "_").replace("-", "_")
            if plant_key not in self.stage2_models:
                return {
                    "plant":      plant_name,
                    "disease":    "Unknown",
                    "label":      plant_name,
                    "confidence": round(plant_conf, 4),
                    "stage1_conf": round(plant_conf, 4),
                    "stage2_conf": None,
                    **s1_extra,
                }

            model2 = self.stage2_models[plant_key]
            info2  = self.stage2_infos[plant_key]
            T2     = info2.get("temperature", 1.0)

            logit2 = model2(img) / T2
            prob2  = torch.softmax(logit2, dim=1)
            dis_idx  = prob2.argmax(dim=1).item()
            dis_conf = prob2.max().item()
            label    = info2["disease_labels"][dis_idx]
            disease  = label.split(config.DISPLAY_SEP)[-1] if config.DISPLAY_SEP in label else label

            return {
                "plant":       plant_name,
                "disease":     disease,
                "label":       label,
                "confidence":  round(plant_conf * dis_conf, 4),
                "stage1_conf": round(plant_conf, 4),
                "stage2_conf": round(dis_conf, 4),
                **s1_extra,
            }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_v4.py <image_path>")
        sys.exit(1)

    model = AgroCureV4()
    result = model.predict(sys.argv[1])

    print(f"Plant      : {result['plant']}")
    print(f"Disease    : {result['disease']}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print(f"  Stage 1  : {result['stage1_conf']:.1%}  (plant)")
    print(f"  Stage 2  : {result['stage2_conf']:.1%}  (disease)" if result['stage2_conf'] else "  Stage 2  : N/A (single-disease plant)")
