"""
Fine-tune Stage 1 (plant classifier) to ADD a new plant: Lemon.

Grows the classifier head 16 -> 17, KEEPS the existing 16 plants (their learned
weights are copied over), and learns "Lemon" from your lemon dataset. Trains on
the full dataset so the model doesn't forget the original plants.

Outputs drop-in replacements in  models_lemon/ :
  stage1_plant_classifier.pth
  stage1_class_names.json
  stage1_temperature.json

Deploy: upload those 3 files (replacing the old ones) to your HF model Space,
add  "Lemon": ["Healthy"]  to config_v4.PLANT_DISEASES there, and the connected
app recognises lemons automatically (no app change).

Run:
  pip install torch torchvision pillow
  python finetune_stage1.py
A GPU is strongly recommended (CPU works but is slow). Quick check without
training:  SMOKE=1 python finetune_stage1.py
"""
import os, json, zipfile, random, tempfile, time, copy
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ── EDIT THESE PATHS IF NEEDED ──────────────────────────────────────────────
FULL_DATASET = r"C:\Users\Aziz\Downloads\FINAL_DATASET"          # 16 plants (Plant--Disease folders)
LEMON_ZIP    = r"C:\Users\Aziz\Downloads\lemon_dataset_jpg.zip"  # the new plant's images
NEW_PLANT    = "Lemon"

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
OUT    = os.path.join(BASE, "models_lemon")
EXISTING_W   = os.path.join(MODELS, "stage1_plant_classifier.pth")
EXISTING_CLS = os.path.join(MODELS, "stage1_class_names.json")

# ── hyperparameters ─────────────────────────────────────────────────────────
EPOCHS, BATCH, IMG = 10, 32, 224
BACKBONE_LR, HEAD_LR, WD, DROPOUT = 1e-4, 1e-3, 1e-4, 0.3
VAL_FRAC, SEED = 0.15, 42
SMOKE = bool(os.environ.get("SMOKE"))

random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# ── class order: existing 16 + the new plant appended at the end ────────────
plant_names = json.load(open(EXISTING_CLS))["plant_names"]
if NEW_PLANT not in plant_names:
    plant_names = plant_names + [NEW_PLANT]
idx = {p: i for i, p in enumerate(plant_names)}
NUM = len(plant_names)

# ── gather (path, label) at the PLANT level (Stage 1) ───────────────────────
samples = []
for folder in sorted(os.listdir(FULL_DATASET)):
    fp = os.path.join(FULL_DATASET, folder)
    if not os.path.isdir(fp):
        continue
    plant = folder.split("--")[0].strip()
    if plant not in idx:
        print("  ! skipping unrecognised plant folder:", folder); continue
    for fn in os.listdir(fp):
        if fn.lower().endswith(IMG_EXT):
            samples.append((os.path.join(fp, fn), idx[plant]))

# new plant: extract the zip to a temp dir, then add its images
lemon_dir = os.path.join(tempfile.gettempdir(), "agro_newplant")
os.makedirs(lemon_dir, exist_ok=True)
with zipfile.ZipFile(LEMON_ZIP) as z:
    for n in z.namelist():
        if n.lower().endswith(IMG_EXT):
            dst = os.path.join(lemon_dir, os.path.basename(n))
            with open(dst, "wb") as f:
                f.write(z.read(n))
            samples.append((dst, idx[NEW_PLANT]))

cnt = Counter(l for _, l in samples)
print(f"Classes: {NUM}")
for p in plant_names:
    print(f"  {p:18s} {cnt.get(idx[p], 0)}")
print("total images:", len(samples))

# ── stratified train/val split ──────────────────────────────────────────────
by_cls = {}
for s in samples:
    by_cls.setdefault(s[1], []).append(s)
train, val = [], []
for c, items in by_cls.items():
    random.shuffle(items)
    k = max(1, int(len(items) * VAL_FRAC))
    val += items[:k]; train += items[k:]
random.shuffle(train)
print(f"train={len(train)} val={len(val)}  device={DEVICE}")

tf_train = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), transforms.ColorJitter(.2, .2, .2, .05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tf_val = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), l

def build(n):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, n))
    return m

# ── load existing 16-class model, transfer into a 17-class one ──────────────
old = build(NUM - 1)
old.load_state_dict(torch.load(EXISTING_W, map_location="cpu")["model_state"])
model = build(NUM)
sd_old, sd_new = old.state_dict(), model.state_dict()
for k in sd_new:                                   # copy backbone (matching shapes)
    if k in sd_old and sd_old[k].shape == sd_new[k].shape:
        sd_new[k] = sd_old[k]
model.load_state_dict(sd_new)
with torch.no_grad():                              # preserve the 16 learned rows, init the new one
    ol, nl = old.fc[1], model.fc[1]
    nl.weight[:ol.out_features] = ol.weight
    nl.bias[:ol.out_features] = ol.bias
    nn.init.normal_(nl.weight[ol.out_features:], std=0.01); nl.bias[ol.out_features:] = 0.0
model = model.to(DEVICE)

if SMOKE:
    with torch.no_grad():
        out = model(torch.randn(2, 3, IMG, IMG).to(DEVICE))
    print("SMOKE ok — transfer + forward fine, logits:", tuple(out.shape))
    raise SystemExit(0)

train_dl = DataLoader(DS(train, tf_train), batch_size=BATCH, shuffle=True, num_workers=0)
val_dl   = DataLoader(DS(val, tf_val), batch_size=BATCH, shuffle=False, num_workers=0)

backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
head     = [p for n, p in model.named_parameters() if n.startswith("fc.")]
opt = torch.optim.AdamW([{"params": backbone, "lr": BACKBONE_LR},
                         {"params": head, "lr": HEAD_LR}], weight_decay=WD)
crit = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate():
    model.eval(); correct = tot = 0; per = Counter(); hit = Counter()
    for xb, yb in val_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pr = model(xb).argmax(1)
        correct += (pr == yb).sum().item(); tot += len(yb)
        for y, p in zip(yb.tolist(), pr.tolist()):
            per[y] += 1; hit[y] += int(y == p)
    return correct / tot, per, hit

best_acc, best_state = 0.0, None
lem = idx[NEW_PLANT]
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        run += loss.item() * len(yb)
    acc, per, hit = evaluate()
    lem_recall = hit[lem] / per[lem] if per.get(lem) else 0.0
    print(f"epoch {ep}/{EPOCHS}  loss={run/len(train):.3f}  val_acc={acc:.3f}  "
          f"{NEW_PLANT}_recall={lem_recall:.2f}  ({time.time()-t0:.0f}s)")
    if acc >= best_acc:
        best_acc = acc; best_state = copy.deepcopy(model.state_dict())

# ── temperature scaling on val (calibration) ───────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = torch.cat([model(xb.to(DEVICE)).cpu() for xb, _ in val_dl])
    labels = torch.cat([yb for _, yb in val_dl])
T = nn.Parameter(torch.ones(1))
opt_t = torch.optim.LBFGS([T], lr=0.01, max_iter=60)
def _closure():
    opt_t.zero_grad(); l = nn.functional.cross_entropy(logits / T, labels); l.backward(); return l
opt_t.step(_closure)
temperature = float(T.detach().clamp(0.5, 5.0).item())

# ── save drop-in files ──────────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)
torch.save({"model_state": best_state}, os.path.join(OUT, "stage1_plant_classifier.pth"))
json.dump({"plant_names": plant_names, "plant_to_idx": idx},
          open(os.path.join(OUT, "stage1_class_names.json"), "w"), indent=2)
json.dump({"temperature": temperature},
          open(os.path.join(OUT, "stage1_temperature.json"), "w"), indent=2)

print(f"\nDONE  best val_acc={best_acc:.3f}  temperature={temperature:.3f}")
print(f"Saved to: {OUT}")
print("Next: upload those 3 files to the model Space (replace the old ones) and add")
print('      "Lemon": ["Healthy"]  to config_v4.PLANT_DISEASES there.')
