"""
Plant gate — CLIP-based checks that run BEFORE / AROUND the AgroCure pipeline.

All checks share ONE image embedding (encode the image once, reuse it):
  • check(feat)          — is this a plant leaf at all?  (vs animal/person/object)
  • check_health(feat)   — healthy vs diseased leaf
  • check_framing(feat)  — close-up single leaf vs wide whole-plant/scene shot

CLIP (open_clip ViT-B-32) is open-vocabulary, so each check is just the image
embedding compared (cosine similarity) to a small group of cached text prompts.
No training required. CLIP weights (~350 MB) download once and are cached.
"""

import torch
import open_clip

import config_v4 as config

# ── Tunables ──────────────────────────────────────────────────────────────────
PLANT_THRESHOLD = 0.55     # accept as plant if plant prompts win this share
STAGE1_MIN_CONF = 0.05     # last-resort hard floor: below this we reject outright
HEALTH_THRESHOLD = 0.70    # declare healthy only above this (cautious)
FRAMING_THRESHOLD = 0.50   # below this = looks like a wide shot → show a tip

# ── Open-set / unknown-plant gating (no retraining) ───────────────────────────
# These produce a *warning*, not a rejection: we still show the best guess.
# Tune on a small held-out set of unsupported-species leaves (AUROC / FPR@95).
KNOWN_PLANT_THRESHOLD = 0.45   # CLIP prob mass on a supported species; below = suspect
S1_WARN_CONF          = 0.40   # Stage-1 top-1 below this = uncertain identity
S1_WARN_MARGIN        = 0.12   # top1 - top2 below this = two species nearly tied

# CLIP "this is some other plant" anchors, contrasted against the 16 known species.
OTHER_PLANT_PROMPTS = [
    "a photo of a different plant leaf",
    "a photo of an unknown plant species leaf",
    "a leaf from a plant that is not listed",
    "a houseplant or ornamental leaf",
]


def _plant_to_prompt(name: str) -> str:
    """Turn a config plant name ('Corn-Maize', 'Apple Tree') into a CLIP prompt."""
    clean = name.lower().replace("-", " ")
    return f"a photo of a {clean} leaf"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

PLANT_PROMPTS = [
    "a photo of a plant leaf",
    "a close-up photo of a green leaf",
    "a photo of a diseased plant leaf with spots",
    "a photo of plant foliage",
    "leaves of a crop plant",
]
NONPLANT_PROMPTS = [
    "a photo of an animal",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a horse",
    "a photo of a person",
    "a photo of a human face",
    "a photo of a human hand",
    "a photo of a car",
    "a photo of an everyday object",
    "a photo of furniture",
    "a screenshot with text",
    "a photo of food on a plate",
]

# Health prompts (healthy first). Chosen empirically — CLIP is weak here, so the
# threshold is set high to avoid calling a sick leaf "healthy".
HEALTHY_PROMPTS = [
    "a healthy leaf",
    "a green leaf without spots",
]
DISEASED_PROMPTS = [
    "a sick leaf",
    "a leaf with spots or lesions",
]

# Framing prompts (close-up first). Used only for a soft tip, never to reject.
CLOSEUP_PROMPTS = [
    "a close-up photo of a single leaf",
    "a macro photo of one leaf filling the frame",
]
WIDE_PROMPTS = [
    "a wide photo of a whole plant",
    "a photo of several plants or a garden",
    "a photo of a potted plant seen from far away",
]


class PlantGate:
    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

        def encode(prompts):
            with torch.no_grad():
                t = self.model.encode_text(tokenizer(prompts).to(device))
                return t / t.norm(dim=-1, keepdim=True)

        self.prompts = PLANT_PROMPTS + NONPLANT_PROMPTS
        self.n_plant = len(PLANT_PROMPTS)
        self.text_features = encode(self.prompts)

        self.n_healthy = len(HEALTHY_PROMPTS)
        self.health_features = encode(HEALTHY_PROMPTS + DISEASED_PROMPTS)

        self.n_closeup = len(CLOSEUP_PROMPTS)
        self.framing_features = encode(CLOSEUP_PROMPTS + WIDE_PROMPTS)

        # Known-plant cross-check: the 16 supported species vs "some other plant".
        self.known_names = list(config.PLANT_DISEASES.keys())
        self.n_known = len(self.known_names)
        self.known_features = encode(
            [_plant_to_prompt(n) for n in self.known_names] + OTHER_PLANT_PROMPTS
        )

    @torch.no_grad()
    def embed(self, pil_img):
        """Encode the image once; reuse the result across all checks."""
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    def _probs(self, feat, text_features):
        logits = self.model.logit_scale.exp() * feat @ text_features.T
        return logits.softmax(dim=-1)[0]

    @torch.no_grad()
    def check(self, feat) -> dict:
        """Is this a plant leaf? Return {'is_plant', 'top_label', 'top_conf', 'reason'}."""
        probs = self._probs(feat, self.text_features)
        plant_prob = probs[: self.n_plant].sum().item()
        best_idx = int(probs.argmax().item())
        best_label = self.prompts[best_idx]
        best_conf = round(probs[best_idx].item(), 4)

        if plant_prob >= PLANT_THRESHOLD:
            return {"is_plant": True, "top_label": best_label,
                    "top_conf": best_conf, "reason": f"plant score {plant_prob:.0%}"}

        nice = best_label.replace("a photo of ", "").replace("a ", "", 1)
        return {"is_plant": False, "top_label": best_label, "top_conf": best_conf,
                "reason": f"looks like: {nice} ({best_conf:.0%})"}

    @torch.no_grad()
    def check_health(self, feat) -> dict:
        """Healthy vs diseased. Return {'healthy', 'healthy_score'}."""
        probs = self._probs(feat, self.health_features)
        healthy_score = probs[: self.n_healthy].sum().item()
        return {"healthy": healthy_score >= HEALTH_THRESHOLD,
                "healthy_score": round(healthy_score, 4)}

    @torch.no_grad()
    def check_known_plant(self, feat) -> dict:
        """CLIP zero-shot: does this look like one of the 16 supported species?

        Returns the probability mass on known species, plus CLIP's own closest
        supported plant (useful to cross-check against the ResNet's pick).
        """
        probs = self._probs(feat, self.known_features)
        known_block = probs[: self.n_known]
        known_score = known_block.sum().item()
        best = int(known_block.argmax().item())
        return {
            "known_score": round(known_score, 4),
            "clip_plant": self.known_names[best],
            "clip_plant_conf": round(known_block[best].item(), 4),
        }

    @torch.no_grad()
    def check_framing(self, feat) -> dict:
        """Close-up vs wide shot. Return {'closeup', 'closeup_score'}."""
        probs = self._probs(feat, self.framing_features)
        closeup_score = probs[: self.n_closeup].sum().item()
        return {"closeup": closeup_score >= FRAMING_THRESHOLD,
                "closeup_score": round(closeup_score, 4)}
