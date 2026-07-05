"""
================================================================
🍽️  FoodScan — Calorie Estimation from a Single Food Photo
    Streamlit app · M2 IASD — Université Paris Dauphine PSL
----------------------------------------------------------------
Model : SigLIP-So400M (ViT, 428M params) fine-tuned with a
        triple objective — Supervised Contrastive + Huber
        regression + 703-way calorie classification.
Inference (single image, no retrieval index needed):
  1. Regression head  -> expm1(log-calories)
  2. Classification   -> softmax over the 703 calorie values
                         seen in training
  3. Confidence-weighted blend, snapped to the nearest known
     training calorie value (96% of dishes belong to a known
     group, so snapping is near-lossless).
================================================================
"""

import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm

# ────────────────────────────────────────────────────────────────
#  EDIT ME — team info shown in the sidebar
# ────────────────────────────────────────────────────────────────
TEAM_NAMES = "ALDARWISH Giuliano . MILED Mayy"          # <── put your names here
KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/giuns07/notebook7d78200f2b/output?select=best.model"   # <── your public notebook URL
PUBLIC_LB_MAE = 40.6                              # best public leaderboard MAE (kcal)

MODEL_NAME = "vit_so400m_patch14_siglip_224.v2_webli"
IMG_SIZE = 224
LOCAL_MODEL_PATH = "best.model"                   # used if the file sits next to app.py

# ────────────────────────────────────────────────────────────────
#  Page config + design system
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FoodScan · Calorie Estimator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800;900&family=Inter:wght@400;500;600&display=swap');

:root{
  --paper:#FBF9F4; --ink:#111111; --muted:#6B675E;
  --tomato:#D93B2B; --basil:#2E7D46; --line:#E4E0D6;
}
html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{ background:var(--paper); }
#MainMenu, footer{ visibility:hidden; }
header{ visibility:hidden; }
/* …but keep the sidebar open/close chevron usable */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"]{ visibility:visible !important; }
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg{ color:var(--ink) !important; fill:var(--ink) !important; }
.block-container{ padding-top:2.2rem; max-width:1150px; }

/* Force ink text everywhere — overrides Streamlit dark theme (white text) */
.stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown, [data-testid="stMarkdownContainer"] *{ color:var(--ink); }
[data-testid="stSidebar"] *{ color:var(--ink); }
[data-testid="stSidebar"] a{ color:var(--tomato) !important; }
.stTabs [data-baseweb="tab"]{ color:var(--ink); }
.stTabs [aria-selected="true"]{ color:var(--tomato) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] p{ color:var(--ink); }
[data-testid="stImageCaption"]{ color:var(--muted) !important; }
.nf-card, .nf-card *{ color:var(--ink); }
.nf-foot, .nf-foot *{ color:var(--muted) !important; }
.nb-pct{ color:var(--muted) !important; }
.side-h{ color:var(--muted) !important; }
.fs-tag{ color:var(--muted) !important; }
/* file uploader: keep its box readable in both themes */
[data-testid="stFileUploaderDropzone"]{ background:#fff; border:1.5px dashed var(--line); }
[data-testid="stFileUploaderDropzone"] *{ color:var(--ink) !important; }

/* wordmark */
.fs-brand{ font-family:'Archivo',sans-serif; font-weight:900; font-size:2.6rem;
  letter-spacing:-0.03em; color:var(--ink); line-height:1; }
.fs-brand .dot{ color:var(--tomato); }
.fs-tag{ color:var(--muted); font-size:0.95rem; margin-top:0.35rem; }
.fs-rule{ height:3px; background:var(--ink); margin:1.1rem 0 1.6rem 0; border:none; }

/* nutrition-facts result card (signature element) */
.nf-card{ background:#fff; border:2.5px solid var(--ink); border-radius:4px;
  padding:1.1rem 1.25rem; box-shadow:6px 6px 0 var(--line); }
.nf-title{ font-family:'Archivo',sans-serif; font-weight:900; font-size:1.9rem;
  letter-spacing:-0.02em; border-bottom:10px solid var(--ink); padding-bottom:.35rem; }
.nf-row{ display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px solid var(--line); padding:.4rem 0; font-size:.95rem; }
.nf-row.thick{ border-bottom:5px solid var(--ink); }
.nf-kcal-label{ font-family:'Archivo',sans-serif; font-weight:800; font-size:1.25rem; }
.nf-kcal{ font-family:'Archivo',sans-serif; font-weight:900; font-size:3.4rem;
  line-height:1; letter-spacing:-0.03em; }
.nf-kcal small{ font-size:1.15rem; font-weight:800; color:var(--muted); margin-left:.25rem;}
.nf-foot{ font-size:.78rem; color:var(--muted); padding-top:.55rem; line-height:1.45;}

/* neighbor bars */
.nb-wrap{ margin-top:.35rem; }
.nb-row{ display:flex; align-items:center; gap:.6rem; margin:.3rem 0; font-size:.88rem;}
.nb-val{ width:86px; font-weight:600; font-variant-numeric:tabular-nums; }
.nb-bar{ flex:1; background:var(--line); border-radius:3px; height:12px; overflow:hidden;}
.nb-fill{ height:100%; background:var(--ink); }
.nb-fill.top{ background:var(--tomato); }
.nb-pct{ width:52px; text-align:right; color:var(--muted);
  font-variant-numeric:tabular-nums; }

/* misc */
.stButton>button{ background:var(--ink); color:#fff; border:none; border-radius:4px;
  font-weight:600; padding:.55rem 1.3rem; }
.stButton>button:hover{ background:var(--tomato); color:#fff; }
[data-testid="stSidebar"]{ background:#F3F0E8; border-right:1px solid var(--line); }
.side-h{ font-family:'Archivo',sans-serif; font-weight:800; font-size:.8rem;
  letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-top:1rem;}
.side-kv{ display:flex; justify-content:space-between; font-size:.88rem;
  padding:.28rem 0; border-bottom:1px dashed var(--line); }
.side-kv b{ font-variant-numeric:tabular-nums; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
#  Model definition — must mirror the training notebook exactly
# ────────────────────────────────────────────────────────────────
class FoodSigLIP(nn.Module):
    def __init__(self, model_name: str, n_classes: int,
                 img_size: int = IMG_SIZE, proj_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size)
        d = self.backbone.num_features
        self.proj_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.25),
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(256, 1))
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.35),
            nn.Linear(d, min(1024, d)), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(min(1024, d), n_classes))

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() == 3:
            f = f.mean(1)
        return (self.reg_head(f).squeeze(-1),
                self.cls_head(f),
                F.normalize(self.proj_head(f), dim=1))


# ────────────────────────────────────────────────────────────────
#  Checkpoint loading (local file, or Hugging Face Hub fallback)
# ────────────────────────────────────────────────────────────────
def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


def ensure_checkpoint() -> str:
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    repo = _secret("HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo,
            filename=_secret("HF_FILENAME", "best.model"),
            token=_secret("HF_TOKEN") or None)
    st.error(
        "**Model weights not found.** Place `best.model` next to `app.py`, "
        "or set the `HF_REPO` secret to a Hugging Face repo containing it "
        "(see README_DEPLOY.md).")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    """Memory-frugal loading: weights are copied tensor-by-tensor into the
    freshly built model (dtype-converting in place), so a fp16 checkpoint
    never gets duplicated as a full fp32 dict in RAM — critical on the
    free hosting tier."""
    path = ensure_checkpoint()
    model = FoodSigLIP(MODEL_NAME, 703)                 # placeholder n_classes
    try:                                                # mmap: tensors stay on disk
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    idx_to_cal = np.asarray(ckpt["idx_to_cal"], dtype=np.float64)
    if int(ckpt["n_classes"]) != 703:                   # rebuild only if needed
        model = FoodSigLIP(MODEL_NAME, int(ckpt["n_classes"]))

    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    matched, skipped = 0, []
    with torch.no_grad():
        for k, v in ckpt["state_dict"].items():
            k = k.replace("module.", "")
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)                         # in-place, converts fp16→fp32
                matched += 1
            else:
                skipped.append(k)
    del ckpt
    load_note = ("strict" if not skipped and matched == len(own)
                 else f"tolerant ({matched} loaded / {len(skipped)} skipped)")
    model.eval()
    torch.set_grad_enabled(False)
    return model, idx_to_cal, load_note


# ────────────────────────────────────────────────────────────────
#  Inference
# ────────────────────────────────────────────────────────────────
_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def predict(model, idx_to_cal, pil_img: Image.Image):
    """2-view TTA (original + horizontal flip), then a confidence-weighted
    blend of the classification expectation and the regression output,
    snapped to the nearest calorie value seen in training."""
    views = torch.stack([_TF(pil_img),
                         _TF(pil_img.transpose(Image.FLIP_LEFT_RIGHT))])
    with torch.inference_mode():
        reg, cls, _ = model(views)

    reg_kcal = float(np.clip(np.expm1(reg.mean().item()), 50, 5000))

    probs = F.softmax(cls.float(), dim=1).mean(0).numpy()
    order = probs.argsort()[::-1][:5]
    top_cals, top_p = idx_to_cal[order], probs[order]
    conf = float(top_p.sum())                               # mass on top-5
    cls_kcal = float((top_cals * (top_p / top_p.sum())).sum())

    w_cls = float(np.clip(0.25 + conf, 0.25, 0.75))         # trust cls when confident
    raw = w_cls * cls_kcal + (1.0 - w_cls) * reg_kcal
    final = float(idx_to_cal[np.argmin(np.abs(idx_to_cal - raw))])

    return {"final": final, "reg": reg_kcal, "cls": cls_kcal,
            "conf": conf, "top": list(zip(top_cals.tolist(), top_p.tolist()))}


# ────────────────────────────────────────────────────────────────
#  Sidebar — model card
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fs-brand" style="font-size:1.6rem">Food'
                '<span class="dot">Scan</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-h">Model card</div>', unsafe_allow_html=True)
    for k, v in [("Backbone", "SigLIP-So400M"), ("Parameters", "428 M"),
                 ("Input", f"{IMG_SIZE}×{IMG_SIZE} px"),
                 ("Calorie classes", "703"),
                 ("Public LB MAE", f"{PUBLIC_LB_MAE:.1f} kcal")]:
        st.markdown(f'<div class="side-kv"><span>{k}</span><b>{v}</b></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="side-h">Training recipe</div>', unsafe_allow_html=True)
    st.markdown(
        "- Supervised Contrastive + Huber + CE (40/40/20)\n"
        "- 3-phase progressive unfreezing + SWA\n"
        "- Test-time augmentation at inference\n"
        "- Predictions snapped to known calorie values"
    )
    st.markdown('<div class="side-h">Links</div>', unsafe_allow_html=True)
    st.markdown(f"[Kaggle notebook]({KAGGLE_NOTEBOOK_URL})")
    st.markdown('<div class="side-h">Team</div>', unsafe_allow_html=True)
    st.markdown(TEAM_NAMES)
    st.caption("Deep Learning for Images · M2 IASD\nUniversité Paris Dauphine – PSL")


# ────────────────────────────────────────────────────────────────
#  Main layout
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="fs-brand">Food<span class="dot">Scan</span></div>'
            '<div class="fs-tag">Point it at a dish. Get the calories. '
            'One photo, one number — powered by a 428M-parameter vision '
            'transformer fine-tuned on 3,098 real dishes.</div>'
            '<hr class="fs-rule">', unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown("#### 1 · Your photo")
    tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
    img_file = None
    with tab_up:
        img_file = st.file_uploader(
            "Upload a food photo (JPG / PNG)", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed")
    with tab_cam:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam is not None:
            img_file = cam

    pil_img = None
    if img_file is not None:
        pil_img = Image.open(img_file).convert("RGB")
        st.image(pil_img, use_container_width=True, caption="Input image")

with col_out:
    st.markdown("#### 2 · Estimate")
    if pil_img is None:
        st.markdown(
            '<div class="nf-card"><div class="nf-title">Nutrition Facts</div>'
            '<div class="nf-row thick"><span class="nf-kcal-label">Calories</span>'
            '<span class="nf-kcal" style="color:var(--line)">···</span></div>'
            '<div class="nf-foot">Upload a photo on the left — the estimate '
            'appears here as a nutrition label.</div></div>',
            unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing the dish…"):
            model, idx_to_cal, load_note = load_model()
            t0 = __import__("time").time()
            out = predict(model, idx_to_cal, pil_img)
            dt = __import__("time").time() - t0

        pct_day = out["final"] / 2000 * 100
        walk_min = out["final"] / 4.0          # ≈4 kcal/min brisk walking

        bars = ""
        for i, (c, p) in enumerate(out["top"]):
            width = max(4, p / out["top"][0][1] * 100)
            bars += (f'<div class="nb-row"><span class="nb-val">{c:,.0f} kcal</span>'
                     f'<div class="nb-bar"><div class="nb-fill{" top" if i == 0 else ""}"'
                     f' style="width:{width:.0f}%"></div></div>'
                     f'<span class="nb-pct">{p*100:.1f}%</span></div>')

        st.markdown(f"""
<div class="nf-card">
  <div class="nf-title">Nutrition Facts</div>
  <div class="nf-row"><span>Serving — as pictured (1 dish)</span></div>
  <div class="nf-row thick">
    <span class="nf-kcal-label">Calories</span>
    <span class="nf-kcal">{out["final"]:,.0f}<small>kcal</small></span>
  </div>
  <div class="nf-row"><span>Share of a 2,000 kcal day</span><b>{pct_day:.0f}%</b></div>
  <div class="nf-row"><span>≈ brisk walking to burn it</span><b>{walk_min:.0f} min</b></div>
  <div class="nf-row"><span>Regression head</span><b>{out["reg"]:,.0f} kcal</b></div>
  <div class="nf-row"><span>Classifier expectation (top-5)</span><b>{out["cls"]:,.0f} kcal</b></div>
  <div class="nf-row"><span>Model confidence (top-5 mass)</span><b>{out["conf"]*100:.0f}%</b></div>
  <div style="margin-top:.7rem"><b style="font-family:'Archivo';font-size:.85rem;
    letter-spacing:.08em;text-transform:uppercase">Closest known dishes</b>
    <div class="nb-wrap">{bars}</div></div>
  <div class="nf-foot">Estimated from a single photo in {dt:.2f}s on CPU ·
    weights loaded ({load_note}). Portion size, cooking fat and hidden
    ingredients are invisible to any camera — treat this as an informed
    estimate, not a lab measurement.</div>
</div>""", unsafe_allow_html=True)

with st.expander("How does FoodScan work?"):
    st.markdown("""
**Backbone.** A SigLIP-So400M vision transformer (428M parameters), pre-trained
on the WebLI image–text corpus, fine-tuned end-to-end with progressive
unfreezing and Stochastic Weight Averaging.

**Triple objective.** During training the model optimizes three heads at once:
a *Supervised Contrastive* loss that pulls photos of equally-caloric dishes
together in feature space, a *Huber regression* on log-calories, and a
*703-way classification* over every calorie value present in the training set.

**Inference in this app.** Each photo is encoded twice (original + mirrored),
the regression and classification heads are blended according to the
classifier's confidence, and the result is snapped to the nearest calorie
value ever observed in training — since 96% of test dishes belong to a known
dish group, snapping removes noise at almost no cost.

**Honest limits.** Calorie estimation from one RGB photo is intrinsically
ill-posed: the camera cannot see portion weight, oil quantity, or sugar
content. The model reaches a mean absolute error of ~40 kcal on the public
leaderboard of the FoodScan challenge.
""")
