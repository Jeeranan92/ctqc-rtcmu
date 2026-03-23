"""
CT Phantom QC Tool — Full Pipeline
====================================
Steps:
  1. Load entire folder of DICOM files
  2. Scroll through slices with mouse wheel
  3. Auto-detect uniformity module slices
  4. Insert cylindrical lesions via image-domain
     Set 1: −10, −20, −50, −200, −500 HU  ×  4 sizes
     Set 2: −5, 0, +5 HU               ×  4 sizes (for CHO)
  5a. MTF/TTF: draw circle on lesion → ESF → LSF → FFT
  5b. CHO d': square ROI on Set-2 lesions with Gabor channels

Math background is displayed inline in the UI.

Install:
    pip install streamlit pydicom numpy scipy matplotlib Pillow scikit-image pandas
Run:
    streamlit run ctqc_app.py
"""

import io, os, copy, glob, warnings, math
import numpy as np
import matplotlib
import cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import uuid

from scipy.signal import find_peaks
from scipy.ndimage import center_of_mass
from scipy.signal import windows
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from matplotlib.patches import Circle, Rectangle
from scipy.fft import fft, fftfreq
from scipy.special import erf
from numpy.fft import fft2, ifft2
from scipy.ndimage import map_coordinates

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from skimage.transform import radon, iradon
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════
st.set_page_config(page_title="CT Phantom QC", page_icon="⚕",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,[class*="css"]{font-family:'JetBrains Mono',monospace;}
.stApp{background:#060910;color:#d8e8ff;}
section[data-testid="stSidebar"]{background:#0e1520!important;border-right:1px solid #1e2a3a;}
[data-testid="metric-container"]{background:#111b2a;border:1px solid #1e2a3a;border-radius:8px;padding:10px!important;}
[data-testid="metric-container"] label{font-size:9px!important;letter-spacing:2px;color:#4a6080;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;}
.vitTitle{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;
  background:linear-gradient(135deg,#00e5ff,#7b61ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.secLabel{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:#3a5060;
  border-bottom:1px solid #1e2a3a;padding-bottom:5px;margin:14px 0 8px 0;}
.mathbox{background:#0a1520;border:1px solid #1e3a4a;border-radius:8px;
  padding:12px 16px;font-size:11px;color:#7ac8e0;line-height:1.8;margin:8px 0;}
.infobox{background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.2);
  border-radius:6px;padding:8px 12px;font-size:11px;color:#8ad8ef;margin:6px 0;}
.warnbox{background:rgba(255,152,0,.07);border:1px solid rgba(255,152,0,.25);
  border-radius:6px;padding:8px 12px;font-size:11px;color:#f5c060;margin:6px 0;}
.okbox{background:rgba(0,230,118,.06);border:1px solid rgba(0,230,118,.2);
  border-radius:6px;padding:8px 12px;font-size:11px;color:#70ffc0;margin:6px 0;}
div.stButton>button[kind="primary"]{background:linear-gradient(135deg,#00e5ff,#7b61ff)!important;
  color:#000!important;font-weight:700;border:none;}
div.stButton>button{border-radius:7px!important;font-family:'JetBrains Mono',monospace!important;}

/* ── Homepage tool cards ── */
.tool-card{
  position:relative;overflow:hidden;
  background:linear-gradient(145deg,#0e1825,#0a1320);
  border:1px solid #1a2a3a;border-radius:16px;
  padding:28px 24px 22px;cursor:pointer;
  transition:all .25s ease;
}
.tool-card:hover{border-color:#00e5ff55;transform:translateY(-3px);
  box-shadow:0 12px 40px rgba(0,229,255,.12);}
.tool-card-icon{font-size:36px;margin-bottom:12px;display:block;}
.tool-card-title{font-family:'Syne',sans-serif;font-weight:800;font-size:17px;
  color:#d8e8ff;margin-bottom:6px;letter-spacing:.3px;}
.tool-card-desc{font-size:10px;color:#4a7090;line-height:1.6;letter-spacing:.5px;}
.tool-card-tag{display:inline-block;font-size:8px;letter-spacing:2px;
  padding:2px 8px;border-radius:20px;margin-top:12px;font-weight:600;}
.tag-std{background:rgba(0,229,255,.1);color:#00e5ff;border:1px solid rgba(0,229,255,.25);}
.tag-adv{background:rgba(123,97,255,.1);color:#a78bff;border:1px solid rgba(123,97,255,.25);}
.tag-sim{background:rgba(255,152,0,.1);color:#ffb74d;border:1px solid rgba(255,152,0,.25);}
.tag-log{background:rgba(0,230,118,.1);color:#69ffb0;border:1px solid rgba(0,230,118,.25);}
.tool-card-accent{position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;}
.accent-cyan{background:linear-gradient(90deg,#00e5ff,#0096ff);}
.accent-purple{background:linear-gradient(90deg,#7b61ff,#b44fff);}
.accent-orange{background:linear-gradient(90deg,#ff9800,#ff5722);}
.accent-green{background:linear-gradient(90deg,#00e676,#00bcd4);}

/* ── Topbar nav ── */
.topbar{display:flex;align-items:center;gap:12px;
  padding:10px 20px;background:#0a0f18;
  border-bottom:1px solid #1a2a3a;margin-bottom:20px;
  border-radius:10px;}
.topbar-logo{font-family:'Syne',sans-serif;font-weight:800;font-size:16px;
  background:linear-gradient(135deg,#00e5ff,#7b61ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  white-space:nowrap;}
.topbar-sep{color:#1e2a3a;font-size:18px;}
.topbar-page{font-size:11px;color:#3a6080;letter-spacing:2px;text-transform:uppercase;}
.home-hero{text-align:center;padding:40px 0 30px;}
.home-title{font-family:'Syne',sans-serif;font-weight:800;font-size:42px;
  background:linear-gradient(135deg,#00e5ff 20%,#7b61ff 60%,#ff6eb4 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  line-height:1.1;margin-bottom:10px;}
.home-sub{font-size:12px;letter-spacing:4px;color:#3a6080;text-transform:uppercase;}
.home-divider{width:60px;height:2px;
  background:linear-gradient(90deg,#00e5ff,#7b61ff);
  margin:18px auto;border-radius:2px;}

/* ── pass/fail badges ── */
.pass-badge{display:inline-block;background:rgba(0,230,118,.15);border:1px solid rgba(0,230,118,.4);
  color:#70ffc0;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.fail-badge{display:inline-block;background:rgba(255,82,82,.15);border:1px solid rgba(255,82,82,.4);
  color:#ff8a80;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.metric-row{display:flex;gap:12px;margin:8px 0;}
.metric-item{background:#111b2a;border:1px solid #1e2a3a;border-radius:6px;
  padding:8px 14px;flex:1;text-align:center;}
.metric-val{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:#00e5ff;}
.metric-lbl{font-size:9px;letter-spacing:2px;color:#3a5060;text-transform:uppercase;}
.qc-card{background:#0e1520;border:1px solid #1e2a3a;border-radius:10px;padding:16px 20px;margin-bottom:10px;}
.qc-title{font-family:'Syne',sans-serif;font-weight:800;font-size:15px;
  background:linear-gradient(135deg,#00e5ff,#7b61ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;}
.qc-desc{font-size:10px;letter-spacing:1px;color:#3a5060;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════
CONTRAST_SET1 = [-10, -20, -50, -200, -500]   # HU
CONTRAST_SET2 = [-5, 0, 5]                    # HU

DIAMETERS_SET1 = [15]        # สำหรับ MTF
DIAMETERS_SET2 = [6, 5, 4, 3]  # สำหรับ CHO

# Placement ring: fraction of phantom radius
PLACEMENT_RINGS = {
    15: 0.25,
    6: 0.30,
    5: 0.48,
    4: 0.62,
    3: 0.74
}

# ══════════════════════════════════════════════════
#  COLOR THEME
# ══════════════════════════════════════════════════
DARK  = "#080b10"
SURF  = "#0e1520"
MUTED = "#3a5060"
ACC   = "#00e5ff"
ACC2  = "#7b61ff"

# ══════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════
def _init():
    defs = dict(
        slices=[], current=0, wl=0, ww=400,
        ui_section="viewer",
        uniformity_idx=[], inserted_set1=False, inserted_set2=False,
        mtf_result=None, cho_result=None, bead_slices=[], log=[],
        active_series=None, psf_info=None, nps_cache=None,
        pv_inserts=None,
        _psf_wire_centers=[], _psf_patch_half=7,
        psf_exp=True,

        qc_status={
            "uniformity": False,
            "nps": False,
            "pve": False,
            "psf": False
        },
        basic_geometry_result=None,
        basic_square_result=None,
        basic_linearity_result=None,
        basic_accuracy_result=None,
        basic_noise_result=None,
        basic_noise_baseline=None,
        basic_slice_result=None,
        current_page="home",

        nps_results=[], 
        linearity_idx=[], linearity_from=0, linearity_to=0
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

def slog(msg, lv="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    ic = {"info":"·","ok":"✓","warn":"⚠","error":"✗"}.get(lv,"·")
    st.session_state.log.append(f"[{ts}] {ic} {msg}")

def _get(ds, key, default=None):
    """Safe DICOM getter for common tags."""
    try:
        v = getattr(ds, key, default)
        # Convert pydicom types to python primitives
        if hasattr(v, "value"):
            v = v.value
        return v
    except Exception:
        return default

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _as_str(x, default=""):
    try:
        return str(x)
    except Exception:
        return default
# ══════════════════════════════════════════════════
#  DICOM LOADING
# ══════════════════════════════════════════════════
def parse_dicom(raw: bytes, name: str) -> dict | None:

    if not HAS_PYDICOM:
        return None

    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)

        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))

        hu = arr * slope + intercept

        ps = getattr(ds, "PixelSpacing", [0.5, 0.5])

        # ─────────────────────────────
        # Protocol tags
        # ─────────────────────────────

        kvp = _as_float(_get(ds, "KVP", None))

        ma = _as_float(_get(ds, "XRayTubeCurrent", None))
        ms = _as_float(_get(ds, "ExposureTime", None))
        mas = _as_float(_get(ds, "Exposure", None))

        if mas is None and (ma is not None and ms is not None):
            mas = ma * (ms / 1000.0)

        # Reconstruction kernel
        kernel = _as_str(_get(ds, "ConvolutionKernel", ""))

        if kernel == "":
            kernel = _as_str(_get(ds, "ReconstructionKernel", ""))

        if kernel == "":
            kernel = _as_str(_get(ds, "FilterType", ""))

        # Scanner information
        manu = _as_str(_get(ds, "Manufacturer", ""))
        model = _as_str(_get(ds, "ManufacturerModelName", ""))
        station = _as_str(_get(ds, "StationName", ""))

        series = _as_str(_get(ds, "SeriesDescription", ""))
        protocol_name = _as_str(_get(ds, "ProtocolName", ""))

        # Reconstruction algorithm tags
        recon_method = _as_str(_get(ds, "ReconstructionMethod", ""))
        iter_desc = _as_str(_get(ds, "IterativeReconstructionDescription", ""))
        ir_flag = _as_str(_get(ds, "IterativeReconstruction", ""))

        dlir_level = _as_str(_get(ds, "ReconstructionAlgorithm", ""))

        # ─────────────────────────────
        # Return dictionary
        # ─────────────────────────────

        return dict(
            name=name,
            ds=ds,

            hu_orig=hu.copy(),
            hu_mod=None,

            pixel_spacing=float(ps[0]),
            slice_thickness=float(getattr(ds, "SliceThickness", 1.0)),
            slice_location=float(getattr(ds, "SliceLocation", 0.0)),

            rows=int(ds.Rows),
            cols=int(ds.Columns),

            is_uniformity=False,
            lesions=[],

            # protocol info
            kvp=kvp,
            ma=ma,
            ms=ms,
            mas=mas,
            kernel=kernel,

            manufacturer=manu,
            model=model,
            station=station,

            series_desc=series,
            protocol_name=protocol_name,

            recon_method=recon_method,
            iter_desc=iter_desc,
            ir_flag=ir_flag,
            dlir_level=dlir_level
        )

    except Exception as e:
        slog(f"DICOM error [{name}]: {e}", "error")
        return None


def parse_image_fallback(raw: bytes, name: str) -> dict | None:
    """Fallback parser for PNG/JPG images (non-DICOM)."""
    try:
        from PIL import Image as _PIL
        img = _PIL.open(io.BytesIO(raw)).convert("L")
        arr = np.array(img, dtype=np.float32)
        hu = arr * (2000.0 / 255.0) - 1000.0
        rows, cols = hu.shape
        return dict(
            name=name, ds=None,
            hu_orig=hu.copy(), hu_mod=None,
            pixel_spacing=0.5, slice_thickness=1.0, slice_location=0.0,
            rows=rows, cols=cols,
            is_uniformity=False, lesions=[],
            kvp=None, ma=None, ms=None, mas=None,
            kernel="", manufacturer="", model="", station="",
            series_desc="", protocol_name="",
            recon_method="", iter_desc="", ir_flag="", dlir_level=""
        )
    except Exception as e:
        slog(f"Image fallback error [{name}]: {e}", "error")
        return None
    
def detect_marker_slice(hu):

    phantom = detect_phantom_circle(hu)

    if phantom is None:
        return False

    cx, cy, r = phantom

    # outer ring area
    r1 = r * 0.88
    r2 = r * 0.98

    h,w = hu.shape
    yy,xx = np.ogrid[:h,:w]

    dist = np.sqrt((xx-cx)**2+(yy-cy)**2)

    ring_mask = (dist>r1) & (dist<r2)

    ring_pixels = hu[ring_mask]

    # white tick threshold
    white_pixels = ring_pixels > 700

    ratio = np.sum(white_pixels)/len(ring_pixels)

    return ratio > 0.002

def scan_module_slices(slices):

    detected=[]

    for i,sl in enumerate(slices):

        hu=sl["hu_orig"]

        if detect_marker_slice(hu):

            detected.append(i)

    return detected
# ══════════════════════════════════════════════════
#  UNIFORMITY MODULE DETECTION
# ══════════════════════════════════════════════════
def find_phantom_circle(hu:np.ndarray) -> tuple:
    """
    Locate phantom boundary using radial threshold scan from image center.
    Returns (cx_px, cy_px, radius_px) of the phantom body.

    Strategy:
    - Threshold: find pixels significantly above air (> -500 HU)
    - Use centroid of thresholded mask as center
    - Measure radius as mean distance from centroid to edge
    """
    rows, cols = hu.shape
    # Phantom body: everything that is NOT air
    body = hu > -500
    if not np.any(body):
        return cols//2, rows//2, min(rows,cols)//3

    yy, xx = np.where(body)
    cx = float(xx.mean())
    cy = float(yy.mean())

    # Radius: 90th percentile of distances from centroid to body pixels
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    radius = float(np.percentile(dist, 90))
    return int(round(cx)), int(round(cy)), radius

def apply_window(img, wl, ww):
    low = wl - ww/2
    high = wl + ww/2
    img = np.clip(img, low, high)
    return (img - low) / (high - low + 1e-8)

def detect_phantom_circle(hu):
    """
    Alias for find_phantom_circle — returns (cx, cy, r).
    Used by insert_clock_lesions and detect_marker_slice.
    """
    return find_phantom_circle(hu)


def detect_bead(hu, thresh=600):
    """
    Detect a single bright bead (wire/bead phantom) in a slice.
    Returns (cx, cy) of brightest connected component, or None.
    """
    mask = (hu > thresh).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_val = -np.inf
    for i in range(1, nlab):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 <= area <= 200:
            cx = int(round(cent[i][0]))
            cy = int(round(cent[i][1]))
            peak = float(hu[cy, cx])
            if peak > best_val:
                best_val = peak
                best = (cx, cy)
    return best


def score_uniformity(sl:dict) -> tuple:
    """
    Score a slice for CTP486 uniformity module.
    Returns (total_score, breakdown_dict) for transparency.

    Simple, transparent criteria — no complex penalties:
    ─────────────────────────────────────────────────────
    C1 (30 pts): Inner-circle mean HU near water (0 ± 200)
    C2 (30 pts): Inner-circle std HU very low (< 60 HU)
    C3 (20 pts): Whole-phantom std low (< 80 HU)
    C4 (20 pts): Inner-circle pixel uniformity
                 (% pixels within ±50 HU of mean)
    ─────────────────────────────────────────────────────
    Max = 100 pts.  Threshold 70+ = likely uniformity.
    """
    hu   = sl["hu_orig"]
    rows, cols = hu.shape
    cx,cy,r_phantom  = find_phantom_circle(hu)

    pad = int(r_phantom *1.2)

    img_crop = hu[
        int(cy-pad):int(cy+pad),
        int(cx-pad):int(cx+pad)
    ]

    # Inner circle = 55% of phantom radius
    r_inner = r_phantom * 1.05
    yy2, xx2 = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dist2    = np.sqrt((xx2 - cx)**2 + (yy2 - cy)**2)
    inner_mask  = dist2 <= r_inner
    phantom_mask = dist2 <= r_phantom * 1.05  # whole phantom body

    bd = {}   # breakdown

    if not np.any(inner_mask):
        bd = {"C1_mean":0,"C2_inner_std":0,"C3_phantom_std":0,"C4_uniformity":0,
              "mu":0,"inner_std":0,"phantom_std":0,"frac50":0}
        return 0.0, bd

    inner_hu   = hu[inner_mask]
    phantom_hu = hu[phantom_mask] if np.any(phantom_mask) else inner_hu

    mu          = float(inner_hu.mean())
    inner_std   = float(inner_hu.std())
    phantom_std = float(phantom_hu.std())
    # texture check (low contrast module has higher gradients)
    gx = np.gradient(hu, axis=1)
    gy = np.gradient(hu, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_score = float(np.mean(grad_mag[inner_mask]))
    
    frac50      = float(np.mean(np.abs(inner_hu - mu) < 50))

    # C1: mean near water
    if abs(mu) <= 50:
        c1 = 30.0
    elif abs(mu) <= 200:
        c1 = 30.0 * max(0, 1 - (abs(mu) - 50) / 150)
    else:
        c1 = 0.0

    # C2: inner std (lower = more uniform = more likely uniformity module)
    if inner_std < 20:
        c2 = 30.0
    elif inner_std < 60:
        c2 = 30.0 * (1 - (inner_std - 20) / 40)
    else:
        c2 = 0.0

    # C3: whole phantom std
    if phantom_std < 50:
        c3 = 20.0
    elif phantom_std < 120:
        c3 = 20.0 * (1 - (phantom_std - 50) / 50)
    else:
        c3 = 0.0

    # C4: fraction of pixels within ±50 HU of mean
    c4 = frac50 * 20.0

    # C5: texture penalty
    if grad_score < 20:
        c5 = 10
    elif grad_score < 40:
        c5 = 10 * (1 - (grad_score-20)/20)
    else:
        c5 = 0

    total = c1 + c2 + c3 + c4 + c5
    bd = {
        "C1_mean":     round(c1, 1),
        "C2_inner_std":round(c2, 1),
        "C3_phantom_std": round(c3, 1),
        "C4_uniformity":  round(c4, 1),
        "C5_texture": round(c5,1),
        "grad": round(grad_score,1),
        "mu":          round(mu, 1),
        "inner_std":   round(inner_std, 1),
        "phantom_std": round(phantom_std, 1),
        "frac50":      round(frac50 * 100, 1),
    }
    return round(total, 1), bd

def detect_uniformity(sl:dict, threshold:float=80.0) -> bool:
    score, bd = score_uniformity(sl)

    if abs(bd["mu"]) > 20:
        return False

    return score >= threshold

def rank_uniformity_slices(slices:list) -> list:
    """Returns list of (index, score, breakdown) sorted by score desc."""
    scored = []
    for i, s in enumerate(slices):
        sc, bd = score_uniformity(s)
        scored.append((i, sc, bd))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
# ══════════════════════════════════════════════════
#  PSF BEAD DETECTION
# ══════════════════════════════════════════════════

def detect_psf_beads(slices, start_slice, end_slice, thresh, patch_half):
    
    bead_slices = []

    for i in range(start_slice, end_slice + 1):

        sl = slices[i]
        hu = sl["hu_orig"]

        # normalize for OpenCV
        img = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # threshold bright pixels
        mask = hu > thresh

        mask = mask.astype(np.uint8)

        # remove noise
        mask = cv2.medianBlur(mask, 5)

        # connected components
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask)

        beads = []

        for j in range(1, nlab):

            area = stats[j, cv2.CC_STAT_AREA]

            # bead ต้องเล็ก
            if 3 < area < 150:

                cx = int(cent[j][0])
                cy = int(cent[j][1])

                beads.append((cx, cy))

        if beads:

            sl["beads"] = beads
            bead_slices.append(i)

    return bead_slices

def image_domain_insert(hu:np.ndarray, cx:int, cy:int,
                        r_px:float, contrast_hu:float)->np.ndarray:
    """
    Post-reconstruction lesion insertion

    I_new(x,y) = I(x,y) + C · (Disk(x,y) * PSF)

    Disk = binary circular object
    PSF  = Gaussian blur to simulate CT system resolution
    """

    rows, cols = hu.shape

    yy, xx = np.ogrid[:rows, :cols]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)

    lesion = (dist <= r_px + 0.5).astype(np.float32)

    # simulate CT system blur
    lesion = gaussian_filter(lesion, sigma=0.5)

    hu_new = hu.copy()
    hu_new += lesion * contrast_hu

    return hu_new.astype(np.float32)

def extract_bead_candidates(hu, thresh=150, patch_half=3):
    """
    Detect bright bead candidates from one slice.
    Return list of dicts: [{"cx":..., "cy":..., "bbox":(...)}]
    """
    mask = (hu > thresh).astype(np.uint8)

    # denoise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)

    beads = []
    h, w = hu.shape

    for i in range(1, nlab):
        area = stats[i, cv2.CC_STAT_AREA]

        # bead ควรเป็นวัตถุเล็ก
        if 1 <= area <= 100:
            cx = int(round(cent[i][0]))
            cy = int(round(cent[i][1]))

            x1 = max(0, cx - patch_half)
            x2 = min(w, cx + patch_half + 1)
            y1 = max(0, cy - patch_half)
            y2 = min(h, cy + patch_half + 1)

            # ต้องได้ patch ครบ
            if (x2 - x1) == (2 * patch_half + 1) and (y2 - y1) == (2 * patch_half + 1):
                beads.append({
                    "cx": cx,
                    "cy": cy,
                    "bbox": (x1, y1, x2, y2),
                    "area": int(area)
                })

    return beads


def crop_patch(img, cx, cy, patch_half):
    x1 = cx - patch_half
    x2 = cx + patch_half + 1
    y1 = cy - patch_half
    y2 = cy + patch_half + 1
    return img[y1:y2, x1:x2].copy()


def refine_centroid_subpixel(patch):
    """
    หา centroid แบบ intensity-weighted
    """
    p = patch.astype(np.float64)
    p = p - np.min(p)
    if np.sum(p) <= 0:
        cy = (p.shape[0] - 1) / 2
        cx = (p.shape[1] - 1) / 2
        return cx, cy

    cy, cx = center_of_mass(p)
    return float(cx), float(cy)


def shift_patch_subpixel(patch, shift_x, shift_y):
    """
    เลื่อน patch แบบ subpixel
    """
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(
        patch.astype(np.float32),
        M,
        (patch.shape[1], patch.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return shifted


def build_mean_psf_from_beads(hu, beads, patch_half=10):
    """
    crop patch ทุก bead, align centroid เข้ากลาง, average เป็น mean PSF
    """
    patches = []
    target_c = patch_half  # center index

    for b in beads:
        cx, cy = b["cx"], b["cy"]
        patch = crop_patch(hu, cx, cy, patch_half)

        if patch.shape != (2 * patch_half + 1, 2 * patch_half + 1):
            continue

        sub_cx, sub_cy = refine_centroid_subpixel(patch)

        shift_x = target_c - sub_cx
        shift_y = target_c - sub_cy

        patch_aligned = shift_patch_subpixel(patch, shift_x, shift_y)
        patches.append(patch_aligned)

    if len(patches) == 0:
        return None, []

    mean_psf = np.mean(np.stack(patches, axis=0), axis=0)
    return mean_psf, patches


def radial_average_2d(img2d):
    h, w = img2d.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = np.floor(r).astype(int)

    prof = []
    for rr in range(r_int.max() + 1):
        vals = img2d[r_int == rr]
        if vals.size > 0:
            prof.append(vals.mean())
        else:
            prof.append(np.nan)

    return np.array(prof, dtype=float)

def compute_fwhm(profile, pixel_spacing):
    """
    FWHM จาก line profile ที่มี peak เดียว
    """
    y = profile.astype(float)
    y = y - np.min(y)

    ymax = np.max(y)
    if ymax <= 0:
        return np.nan

    half = ymax / 2.0
    idx = np.where(y >= half)[0]

    if len(idx) < 2:
        return np.nan

    fwhm_px = idx[-1] - idx[0]
    return fwhm_px * pixel_spacing


def find_mtf_freq(mtf, freq, level):
    """
    หา frequency ที่ MTF ตัด level เช่น 0.5 หรือ 0.1
    """
    mtf = np.asarray(mtf)
    freq = np.asarray(freq)

    valid = np.isfinite(mtf) & np.isfinite(freq)
    mtf = mtf[valid]
    freq = freq[valid]

    if len(mtf) < 2:
        return np.nan

    for i in range(len(mtf) - 1):
        y1, y2 = mtf[i], mtf[i + 1]
        if (y1 >= level and y2 <= level) or (y1 <= level and y2 >= level):
            x1, x2 = freq[i], freq[i + 1]
            if y2 == y1:
                return x1
            return x1 + (level - y1) * (x2 - x1) / (y2 - y1)

    return np.nan


def analyze_psf_from_mean_psf(mean_psf, pixel_spacing):
    """
    Bead / point method:
    PSF -> 2D FFT -> 2D MTF -> radial average
    """

    psf = mean_psf.astype(np.float64).copy()

    # -------------------------
    # 1) background subtraction
    # ใช้ median ของ border
    # -------------------------
    border = np.concatenate([
        psf[0, :], psf[-1, :],
        psf[:, 0], psf[:, -1]
    ])
    bg = np.median(border)

    psf_bg = psf - bg
    psf_bg[psf_bg < 0] = 0

    if np.sum(psf_bg) <= 0:
        return None

    # -------------------------
    # 2) normalize PSF
    # -------------------------
    psf_norm2d = psf_bg / np.sum(psf_bg)

    # -------------------------
    # 3) line profile for display only
    # -------------------------
    h, w = psf_norm2d.shape
    cy = h // 2
    cx = w // 2

    line_x = psf_norm2d[cy, :].copy()
    if np.max(line_x) > 0:
        line_x_disp = line_x / np.max(line_x)
    else:
        line_x_disp = line_x

    # radial PSF profile for display
    rad_psf = radial_profile(psf_norm2d, center=(cx, cy), max_r=min(cx, cy))

    # -------------------------
    # 4) 2D FFT -> 2D MTF
    # -------------------------
    otf2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_norm2d)))
    mtf2 = np.abs(otf2)

    dc = mtf2[cy, cx]
    if dc <= 0:
        return None

    mtf2 = mtf2 / dc

    # -------------------------
    # 5) radial average of 2D MTF
    # -------------------------
    mtf_rad = radial_average_2d(mtf2)

    # spatial frequency axis (mm^-1)
    # df = 1 / (N * pixel_spacing)
    df = 1.0 / (psf_norm2d.shape[0] * pixel_spacing)
    freq = np.arange(len(mtf_rad)) * df

    # เอาเฉพาะช่วงที่สมเหตุสมผลถึง Nyquist
    nyquist = 1.0 / (2.0 * pixel_spacing)
    valid = freq <= nyquist
    freq = freq[valid]
    mtf_rad = mtf_rad[valid]

    # -------------------------
    # 6) FWHM from center line (display metric)
    # -------------------------
    fwhm_mm = compute_fwhm(line_x_disp, pixel_spacing)

    mtf50 = find_mtf_freq(mtf_rad, freq, 0.5)
    mtf10 = find_mtf_freq(mtf_rad, freq, 0.1)

    return {
        "psf_2d": psf_norm2d,
        "mtf_2d": mtf2,
        "line_profile": line_x_disp,
        "radial_profile": rad_psf,
        "freq": freq,
        "mtf": mtf_rad,
        "fwhm_mm": fwhm_mm,
        "mtf50": mtf50,
        "mtf10": mtf10,
    }

def plot_psf_analysis(slice_img, beads, mean_psf, result, patch_half, wl=None, ww=None):
    fig = plt.figure(figsize=(14, 8))

    # 1) original slice + boxes
    ax1 = plt.subplot(2, 3, 1)
    disp = slice_img.copy()

    if wl is not None and ww is not None:
        low = wl - ww / 2
        high = wl + ww / 2
        disp = np.clip(disp, low, high)
        disp = (disp - low) / (high - low + 1e-12)

    ax1.imshow(disp, cmap="gray")
    for b in beads:
        x1, y1, x2, y2 = b["bbox"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor="red", linewidth=1)
        ax1.add_patch(rect)
        ax1.plot(b["cx"], b["cy"], "y+", ms=6)
    ax1.set_title(f"Detected beads: {len(beads)}")
    ax1.axis("off")

    # 2) mean PSF
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(result["psf_2d"], cmap="hot")
    ax2.set_title("Mean PSF")
    ax2.axis("off")

    # 3) center line PSF
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(result["line_profile"], lw=1.5)
    ax3.set_title("Center Line PSF")
    ax3.set_xlabel("Pixel")
    ax3.set_ylabel("Normalized intensity")
    ax3.grid(alpha=0.3)

    # 4) radial PSF
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(result["radial_profile"], lw=1.5)
    ax4.set_title("Radial PSF Profile")
    ax4.set_xlabel("Radius (px)")
    ax4.set_ylabel("Mean intensity")
    ax4.grid(alpha=0.3)

     # -------------------------
    # 5) radial average of 2D MTF
    # -------------------------
    mtf_rad = radial_average_2d(mtf2)

    # spatial frequency axis (mm^-1)
    # df = 1 / (N * pixel_spacing)
    df = 1.0 / (psf_norm2d.shape[0] * pixel_spacing)
    freq = np.arange(len(mtf_rad)) * df

    # เอาเฉพาะช่วงที่สมเหตุสมผลถึง Nyquist
    nyquist = 1.0 / (2.0 * pixel_spacing)
    valid = freq <= nyquist
    freq = freq[valid]
    mtf_rad = mtf_rad[valid]

    # -------------------------
    # 6) FWHM from center line (display metric)
    # -------------------------
    fwhm_mm = compute_fwhm(line_x_disp, pixel_spacing)

    mtf50 = find_mtf_freq(mtf_rad, freq, 0.5)
    mtf10 = find_mtf_freq(mtf_rad, freq, 0.1)

    return {
        "psf_2d": psf_norm2d,
        "mtf_2d": mtf2,
        "line_profile": line_x_disp,
        "radial_profile": rad_psf,
        "freq": freq,
        "mtf": mtf_rad,
        "fwhm_mm": fwhm_mm,
        "mtf50": mtf50,
        "mtf10": mtf10,
    }

def _make_circle_mask(size:int)->np.ndarray:
    cx=cy=(size-1)/2.0
    yy,xx=np.ogrid[:size,:size]
    return (xx-cx)**2+(yy-cy)**2<=cx**2

def _make_circle_mask_rect(rows:int,cols:int)->np.ndarray:
    cx=(cols-1)/2.0;cy=(rows-1)/2.0;r=min(cx,cy)
    yy,xx=np.ogrid[:rows,:cols]
    return (xx-cx)**2+(yy-cy)**2<=r**2

def _crop_sq(arr,rows,cols,pad_r,pad_c):
    h,w=arr.shape
    r0=min(pad_r,h);c0=min(pad_c,w)
    r1=min(r0+rows,h);c1=min(c0+cols,w)
    cr=arr[r0:r1,c0:c1]
    if cr.shape==(rows,cols):return cr
    out=np.zeros((rows,cols),dtype=arr.dtype)
    out[:cr.shape[0],:cr.shape[1]]=cr
    return out

def projection_domain_insert(hu:np.ndarray,cx:int,cy:int,
                              r_px:float,contrast_hu:float,
                              n_angles:int=360)->np.ndarray:
    """
    Hybrid Radon lesion insertion.
    1. Forward-project patient image (Radon)
    2. Add analytical lesion sinogram (calibrated scale)
    3. Reconstruct with FBP (ramp filter)
    Noise texture inside lesion follows FBP PSF — more realistic than image-domain.
    Falls back to image_domain_insert if scikit-image unavailable.
    """
    if not HAS_SKIMAGE:
        return image_domain_insert(hu,cx,cy,r_px,contrast_hu)

    from skimage.transform import radon as _radon,iradon as _iradon

    rows,cols=hu.shape
    size=max(rows,cols)
    if size%2!=0:size+=1
    pad_r=(size-rows)//2;pad_c=(size-cols)//2

    hu_sq=np.pad(hu,((pad_r,size-rows-pad_r),(pad_c,size-cols-pad_c)),
                 mode="edge").astype(np.float64)
    cx_sq=cx+pad_c;cy_sq=cy+pad_r

    cmask=_make_circle_mask(size)
    hu_sq[~cmask]=0.0

    angles=np.linspace(0.,180.,n_angles,endpoint=False)
    sino_p=_radon(hu_sq,theta=angles,circle=True)

    disk=np.zeros((size,size),dtype=np.float64)
    yy,xx=np.ogrid[:size,:size]
    lmask=((xx-cx_sq)**2+(yy-cy_sq)**2<=r_px**2)&cmask
    disk[lmask]=1.0
    sino_l=_radon(disk,theta=angles,circle=True)

    recon_unit=_iradon(sino_l,theta=angles,filter_name="ramp",circle=True)
    mean_unit=float(recon_unit[lmask].mean()) if lmask.any() else 1.0
    if abs(mean_unit)<1e-9:mean_unit=1.0
    scale=contrast_hu/mean_unit

    hu_recon=_iradon(sino_p+sino_l*scale,theta=angles,
                     filter_name="ramp",circle=True)
    hu_new=_crop_sq(hu_recon,rows,cols,pad_r,pad_c).astype(np.float32)

    orig_circ=_make_circle_mask_rect(rows,cols)
    hu_new[~orig_circ]=hu[~orig_circ]
    return hu_new

# ══════════════════════════════════════════════════
#  LESION GRID PLACEMENT
# ══════════════════════════════════════════════════
def lesion_centers_ring(n:int, ring_r_px:float, cx:float, cy:float):
    """Evenly space n lesion centers on a ring."""
    return [(cx + ring_r_px*math.cos(2*math.pi*i/n),
             cy + ring_r_px*math.sin(2*math.pi*i/n)) for i in range(n)]

def insert_clock_lesions(sl, contrast_hu, diam_mm, method="image"):
    """
    Insert 5 lesions at clock positions (12,3,6,9,center).
    method: "image"      — image-domain (fast, Gaussian blur)
            "projection" — projection-domain Hybrid Radon (slower, realistic PSF)
    """
    hu = sl["hu_orig"].copy()
    ps = sl["pixel_spacing"]
    phantom = detect_phantom_circle(hu)

    if phantom is not None:

        cx, cy, r = phantom

        inner_r = int(r * 0.6)

        h, w = hu.shape
        yy, xx = np.ogrid[:h, :w]

        mask = (xx - cx)**2 + (yy - cy)**2 <= inner_r**2

        hu_mask = hu.copy()
        hu_mask[~mask] = hu.mean()

    else:

        hu_mask = hu
    cx, cy, r_phantom = find_phantom_circle(hu)
    ring = r_phantom * 0.40
    r_px = (diam_mm / 2) / ps

    centers = [
        (cx, cy - ring),  # 12
        (cx + ring, cy),  # 3
        (cx, cy + ring),  # 6
        (cx - ring, cy),  # 9
        (cx, cy)          # center
    ]

    sl["lesions"] = []

    if method == "projection":
        # Projection-domain: insert all lesions cumulatively
        # (each lesion baked into sinogram of accumulated image)
        for i, (lx, ly) in enumerate(centers):
            hu = projection_domain_insert(
                hu, int(lx), int(ly), r_px, contrast_hu)
            sl["lesions"].append({
                "cx": int(lx), "cy": int(ly),
                "r_px": r_px, "diam_mm": diam_mm,
                "contrast": contrast_hu,
                "position": ["12","3","6","9","center"][i]
            })
    else:
        # Image-domain (original, fast)
        for i, (lx, ly) in enumerate(centers):
            hu = image_domain_insert(
                hu, int(lx), int(ly), r_px, contrast_hu)
            sl["lesions"].append({
                "cx": int(lx), "cy": int(ly),
                "r_px": r_px, "diam_mm": diam_mm,
                "contrast": contrast_hu,
                "position": ["12","3","6","9","center"][i]
            })

    sl["hu_mod"] = hu
# ══════════════════════════════════════════════════
#  WINDOWING & DISPLAY
# ══════════════════════════════════════════════════
def wl_ww(hu:np.ndarray, wl:float, ww:float)->np.ndarray:
    lo,hi = wl-ww/2, wl+ww/2
    return np.clip((hu-lo)/(hi-lo)*255,0,255).astype(np.uint8)

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches=None,
        pad_inches=0
    )
    buf.seek(0)
    return buf.read()

def dark_style():
    plt.rcParams.update({
        "figure.facecolor":DARK,"axes.facecolor":SURF,
        "axes.edgecolor":"#1e2a3a","axes.labelcolor":"#d8e8ff",
        "xtick.color":MUTED,"ytick.color":MUTED,
        "text.color":"#d8e8ff","grid.color":"#151e28",
        "grid.linewidth":0.5,"font.family":"monospace"})

def render_slice(sl:dict, wl:float, ww_val:float,
                 show_lesions=True, highlight_set=None)->bytes:

    dark_style()

    # ── image data ──
    hu = sl.get("hu_mod")
    if hu is None:
        hu = sl.get("hu_orig")

    img = wl_ww(hu, wl, ww_val)

    fig, ax = plt.subplots(figsize=(5,5), facecolor=DARK)

    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")

    # ── draw lesion overlay ──
    lesions = sl.get("lesions", [])

    if show_lesions and lesions:

        for l in lesions:

            cx = l.get("cx")
            cy = l.get("cy")
            r  = l.get("r_px")

            if cx is None:
                continue

            circ = Circle(
                (cx, cy),
                r,
                fill=False,
                edgecolor="#ff5252",
                lw=1.4
            )

            ax.add_patch(circ)

            # label lesion
            pos = l.get("position")
            if pos:
                ax.text(
                cx,
                cy,
                pos,
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.6, pad=1)
            )
    # ── title ──
    name = sl.get("name","Slice")

    ax.set_title(
        f"{name} | WL{wl}/WW{ww_val}",
        fontsize=7,
        color=MUTED,
        pad=3
    )

    plt.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02)

    data = fig2img(fig)

    plt.close(fig)

    return data

def render_bead_preview(idx, bead):

    sl = st.session_state.slices[idx]

    hu = sl["hu_orig"]

    x,y,r = bead

    img = wl_ww(hu,0,400)

    fig,ax = plt.subplots(figsize=(4,4))

    ax.imshow(img,cmap="gray")

    rect = Rectangle(
        (x-20,y-20),
        40,
        40,
        fill=False,
        color="cyan",
        linestyle="--",
        linewidth=1
    )

    ax.add_patch(rect)

    ax.plot(x,y,"r+")

    ax.set_title(f"Slice {idx}")

    ax.axis("off")

    return fig2img(fig)

def insert_lesions_all_uniformity(contrast_hu, diam_mm):

    if "uniformity_idx" not in st.session_state:
        st.warning("Run Uniformity detection first")
        return

    for idx in st.session_state.uniformity_idx:

        sl = st.session_state.slices[idx]

        insert_clock_lesions(
            sl,
            contrast_hu,
            diam_mm
        )

def detect_single_bead(hu, thresh=150):

    hu = np.nan_to_num(hu)

    mask = hu > thresh

    if np.sum(mask) == 0:
        return None

    y, x = np.where(mask)

    idx = np.argmax(hu[y, x])

    cy = y[idx]
    cx = x[idx]

    return cx, cy

def extract_bead_patch(hu, cx, cy, half):

    y1 = cy - half
    y2 = cy + half + 1
    x1 = cx - half
    x2 = cx + half + 1

    return hu[y1:y2, x1:x2]

def build_psf_from_slices(slices, slice_range, thresh, patch_half):

    patches = []

    for i in slice_range:

        hu = slices[i]["hu_orig"]

        bead = detect_single_bead(hu, thresh)

        if bead is None:
            continue

        cx, cy = bead

        patch = extract_bead_patch(hu, cx, cy, patch_half)

        if patch.shape == (2*patch_half+1, 2*patch_half+1):
            patches.append(patch)

    if len(patches) == 0:
        return None

    psf = np.mean(patches, axis=0)

    return psf

def analyze_psf(psf, pixel_spacing):

    import numpy as np

    center = psf.shape[0] // 2

    line = psf[center]

    line = line - np.min(line)
    line = line / np.max(line)

    lsf = line

    win = np.hanning(len(lsf))

    lsf = lsf * win

    mtf = np.abs(np.fft.fft(lsf))
    mtf = mtf / np.max(mtf)

    freq = np.fft.fftfreq(len(lsf), d=pixel_spacing)

    half = len(freq)//2

    return freq[:half], mtf[:half]
# ══════════════════════════════════════════════════
#  MTF / TTF ANALYSIS
# ══════════════════════════════════════════════════
def compute_esf_radial(hu:np.ndarray, cx:int, cy:int, r_px:float,
                        half_w_px:int=None)->tuple:
    """
    Oversampled radial ESF from circular lesion edge.

    Math:
    ──────────────────────────────────────────────────────────
    ESF(r) = mean HU at radial distance r from lesion center
    LSF(r) = d/dr [ESF(r)]   (gradient = line spread function)
    MTF(f) = |FFT{LSF(r)}| / |FFT{LSF}|_{f=0}

    TTF (Task Transfer Function) = same as MTF but measured
    at a specific contrast level, showing how contrast
    modulates with spatial frequency.
    ──────────────────────────────────────────────────────────
    """
    if half_w_px is None: half_w_px = int(r_px*2.5)+5
    rows,cols = hu.shape
    yy,xx = np.ogrid[:rows,:cols]
    dist  = np.sqrt((xx-cx)**2+(yy-cy)**2)

    # Radial bin: 0.25 px resolution
    r_min = max(0, r_px - half_w_px)
    r_max = r_px + half_w_px
    bins  = np.arange(r_min, r_max, 0.25)
    esf   = []
    for i in range(len(bins)-1):
        m = (dist>=bins[i])&(dist<bins[i+1])
        esf.append(float(hu[m].mean()) if np.any(m) else np.nan)

    esf   = np.array(esf)
    r_mid = (bins[:-1]+bins[1:])/2

    # Remove NaN
    valid = ~np.isnan(esf)
    r_mid, esf = r_mid[valid], esf[valid]
    return r_mid, esf

def logistic_fit(r, esf):
    """Fit logistic (Fermi) function to ESF."""
    def fermi(x, A, B, x0, k):
        return A/(1+np.exp(-k*(x-x0)))+B
    try:
        A0 = esf.max()-esf.min()
        B0 = esf.min()
        x0 = float(r[len(r)//2])
        p0 = [A0, B0, x0, 0.5]
        popt,_ = curve_fit(fermi,r,esf,p0=p0,maxfev=8000)
        return fermi(r,*popt), popt
    except:
        return esf, None

def compute_mtf(esf_fitted:np.ndarray, bin_size_mm:float)->tuple:
    """LSF → FFT → MTF"""
    lsf  = np.gradient(esf_fitted)
    lsf -= lsf.mean()
    # Zero-pad
    N    = len(lsf)
    lsf_pad = np.pad(lsf,(N,N))
    mtf  = np.abs(fft(lsf_pad))
    mtf /= (mtf[0]+1e-12)
    freq = fftfreq(len(lsf_pad), d=bin_size_mm)
    pos  = freq>=0
    return freq[pos], mtf[pos]

def run_mtf_analysis(sl:dict, lesion:dict)->dict:
    hu  = sl["hu_mod"] if sl["hu_mod"] is not None else sl["hu_orig"]
    ps  = sl["pixel_spacing"]
    r,esf = compute_esf_radial(hu, lesion["cx"], lesion["cy"], lesion["r_px"])
    esf_fit, popt = logistic_fit(r, esf)
    freq, mtf = compute_mtf(esf_fit, bin_size_mm=0.25*ps)

    # f50, f10
    f50=f10=None
    for i in range(len(mtf)-1):
        if mtf[i]>=0.5>mtf[i+1] and f50 is None:
            f50 = float(np.interp(0.5,[mtf[i+1],mtf[i]],[freq[i+1],freq[i]]))
        if mtf[i]>=0.1>mtf[i+1] and f10 is None:
            f10 = float(np.interp(0.1,[mtf[i+1],mtf[i]],[freq[i+1],freq[i]]))

    return dict(r=r, esf=esf, esf_fit=esf_fit, popt=popt,
                freq=freq, mtf=mtf, f50=f50, f10=f10, ps=ps,
                diam_mm=lesion["diam_mm"], contrast=lesion["contrast"])

def plot_mtf(result:dict)->bytes:
    dark_style()
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    # ESF
    ax=axes[0]
    ax.plot(result["r"]*result["ps"], result["esf"],
            "o", color=MUTED, ms=2.5, alpha=0.6, label="ESF data")
    ax.plot(result["r"]*result["ps"], result["esf_fit"],
            color=ACC, lw=1.5, label="Logistic fit")
    ax.set_xlabel("r (mm)"); ax.set_ylabel("HU"); ax.set_title("Edge Spread Function (ESF)")
    ax.legend(fontsize=7); ax.grid(True)

    # MTF
    ax2=axes[1]
    f,m = result["freq"], result["mtf"]
    valid = f<3.0
    ax2.plot(f[valid], m[valid], color=ACC2, lw=1.5,
             label=f"MTF Ø{result['diam_mm']}mm {result['contrast']} HU")
    ax2.axhline(0.5,color="#ff9800",lw=0.8,ls="--",label=f"f₅₀={result['f50']:.3f}" if result["f50"] else "")
    ax2.axhline(0.1,color="#ff5252",lw=0.8,ls=":",label=f"f₁₀={result['f10']:.3f}" if result["f10"] else "")
    if result["f50"]: ax2.axvline(result["f50"],color="#ff9800",lw=0.6,ls="--",alpha=0.5)
    if result["f10"]: ax2.axvline(result["f10"],color="#ff5252",lw=0.6,ls=":",alpha=0.5)
    ax2.set_xlabel("Spatial frequency (mm⁻¹)"); ax2.set_ylabel("MTF/TTF")
    ax2.set_title("Modulation Transfer Function (MTF / TTF)")
    ax2.set_ylim(-0.05,1.1); ax2.legend(fontsize=7); ax2.grid(True)

    plt.tight_layout(pad=0.8)
    data=fig2img(fig); plt.close(fig); return data

            # ✅ FIX 4: หา MTF50/MTF10 โดยหา index แรกที่ MTF ลดลงถึง threshold
            #           แทน np.interp ที่ผิดพลาดเมื่อ array ไม่ monotonic
def make_circular_mask(shape, cx, cy, radius_px):
    h, w = shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (rr <= radius_px).astype(np.float32)


def normalize_kernel(psf):
    psf = np.array(psf, dtype=np.float32)
    # 1) subtract background (median ของขอบ patch)
    border = np.concatenate([
        psf[0,:], psf[-1,:], psf[:,0], psf[:,-1]
    ])
    bg = float(np.median(border))
    psf = psf - bg
    psf[psf < 0] = 0
    # 2) normalize
    s = psf.sum()
    if s <= 0:
        # fallback → Gaussian sigma=1px
        from scipy.ndimage import gaussian_filter
        g = np.zeros_like(psf); g[psf.shape[0]//2, psf.shape[1]//2] = 1.0
        psf = gaussian_filter(g, sigma=1.0)
        s = psf.sum()
    return psf / s

def fft_convolve2d_same(image, kernel):
    """
    2D convolution — ใช้ scipy.signal.fftconvolve พร้อม reflect padding
    เพื่อป้องกัน edge/wrap-around artifact
    """
    from scipy.signal import fftconvolve
    image  = np.asarray(image,  dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # reflect pad ก่อน convolve
    image_pad = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")

    conv = fftconvolve(image_pad, kernel, mode="valid")

    # trim ให้ได้ขนาดเดิม
    h, w = image.shape
    oh, ow = conv.shape
    y0 = (oh - h) // 2
    x0 = (ow - w) // 2
    return conv[y0:y0+h, x0:x0+w].astype(np.float32)


# =========================================================
# PVE model
# =========================================================
def pve_contrast_scaling(diam_mm, pixel_spacing_mm, slice_thickness_mm=None):
    """
    Partial Volume Effect (PVE) contrast scaling.

    สูตร:
        PVE_xy  = erf( d / (sqrt(2) * sigma_xy) )
        PVE_z   = erf( d_z / (sqrt(2) * sigma_z) )   ← ถ้ามี slice_thickness
        scale   = PVE_xy * PVE_z

    โดย sigma_xy = pixel_spacing / (2*sqrt(2*ln2)) ≈ pixel_spacing * 0.4247
        (แปลง FWHM = pixel_spacing → sigma ของ Gaussian voxel profile)

    เมื่อ lesion เล็กกว่า voxel → scale ต่ำ (contrast ลดลงมาก)
    เมื่อ lesion ใหญ่กว่า voxel → scale → 1.0
    """
    from scipy.special import erf as _erf

    # sigma ของ in-plane voxel profile (Gaussian approximation)
    sigma_xy = pixel_spacing_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # PVE in-plane: fraction of lesion inside one voxel column
    r_mm = diam_mm / 2.0
    pve_xy = float(_erf(r_mm / (np.sqrt(2.0) * sigma_xy)))
    pve_xy = np.clip(pve_xy, 0.0, 1.0)

    # PVE through-plane
    if slice_thickness_mm is not None and slice_thickness_mm > 0:
        sigma_z = slice_thickness_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        pve_z   = float(_erf(r_mm / (np.sqrt(2.0) * sigma_z)))
        pve_z   = np.clip(pve_z, 0.0, 1.0)
    else:
        pve_z = 1.0

    scale = float(pve_xy * pve_z)
    scale = float(np.clip(scale, 0.01, 1.0))
    return scale


# =========================================================
# NPS-like correlated noise
# =========================================================
def generate_correlated_noise(shape, noise_sd, beta=2.0, seed=None,
                              nps_2d=None):
    """
    สร้าง correlated noise โดยใช้ NPS (Noise Power Spectrum) เป็น filter

    วิธี 1 — ถ้ามี nps_2d (วัดจากภาพจริง):
        noise = IFFT( FFT(white_noise) * sqrt(NPS_2D) )
        → texture ตรงกับ CT จริง

    วิธี 2 — ถ้าไม่มี nps_2d (ใช้ power-law approximation):
        Power(f) ~ 1 / f^beta
        beta=0 → white noise
        beta=2 → CT-like correlated noise

    ทั้งสองวิธีปรับ SD ให้ตรง noise_sd (HU) ด้วย rescale
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    white = rng.normal(0, 1, size=(h, w)).astype(np.float32)
    W = fft2(white)

    if nps_2d is not None:
        # ── วิธี 1: ใช้ NPS วัดจริง ──
        nps = np.asarray(nps_2d, dtype=np.float32)
        if nps.shape != (h, w):
            from scipy.ndimage import zoom as _zoom
            nps = _zoom(nps, (h / nps.shape[0], w / nps.shape[1]))
            nps = np.clip(nps, 0, None)
        nps_shift = np.fft.ifftshift(nps)
        filt = np.sqrt(nps_shift + 1e-12)
    else:
        # ── วิธี 2: power-law NPS ──
        fy = np.fft.fftfreq(h)
        fx = np.fft.fftfreq(w)
        FX, FY = np.meshgrid(fx, fy)
        FR = np.sqrt(FX**2 + FY**2)
        filt = np.ones_like(FR, dtype=np.float32)
        nonzero = FR > 0
        filt[nonzero] = 1.0 / (FR[nonzero] ** (beta / 2.0))
        filt[~nonzero] = 0.0

    noise = np.real(ifft2(W * filt))
    noise = noise - noise.mean()
    std = noise.std()
    if std > 0:
        noise = noise / std

    return noise * float(noise_sd)


# =========================================================
# Main image-domain lesion insertion
# =========================================================
def insert_lesion_image_domain(
    background_hu,
    cx,
    cy,
    diam_mm,
    contrast_hu,
    pixel_spacing_mm,
    psf,
    slice_thickness_mm=None,
    add_noise=False,
    noise_sd=None,
    noise_beta=2.0,
    nps_2d=None,
    seed=None
):
    """
    Image-domain lesion insertion pipeline

    ขั้นตอนการสร้าง simulated lesion:
    ════════════════════════════════════════════════════════

    STEP 1 — Ideal lesion mask (disk)
        lesion_mask = CircularDisk(cx, cy, r_px)   ∈ {0,1}
        lesion_nominal = lesion_mask × C_nominal    [HU]

    STEP 2 — PSF blur (MTF/TTF)
        PSF วัดจาก bead scan → normalize → FFT convolve
        lesion_blurred = lesion_nominal ⊛ PSF
        → จำลองว่า CT system มี finite resolution
        → edge ของ lesion จะ blur ตาม PSF width (FWHM)
        → MTF50/MTF10 บอกว่า frequency ไหนถูก attenuate เท่าไหร่

    STEP 3 — PVE scaling (Partial Volume Effect)
        PVE_xy = erf( r / (√2 · σ_xy) )
        PVE_z  = erf( r / (√2 · σ_z)  )   [ถ้ามี slice_thickness]
        scale  = PVE_xy × PVE_z
        lesion_pve = lesion_blurred × scale
        → lesion เล็กกว่า voxel → contrast ลดลงมาก
        → lesion ใหญ่กว่า voxel → contrast ≈ nominal

    STEP 4 — Add to background
        out_hu = background + lesion_pve

    STEP 5 — Correlated noise (NPS)  [optional]
        ถ้า nps_2d ให้มา → noise = IFFT(FFT(white) × √NPS)
        ถ้าไม่มี nps_2d  → noise = power-law(beta)
        noise_sd วัดจาก ROI ใน background จริง
        out_hu = out_hu + noise
    ════════════════════════════════════════════════════════
    """
    bg       = np.asarray(background_hu, dtype=np.float32)
    psf_norm = normalize_kernel(psf)
    H, W     = bg.shape

    # ── STEP 1: ideal lesion mask ──
    radius_px    = (diam_mm / 2.0) / pixel_spacing_mm
    lesion_mask  = make_circular_mask(bg.shape, cx, cy, radius_px)
    lesion_nominal = lesion_mask * float(contrast_hu)

    # ── STEP 2: PSF blur เฉพาะ patch รอบ lesion ──
    pad      = int(np.ceil(max(psf_norm.shape) / 2)) + 4
    half_box = int(np.ceil(radius_px)) + pad

    x0 = max(0, int(cx) - half_box);  x1 = min(W, int(cx) + half_box + 1)
    y0 = max(0, int(cy) - half_box);  y1 = min(H, int(cy) + half_box + 1)

    disk_patch = lesion_nominal[y0:y1, x0:x1].copy()

    # ── ตรวจ PSF FWHM — ถ้า spread เกิน 3 pixel ใช้ Gaussian แทน ──
    psf_center_line = psf_norm[psf_norm.shape[0]//2, :]
    fwhm_est = compute_fwhm(psf_center_line, 1.0)
    if not np.isfinite(fwhm_est) or fwhm_est > 3.0:
        # สร้าง Gaussian PSF จาก pixel_spacing แทน
        sigma_px = (pixel_spacing_mm / 2.355)  # FWHM = 1 pixel
        from scipy.ndimage import gaussian_filter
        patch_blurred = gaussian_filter(disk_patch, sigma=sigma_px)
    else:
        patch_blurred = fft_convolve2d_same(disk_patch, psf_norm)

    lesion_blurred                = np.zeros_like(bg)
    lesion_blurred[y0:y1, x0:x1] = patch_blurred

    # ── STEP 3: PVE scaling ──
    pve_scale  = pve_contrast_scaling(
        diam_mm            = diam_mm,
        pixel_spacing_mm   = pixel_spacing_mm,
        slice_thickness_mm = slice_thickness_mm
    )
    lesion_pve = lesion_blurred * pve_scale

   # ── STEP 4: insert into background ──
    out_hu = bg.copy()
    out_hu = out_hu + lesion_pve   # ← add ทั้งภาพ (lesion_pve = 0 นอก patch อยู่แล้ว)

    # ── STEP 5: noise ใส่ทั้งภาพ ไม่ใช่แค่ patch ──
    added_noise = None
    if add_noise:
        if noise_sd is None:
            yy, xx   = np.ogrid[:H, :W]
            roi      = np.sqrt((xx-cx)**2 + (yy-cy)**2) <= radius_px * 3
            noise_sd = float(bg[roi].std()) if np.any(roi) else 5.0

        added_noise = generate_correlated_noise(
            shape    = bg.shape,
            noise_sd = noise_sd,
            beta     = noise_beta,
            seed     = seed,
            nps_2d   = nps_2d
        )
        out_hu = out_hu + added_noise   

    details = {
        "radius_px":       radius_px,
        "lesion_mask":     lesion_mask,
        "lesion_nominal":  lesion_nominal,
        "lesion_blurred":  lesion_blurred,
        "pve_scale":       pve_scale,
        "lesion_pve":      lesion_pve,
        "noise":           added_noise,
        "noise_sd_used":   noise_sd,
        "patch_box":       (x0, y0, x1, y1),
    }
    return out_hu, details
# ══════════════════════════════════════════════════
#  CHO — Channelized Hotelling Observer
# ══════════════════════════════════════════════════

def gabor_channel(shape:tuple, freq:float, sigma_x:float, sigma_y:float,
                  theta:float, phase:float)->np.ndarray:
    """
    2-D Gabor filter channel.

    Math:
    ──────────────────────────────────────────────────────────
    g(x,y;f,σx,σy,θ,φ) =
        exp(−x'²/2σx² − y'²/2σy²) · cos(2πf·x' + φ)
    where:
        x' =  x cosθ + y sinθ
        y' = −x sinθ + y cosθ
    ──────────────────────────────────────────────────────────
    """
    rows,cols = shape
    cx,cy = cols//2, rows//2
    y,x = np.mgrid[:rows,:cols]
    x = x - cx; y = y - cy
    xp =  x*math.cos(theta) + y*math.sin(theta)
    yp = -x*math.sin(theta) + y*math.cos(theta)
    gauss = np.exp(-xp**2/(2*sigma_x**2) - yp**2/(2*sigma_y**2))
    cosine = np.cos(2*math.pi*freq*xp + phase)
    return (gauss * cosine).astype(np.float32)

def build_gabor_channels(roi_size:int, n_passbands:int=4,
                          n_directions:int=2, n_phases:int=2)->list:
    """
    Build Gabor channel bank (as per Fan et al. / Zhou et al.).

    Parameters match ACR phantom optimization:
      - 4 passbands (spatial frequencies 0.05–0.3 cyc/px)
      - 2 directions (0°, 90°)
      - 2 phases (0°, 90°)
    → 4 × 2 × 2 = 16 channels total
    """
    channels = []
    freqs = np.linspace(0.05, 0.3, n_passbands)
    sigma = 1.0 / (2*math.pi * 0.05)   # bandwidth
    thetas  = [k*math.pi/n_directions for k in range(n_directions)]
    phases  = [k*math.pi/n_phases     for k in range(n_phases)]
    for freq in freqs:
        sx = sigma
        sy = sigma
        for theta in thetas:
            for phase in phases:
                ch = gabor_channel((roi_size,roi_size), freq, sx,sy, theta, phase)
                channels.append(ch.flatten())
    return channels   # list of 1-D vectors, length = roi_size²

def extract_roi_flat(hu:np.ndarray, cx:int, cy:int, half:int)->np.ndarray|None:
    """Extract square ROI, return flattened, or None if out of bounds."""
    rows,cols = hu.shape
    r0,r1 = cy-half, cy+half
    c0,c1 = cx-half, cx+half
    if r0<0 or r1>rows or c0<0 or c1>cols: return None
    return hu[r0:r1, c0:c1].flatten()

def compute_cho_dprime(signal_rois:list, bg_rois:list,
                        channels:list)->dict:
    """
    Channelized Hotelling Observer d'.

    Math:
    ──────────────────────────────────────────────────────────
    1. Channel output for image g:
         v = U^T g
       U = matrix whose columns are the Gabor channel vectors

    2. Mean channel outputs:
         v̄_s = mean over signal-present images
         v̄_b = mean over background images

    3. Channelized covariance (pooled):
         K_v = (K_s + K_b) / 2
       where K_s, K_b are sample covariances of v in each class.
       Regularised: K_v += ε·I

    4. Hotelling discriminant:
         w = K_v⁻¹ (v̄_s − v̄_b)

    5. SNR (index of detectability):
         d' = √[ (v̄_s − v̄_b)^T K_v⁻¹ (v̄_s − v̄_b) ]
            = √(Δv̄^T w)

    6. AUC = Φ(d'/√2)   where Φ = normal CDF
    ──────────────────────────────────────────────────────────
    """
    U = np.array(channels).T          # shape: (npx, nchan)

    def project(rois):
        return np.array([U.T @ r for r in rois])   # (N, nchan)

    vs = project(signal_rois)
    vb = project(bg_rois)

    vs_mean = vs.mean(axis=0)
    vb_mean = vb.mean(axis=0)
    delta   = vs_mean - vb_mean

    Ks = np.cov(vs.T) if vs.shape[0]>1 else np.eye(len(delta))*1e-3
    Kb = np.cov(vb.T) if vb.shape[0]>1 else np.eye(len(delta))*1e-3
    Kv = (Ks + Kb)/2

    # Regularise
    eps = 1e-6 * np.trace(Kv)/len(delta)
    Kv += eps * np.eye(len(delta))

    try:
        Kv_inv = np.linalg.inv(Kv)
    except:
        Kv_inv = np.linalg.pinv(Kv)

    dprime_sq = float(delta @ Kv_inv @ delta)
    dprime    = math.sqrt(max(0, dprime_sq))
    auc       = float(0.5*(1 + erf(dprime/(2*math.sqrt(2)))))

    return dict(dprime=dprime, auc=auc,
                delta=delta, Kv=Kv, vs_mean=vs_mean, vb_mean=vb_mean)

def run_cho_from_clock(sl):

    hu = sl["hu_mod"] if sl["hu_mod"] is not None else sl["hu_orig"]

    lesions = sl.get("lesions", [])

    if len(lesions) < 5:
        return None

    results = []

    for l in lesions:

        cx = l["cx"]
        cy = l["cy"]
        r_px = l["r_px"]

        half = int(r_px*1.5)

        sig = extract_roi_flat(hu, cx, cy, half)

        bg1 = extract_roi_flat(hu, cx+int(r_px*3), cy, half)
        bg2 = extract_roi_flat(hu, cx-int(r_px*3), cy, half)

        if sig is None or bg1 is None:
            continue

        channels = build_gabor_channels(half*2)

        cho = compute_cho_dprime(
            [sig],
            [bg1,bg2],
            channels
        )

        cho["diam_mm"] = l["diam_mm"]
        cho["contrast"] = l["contrast"]
        cho["position"] = l["position"]

        results.append(cho)

    return {"results":results}

def plot_cho(cho_result:dict, sl:dict)->bytes:
    dark_style()
    res = cho_result["results"]
    if not res: return b""

    # Group by diameter
    diams  = sorted(set(r["diam_mm"] for r in res))
    contrasts = sorted(set(r["contrast"] for r in res))

    fig,axes = plt.subplots(1,2,figsize=(12,4),facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    colors = [ACC, ACC2, "#ff9800"]
    for ci,contrast in enumerate(contrasts):
        d_vals = [r["diam_mm"] for r in res if r["contrast"]==contrast]
        dp_vals= [r["dprime"]  for r in res if r["contrast"]==contrast]
        au_vals= [r["auc"]     for r in res if r["contrast"]==contrast]
        axes[0].plot(d_vals, dp_vals, "o-", color=colors[ci%3], lw=1.5,
                     label=f"{contrast:+d} HU", ms=6)
        axes[1].plot(d_vals, au_vals, "o-", color=colors[ci%3], lw=1.5,
                     label=f"{contrast:+d} HU", ms=6)

    axes[0].set_xlabel("Object size (mm)"); axes[0].set_ylabel("d' (index of detectability)")
    axes[0].set_title("CHO d' vs Object Size"); axes[0].legend(fontsize=8)
    axes[0].axhline(1.0,color=MUTED,lw=0.8,ls=":")
    axes[0].grid(True)

    axes[1].set_xlabel("Object size (mm)"); axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC vs Object Size"); axes[1].legend(fontsize=8)
    axes[1].axhline(0.84,color=MUTED,lw=0.8,ls=":",label="AUC=0.84 (d'=2)")
    axes[1].set_ylim(0.4,1.05); axes[1].grid(True)

    plt.suptitle("Channelized Hotelling Observer (CHO) Results", fontsize=11,
                 color="#d8e8ff", y=1.01)
    plt.tight_layout(pad=0.8)
    data=fig2img(fig); plt.close(fig); return data

def plot_cho_rois(sl:dict, cho_result:dict)->bytes:
    """Visualise signal + background ROI squares on the phantom image."""
    dark_style()
    hu  = sl["hu_mod"] if sl["hu_mod"] is not None else sl["hu_orig"]
    img = wl_ww(hu, 0, 400)

    res = cho_result["results"]
    n   = min(len(res),6)
    fig,axes = plt.subplots(2,3,figsize=(12,8),facecolor=DARK) if n>3 else \
               plt.subplots(1,n,figsize=(4*n,4),facecolor=DARK)
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor(DARK)

    for i,r in enumerate(res[:n]):
        ax = axes[i]
        cx,cy,half,offset = r["cx"],r["cy"],r["half"],r["offset"]
        crop_h = half+offset+10
        r0=max(0,cy-crop_h); r1=min(hu.shape[0],cy+crop_h)
        c0=max(0,cx-crop_h); c1=min(hu.shape[1],cx+crop_h)
        ax.imshow(img[r0:r1,c0:c1], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

        # Signal ROI (main)
        s_rect = Rectangle((cx-half-c0, cy-half-r0), half*2, half*2,
                             fill=False, edgecolor="#ff5252", lw=1.5,
                             linestyle="-", label="Signal")
        ax.add_patch(s_rect)

        # BG left
        b1_rect = Rectangle((cx-offset-half-c0, cy-half-r0), half*2, half*2,
                              fill=False, edgecolor=ACC2, lw=1.2,
                              linestyle="--", label="BG")
        ax.add_patch(b1_rect)

        # BG right
        b2_rect = Rectangle((cx+offset-half-c0, cy-half-r0), half*2, half*2,
                              fill=False, edgecolor=ACC2, lw=1.2,
                              linestyle="--")
        ax.add_patch(b2_rect)

        ax.set_title(
            f"Ø{r['diam_mm']}mm {r['contrast']:+d}HU\n"
            f"d'={r['dprime']:.3f}  AUC={r['auc']:.3f}",
            fontsize=8, color="#d8e8ff", pad=3)

    for j in range(n,len(axes)): axes[j].set_visible(False)

    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color="#ff5252",lw=1.5,label="Signal ROI"),
             Line2D([0],[0],color=ACC2,lw=1.2,ls="--",label="BG ROI")]
    fig.legend(handles=handles,loc="lower right",fontsize=8,
               facecolor=SURF,labelcolor="#d8e8ff")
    plt.suptitle("CHO ROI Visualisation — Set 2 Lesions",
                 fontsize=10, color="#d8e8ff")
    plt.tight_layout(pad=0.4)
    data=fig2img(fig); plt.close(fig); return data
# ══════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════

# ══════════════════════════════════════════════════
#  REPORT EXPORT FUNCTIONS
# ══════════════════════════════════════════════════
def build_report_rows():
    rows = []
    for i, sl in enumerate(st.session_state.slices):
        bd = sl.get("uni_bd", {}) or {}
        rows.append({
            "index": i,
            "file": sl.get("name",""),
            "is_uniformity": bool(sl.get("is_uniformity", False)),
            "uni_score": sl.get("uni_score", None),
            "mu": bd.get("mu", None),
            "inner_std": bd.get("inner_std", None),
            "phantom_std": bd.get("phantom_std", None),

            # protocol fields
            "manufacturer": sl.get("manufacturer",""),
            "model": sl.get("model",""),
            "station": sl.get("station",""),
            "protocol_name": sl.get("protocol_name",""),
            "series_desc": sl.get("series_desc",""),
            "kvp": sl.get("kvp", None),
            "mas": sl.get("mas", None),
            "kernel": sl.get("kernel",""),
            "recon_method": sl.get("recon_method",""),
            "iter_desc": sl.get("iter_desc",""),
            "dlir_level": sl.get("dlir_level",""),

            # lesions summary
            "n_lesions": len(sl.get("lesions",[])),
        })
    return rows

def report_csv_bytes():
    import pandas as pd
    df = pd.DataFrame(build_report_rows())
    return df.to_csv(index=False).encode("utf-8")

def report_pdf_bytes():
    # Build a simple 1–2 page PDF: header + key settings + table preview
    import pandas as pd

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(18*mm, (H-18*mm), "CT Phantom QC Report")

    # Timestamp
    c.setFont("Helvetica", 9)
    c.drawString(18*mm, (H-24*mm), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary
    n = len(st.session_state.slices)
    nu = len(st.session_state.uniformity_idx)
    nles = sum(len(s.get("lesions",[])) for s in st.session_state.slices)
    c.setFont("Helvetica", 10)
    c.drawString(18*mm, (H-34*mm), f"Slices: {n}    Uniformity: {nu}    Total lesions: {nles}")

    # Current slice protocol snapshot
    cur = st.session_state.current
    sl = st.session_state.slices[cur] if n else {}
    y = H-46*mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(18*mm, y, "Current slice protocol snapshot")
    y -= 6*mm
    c.setFont("Helvetica", 9)
    lines = [
        f"File: {sl.get('name','')}",
        f"Manufacturer/Model: {sl.get('manufacturer','')} {sl.get('model','')}",
        f"Protocol/Series: {sl.get('protocol_name','')} | {sl.get('series_desc','')}",
        f"kVp: {sl.get('kvp','')}   mAs: {sl.get('mas','')}   Kernel: {sl.get('kernel','')}",
        f"Recon: {sl.get('recon_method','')}   IR/DL: {sl.get('iter_desc','') or sl.get('dlir_level','')}",
    ]
    for ln in lines:
        c.drawString(18*mm, y, ln)
        y -= 5*mm

    # Uniformity top rows preview
    y -= 3*mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(18*mm, y, "Uniformity ranking (top 10)")
    y -= 6*mm
    c.setFont("Helvetica", 8)

    ranked = st.session_state.get("uniformity_ranked", [])[:10]
    header = "rank | file | score | mu | inner_std | kernel | kVp | mAs"
    c.drawString(18*mm, y, header)
    y -= 4*mm

    for k, (idx, sc, bd) in enumerate(ranked, start=1):
        s = st.session_state.slices[idx]
        mu = (s.get("uni_bd",{}) or {}).get("mu","")
        sd = (s.get("uni_bd",{}) or {}).get("inner_std","")
        row = f"{k:>4} | {s.get('name','')[:26]:<26} | {sc:>5} | {str(mu):>6} | {str(sd):>8} | {s.get('kernel','')[:10]:<10} | {s.get('kvp','')} | {s.get('mas','')}"
        c.drawString(18*mm, y, row)
        y -= 4*mm
        if y < 25*mm:
            c.showPage()
            y = H-20*mm
            c.setFont("Helvetica", 8)

    # Add MTF/CHO images if exist (optional)
    # We keep it simple: if session has mtf_result / cho_result, generate plots and embed as raster
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(18*mm, H-18*mm, "Analysis Figures (if available)")
    y = H-28*mm

    try:
        if st.session_state.get("mtf_result") is not None:
            img = plot_mtf(st.session_state.mtf_result)
            # Save to temp and draw
            tmp = io.BytesIO(img)
            from reportlab.lib.utils import ImageReader
            c.drawString(18*mm, y, "MTF/TTF")
            y -= 6*mm
            c.drawImage(ImageReader(tmp), 18*mm, y-80*mm, width=170*mm, height=80*mm, preserveAspectRatio=True, mask='auto')
            y -= 90*mm
    except Exception:
        pass

    try:
        if st.session_state.get("cho_result") is not None and st.session_state.get("cho_result"):
            slc = st.session_state.slices[st.session_state.current]
            img1 = plot_cho_rois(slc, st.session_state.cho_result)
            img2 = plot_cho(st.session_state.cho_result, slc)

            from reportlab.lib.utils import ImageReader
            if img1:
                c.drawString(18*mm, y, "CHO ROI Visualisation")
                y -= 6*mm
                c.drawImage(ImageReader(io.BytesIO(img1)), 18*mm, y-70*mm, width=170*mm, height=70*mm, preserveAspectRatio=True, mask='auto')
                y -= 80*mm
            if img2:
                c.drawString(18*mm, y, "CHO d' / AUC")
                y -= 6*mm
                c.drawImage(ImageReader(io.BytesIO(img2)), 18*mm, y-70*mm, width=170*mm, height=70*mm, preserveAspectRatio=True, mask='auto')
                y -= 80*mm
    except Exception:
        pass

    c.save()
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════
#  SERIES HELPERS
# ══════════════════════════════════════════════════
def get_series_label(sl: dict) -> str:
    """Build a human-readable series label from DICOM tags."""
    parts = []
    sd = sl.get("series_desc", "").strip()
    pn = sl.get("protocol_name", "").strip()
    kr = sl.get("kernel", "").strip()
    kv = sl.get("kvp")
    ma = sl.get("mas")
    if sd:   parts.append(sd)
    elif pn: parts.append(pn)
    if kr:   parts.append(kr)
    if kv:   parts.append(f"{int(kv)}kVp")
    if ma:   parts.append(f"{int(ma)}mAs")
    return " · ".join(parts) if parts else "Unknown Series"

def get_all_series(slices: list) -> list:
    """Return sorted unique series labels."""
    return sorted(set(get_series_label(s) for s in slices))

def get_active_slices(slices: list, active_series) -> list:
    """Filter slices to only those in the active series. None = all."""
    if active_series is None or active_series == "— All Series —":
        return slices
    return [s for s in slices if get_series_label(s) == active_series]

with st.sidebar:
    st.markdown('<div class="vitTitle">⚕ CT Phantom QC</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:#3a5060;letter-spacing:2px;margin-bottom:12px;">& VIRTUAL IMAGING TRIAL · v1.0</div>', unsafe_allow_html=True)

    if not HAS_PYDICOM:
        st.markdown('<div class="warnbox">pydicom ไม่พบ — <code>pip install pydicom</code></div>',
                    unsafe_allow_html=True)
    if not HAS_SKIMAGE:
        st.markdown('<div class="warnbox">scikit-image ไม่พบ (projection-domain fallback) — <code>pip install scikit-image</code></div>',
                    unsafe_allow_html=True)

    # ── Step 1: Load ──
    st.markdown('<div class="secLabel">Step 1 · Load DICOM Folder</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "เลือกทุกไฟล์ในโฟลเดอร์ (Ctrl+A) — รองรับ .dcm, ไฟล์ไม่มีนามสกุล, PNG/JPG",
        type=None,
        accept_multiple_files=True,
        label_visibility="collapsed")

    if uploaded:
        _uploaded_names = {f.name for f in uploaded}
        _current_names  = {s["name"] for s in st.session_state.slices}
        # ถ้าชุดไฟล์เปลี่ยน → ล้างเก่าออกก่อน
        if _uploaded_names != _current_names:
            st.session_state.slices = []
            st.session_state.uniformity_idx    = []
            st.session_state.uniformity_ranked = []
            st.session_state.current           = 0
            st.session_state.inserted_set1     = False
            st.session_state.inserted_set2     = False
            st.session_state.active_series     = None
            st.session_state.nps_results       = []
            st.session_state.qc_status         = {"uniformity":False,"nps":False,"pve":False,"psf":False}
            st.session_state.basic_geometry_result  = None
            st.session_state.basic_square_result    = None
            st.session_state.basic_linearity_result = None
            st.session_state.basic_accuracy_result  = None
            st.session_state.basic_noise_result     = None
            st.session_state.basic_slice_result     = None
            for k in ["_nps_mean","nps_cache","mean_psf","psf_metrics","sim_ready"]:
                st.session_state.pop(k, None)
            with st.spinner(f"Loading {len(uploaded)} files…"):
                for f in sorted(uploaded, key=lambda x: x.name):
                    raw = f.read()
                    rec = None
                    if HAS_PYDICOM:
                        rec = parse_dicom(raw, f.name)
                    if rec is None:
                        rec = parse_image_fallback(raw, f.name)
                    if rec:
                        st.session_state.slices.append(rec)
            st.session_state.slices.sort(key=lambda x: x["name"])
            slog(f"Loaded {len(uploaded)} files (cleared previous)", "ok")
            st.rerun()

        # ── Series Selector ──
        all_series = get_all_series(st.session_state.slices)
        if len(all_series) > 1:
            st.markdown('<div class="secLabel">Series</div>', unsafe_allow_html=True)
            series_options = ["— All Series —"] + all_series
            prev_series = st.session_state.active_series
            sel_series = st.selectbox(
                "เลือก Series",
                series_options,
                index=series_options.index(prev_series)
                      if prev_series in series_options else 0,
                label_visibility="collapsed"
            )
            if sel_series != prev_series:
                st.session_state.active_series = sel_series
                # Reset current to first slice of selected series
                view_slices = get_active_slices(st.session_state.slices, sel_series)
                if view_slices:
                    st.session_state.current = st.session_state.slices.index(view_slices[0])
                st.rerun()

            # Info: how many slices in this series
            view_slices = get_active_slices(st.session_state.slices,
                                            st.session_state.active_series)
            st.caption(f"{len(view_slices)} slices ใน series นี้ / {len(st.session_state.slices)} ทั้งหมด")
        else:
            st.session_state.active_series = "— All Series —"
            view_slices = st.session_state.slices

        # Build index mapping: view position → global slice index
        _view_slices = get_active_slices(st.session_state.slices,
                                         st.session_state.active_series)
        n_view = len(_view_slices)
        _global_idxs = [st.session_state.slices.index(s) for s in _view_slices]

        # Clamp current to valid range for this series
        if st.session_state.current not in _global_idxs and _global_idxs:
            st.session_state.current = _global_idxs[0]
        _view_pos = _global_idxs.index(st.session_state.current) if st.session_state.current in _global_idxs else 0


        # ── WL/WW ──
        st.markdown('<div class="secLabel">Window Level</div>', unsafe_allow_html=True)
        wl_preset = st.selectbox("Preset",
            ["Soft tissue (0/400)","Lung (−600/1500)","Bone (400/1800)","Custom"],
            label_visibility="collapsed")
        presets={"Soft tissue (0/400)":(0,400),"Lung (−600/1500)":(-600,1500),"Bone (400/1800)":(400,1800)}
        if wl_preset in presets:
            st.session_state.wl,st.session_state.ww = presets[wl_preset]
        c1,c2 = st.columns(2)
        wl = c1.number_input("WL",value=st.session_state.wl,step=10)
        ww = c2.number_input("WW",value=st.session_state.ww,step=10,min_value=1)
        st.session_state.wl,st.session_state.ww = wl,ww

        if st.button("🗑 Clear All",use_container_width=True):
            for k in ["slices","uniformity_idx","log"]:
                st.session_state[k]=[]
            for k in ["current","wl","ww"]:
                st.session_state[k]=0 if k!="ww" else 400
            st.session_state["inserted_set1"]=st.session_state["inserted_set2"]=False
            st.rerun()

        if st.session_state.slices:

            n = len(st.session_state.slices)

            slice_idx = st.slider(
                "Navigate slices",
                0,
                n-1,
                st.session_state.current,
                1
            )

            st.session_state.current = slice_idx

            sl = st.session_state.slices[slice_idx]

            img_bytes = render_slice(
                sl,
                st.session_state.wl,
                st.session_state.ww
            )

            st.image(img_bytes, use_container_width=True)

# ══════════════════════════════════════════════════
#  NAVIGATION HELPERS
# ══════════════════════════════════════════════════
def go(page):
    st.session_state.current_page = page
    st.rerun()

def _topbar(page_label, page_icon=""):
    """Render a slim topbar with back-to-home button."""
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⌂ Home", key="topbar_home"):
            go("home")
    with c2:
        st.markdown(
            f'<div class="topbar">'
            f'<span class="topbar-logo">⚕ CT Phantom QC</span>'
            f'<span class="topbar-sep">›</span>'
            f'<span class="topbar-page">{page_icon} {page_label}</span>'
            f'</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  HOMEPAGE
# ══════════════════════════════════════════════════
_page = st.session_state.get("current_page", "home")

if _page == "home":

    # ── Hero ──
    st.markdown("""
    <div class="home-hero">
      <div class="home-title">CT Phantom QC</div>
      <div class="home-sub">Virtual Imaging Trial Platform · v3.0</div>
      <div class="home-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── DICOM status banner ──
    _n_loaded = len(st.session_state.slices)
    if _n_loaded:
        st.markdown(
            f'<div class="okbox" style="text-align:center;font-size:12px;margin-bottom:18px;">'
            f'✓ &nbsp; <b>{_n_loaded} slices</b> โหลดแล้ว — พร้อมใช้งาน</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warnbox" style="text-align:center;font-size:12px;margin-bottom:18px;">'
            '⚠ &nbsp; ยังไม่ได้โหลด DICOM — เปิด Sidebar เพื่อโหลดไฟล์ก่อน</div>',
            unsafe_allow_html=True)
        with st.expander("📂 โหลด DICOM ที่นี่", expanded=not bool(_n_loaded)):
            st.markdown('<div class="secLabel">โหลด DICOM Folder</div>', unsafe_allow_html=True)
            _up_home = st.file_uploader(
                "เลือกทุกไฟล์ในโฟลเดอร์ (Ctrl+A) — รองรับ .dcm และ PNG/JPG",
                type=None, accept_multiple_files=True,
                key="home_uploader", label_visibility="collapsed")
            if _up_home:
                _up_home_names = {f.name for f in _up_home}
                _cur_names_h   = {s["name"] for s in st.session_state.slices}
                if _up_home_names != _cur_names_h:
                    st.session_state.slices = []
                    st.session_state.uniformity_idx    = []
                    st.session_state.uniformity_ranked = []
                    st.session_state.current           = 0
                    st.session_state.inserted_set1     = False
                    st.session_state.inserted_set2     = False
                    st.session_state.active_series     = None
                    st.session_state.nps_results       = []
                    st.session_state.qc_status         = {"uniformity":False,"nps":False,"pve":False,"psf":False}
                    st.session_state.basic_geometry_result  = None
                    st.session_state.basic_square_result    = None
                    st.session_state.basic_linearity_result = None
                    st.session_state.basic_accuracy_result  = None
                    st.session_state.basic_noise_result     = None
                    st.session_state.basic_slice_result     = None
                    for k in ["_nps_mean","nps_cache","mean_psf","psf_metrics","sim_ready"]:
                        st.session_state.pop(k, None)
                    with st.spinner(f"กำลังโหลด {len(_up_home)} ไฟล์…"):
                        for f in sorted(_up_home, key=lambda x: x.name):
                            raw = f.read()
                            rec = None
                            if HAS_PYDICOM:
                                rec = parse_dicom(raw, f.name)
                            if rec is None:
                                rec = parse_image_fallback(raw, f.name)
                            if rec:
                                st.session_state.slices.append(rec)
                    st.session_state.slices.sort(key=lambda x: x["name"])
                    slog(f"Loaded {len(_up_home)} files (cleared previous)", "ok")
                    st.rerun()

    # ── Tool Cards ──
    st.markdown('<div style="font-size:10px;letter-spacing:3px;color:#2a4060;text-transform:uppercase;margin-bottom:14px;">— เลือกเครื่องมือ —</div>', unsafe_allow_html=True)

    _row1 = st.columns(2)
    _row2 = st.columns(2)
    _row3 = st.columns(2)

    # Card 1 — Standard QC
    with _row1[0]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-cyan"></div>
          <span class="tool-card-icon">🏥</span>
          <div class="tool-card-title">Standard Basic QC</div>
          <div class="tool-card-desc">
            IAEA มาตรฐาน · Geometry · Square 50 mm · CT Number Linearity<br>
            ตรวจสอบคุณภาพพื้นฐานของ CT scanner
          </div>
          <span class="tool-card-tag tag-std">STANDARD · IAEA</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด Standard QC →", use_container_width=True, key="nav_basicqc"):
            go("basicqc")

    # Card 2 — DICOM Viewer
    with _row1[1]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-purple"></div>
          <span class="tool-card-icon">📷</span>
          <div class="tool-card-title">DICOM Viewer</div>
          <div class="tool-card-desc">
            เลื่อน slice ด้วย scroll wheel · Window Level/Width · Lesion overlay<br>
            ดู protocol tags และ series information
          </div>
          <span class="tool-card-tag tag-adv">VIEWER</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด Viewer →", use_container_width=True, key="nav_viewer"):
            go("viewer")

    # Card 3 — Advanced QC (NPS + MTF)
    with _row2[0]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-purple"></div>
          <span class="tool-card-icon">📈</span>
          <div class="tool-card-title">Advanced QC</div>
          <div class="tool-card-desc">
            MTF/TTF · Noise Power Spectrum (NPS) · Partial Volume Effect<br>
            วิเคราะห์ spatial resolution และ noise texture เชิงลึก
          </div>
          <span class="tool-card-tag tag-adv">ADVANCED · AAPM TG-233</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด Advanced QC →", use_container_width=True, key="nav_advqc"):
            go("advqc")

    # Card 4 — Image-Domain Simulation
    with _row2[1]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-orange"></div>
          <span class="tool-card-icon">🔬</span>
          <div class="tool-card-title">Image-Domain Simulation</div>
          <div class="tool-card-desc">
            Insert virtual lesions · PSF blur · PVE scaling · Correlated noise<br>
            สร้าง phantom simulation สำหรับ observer study
          </div>
          <span class="tool-card-tag tag-sim">SIMULATION · VIT</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด Simulation →", use_container_width=True, key="nav_sim"):
            go("simulation")

    # Card 5 — CHO Observer
    with _row3[0]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-purple"></div>
          <span class="tool-card-icon">🧪</span>
          <div class="tool-card-title">CHO d′ Observer</div>
          <div class="tool-card-desc">
            Channelized Hotelling Observer · Gabor channels · d′ / AUC<br>
            ประเมิน detectability ของ lesion แบบ task-based
          </div>
          <span class="tool-card-tag tag-adv">OBSERVER · CHO</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด CHO →", use_container_width=True, key="nav_cho"):
            go("cho")

    # Card 6 — Report & Log
    with _row3[1]:
        st.markdown("""
        <div class="tool-card">
          <div class="tool-card-accent accent-green"></div>
          <span class="tool-card-icon">📋</span>
          <div class="tool-card-title">Report & Log</div>
          <div class="tool-card-desc">
            Export PDF/CSV · QC workflow status · Activity log<br>
            สรุปผลและดาวน์โหลดรายงานทั้งหมด
          </div>
          <span class="tool-card-tag tag-log">EXPORT · LOG</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("เปิด Report →", use_container_width=True, key="nav_report"):
            go("report")

    # ── Quick-status footer ──
    if _n_loaded:
        st.markdown("---")
        _qc = st.session_state.qc_status
        _geo_done = st.session_state.basic_geometry_result is not None
        _sq_done  = st.session_state.basic_square_result  is not None
        _lin_done = st.session_state.basic_linearity_result is not None
        def _si(done, ok=True):
            if not done: return "⚪"
            return "🟢" if ok else "🔴"
        st.markdown(f"""
        <div style="background:#0a0f18;border:1px solid #1a2a3a;border-radius:10px;
          padding:14px 20px;display:flex;flex-wrap:wrap;gap:20px;font-size:11px;color:#4a7090;">
          <span>📊 Slices: <b style="color:#00e5ff">{_n_loaded}</b></span>
          <span>{_si(_geo_done)} Geometry QC</span>
          <span>{_si(_sq_done)} Square 50mm</span>
          <span>{_si(_lin_done)} CT Linearity</span>
          <span>{_si(_qc['uniformity'])} Uniformity</span>
          <span>{_si(_qc['nps'])} NPS</span>
          <span>{_si(_qc['pve'])} PVE</span>
          <span>{_si(_qc['psf'])} PSF</span>
        </div>""", unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════
#  DICOM must be loaded for all sub-pages
# ══════════════════════════════════════════════════
n = len(st.session_state.slices)
if not n:
    _topbar("No Data")
    st.markdown("""
    <div style="text-align:center;padding:80px 0;">
      <div style="font-size:60px;opacity:.3;">🫁</div>
      <div style="font-family:'Syne',sans-serif;font-size:18px;color:#2a4060;margin-top:14px;">
        กลับไปหน้าหลักเพื่อโหลด DICOM ก่อน
      </div>
    </div>""", unsafe_allow_html=True)
    if st.button("⌂ กลับหน้าหลัก", type="primary"):
        go("home")
    st.stop()

# ── Series / slice shared setup for all sub-pages ──
_act_ser     = st.session_state.active_series
_all_ser     = get_all_series(st.session_state.slices)
_active      = get_active_slices(st.session_state.slices, st.session_state.active_series)
_active_idxs = [st.session_state.slices.index(s) for s in _active]
n_view       = len(_active_idxs)
if n_view == 0:
    st.warning("No slices in selected series"); st.stop()
wl, ww_val = st.session_state.wl, st.session_state.ww


# ══════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════

# ── BASIC QC PAGE ──────────────────────────────────────────────────────────
if _page == "basicqc":
    _topbar("Standard Basic QC", "🏥")

# ══════════════════════════════════════════════════
#  BASIC QC FUNCTIONS (from app.py)
# ══════════════════════════════════════════════════

# ── ตรวจหา phantom boundary โดยใช้ threshold + centroid (robust กว่า Hough) ──
def _basic_find_phantom(hu):
    """
    Returns (cx, cy, radius_px) as plain Python ints/float.
    ใช้ threshold-based centroid แทน HoughCircles เพื่อความ robust
    """
    body = hu > -500
    if not np.any(body):
        h, w = hu.shape
        return w//2, h//2, min(h,w)//3
    yy, xx = np.where(body)
    cx = float(xx.mean()); cy = float(yy.mean())
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    r = float(np.percentile(dist, 90))
    return int(round(cx)), int(round(cy)), r


def detect_phantom_advanced(hu, px):

    # 1. หา phantom คร่าว ๆ ก่อน
    cx, cy, r = _basic_find_phantom(hu)

    cx = int(round(cx))
    cy = int(round(cy))
    r  = int(round(r))

    # 🔥 2. refine outer radius (เพิ่มตรงนี้!)
    try:
        r = refine_outer_radius_strong(hu, cx, cy, r)
    except:
        pass  # fallback ใช้ r เดิม

    # 🔥 3. หา inner แบบ robust
    innerRadius = _find_inner_radius_robust(hu, cx, cy, r)

    outerCenter = (cx, cy)
    outerRadius = int(r)
    innerCenter = outerCenter

    return outerCenter, outerRadius, innerCenter, int(innerRadius)

def run_geometry_qc(hu, px):
    try:
        outerCenter, outerRadius, innerCenter, innerRadius = detect_phantom_advanced(hu, px)
    except Exception as e:
        return None, f"❌ Detection error: {e}"

    cx, cy = int(innerCenter[0]), int(innerCenter[1])
    r = int(innerRadius)
    if r <= 0:
        return None, "❌ Inner radius detection failed"

    nominalDiameter = 150
    tol_mm      = nominalDiameter * 0.02
    diameter_mm = 2 * r * px
    diff_mm     = diameter_mm - nominalDiameter
    isPass      = abs(diff_mm) <= tol_mm

    # ── ขยาย 3x ก่อน render ──
    SCALE = 3
    img_norm = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output   = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    output   = cv2.resize(output, (output.shape[1]*SCALE, output.shape[0]*SCALE),
                          interpolation=cv2.INTER_LINEAR)

    cxs, cys, rs = cx*SCALE, cy*SCALE, r*SCALE

    # วงกลม phantom
    cv2.circle(output, (cxs, cys), rs, (0, 255, 255), 2)

    # เส้นแนวตั้ง (Vertical diameter)
    cv2.line(output, (cxs, cys - rs), (cxs, cys + rs), (0, 255, 0), 2)

    # เส้นแนวนอน (Horizontal diameter)
    cv2.line(output, (cxs - rs, cys), (cxs + rs, cys), (255, 100, 0), 2)

    col_txt = (0, 255, 0) if isPass else (0, 0, 255)

    # ตัวเลขแนวตั้ง
    cv2.putText(output,
                f"V: {diameter_mm:.2f} mm",
                (cxs + 10, cys - rs//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # ตัวเลขแนวนอน
    cv2.putText(output,
                f"H: {diameter_mm:.2f} mm",
                (cxs - rs + 10, cys - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2, cv2.LINE_AA)

    # PASS/FAIL
    cv2.putText(output,
                "PASS" if isPass else "FAIL",
                (cxs - 60, cys + rs + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_txt, 3, cv2.LINE_AA)

    margin = max(5, int(r * 0.2)) * SCALE
    x1 = max(0, cxs - rs - margin); x2 = min(output.shape[1], cxs + rs + margin)
    y1 = max(0, cys - rs - margin); y2 = min(output.shape[0], cys + rs + margin)
    zoom = output[y1:y2, x1:x2]

    df = pd.DataFrame([[f"{diameter_mm:.2f}", f"{nominalDiameter}",
                        f"{diff_mm:.2f}", "PASS" if isPass else "FAIL"]],
                      columns=["Measured (mm)", "Nominal (mm)", "Diff (mm)", "Status"])

    return {"image": zoom, "diameter": diameter_mm, "nominal": nominalDiameter,
            "tol": tol_mm, "diff": diff_mm, "pass": isPass, "df": df}, None


def run_square_qc(hu, px):
    """
    Square 50 mm QC
    - ใช้ geometry เป็น baseline
    - ใช้ hole detection เป็น optional refine
    - ถ้าตรวจ hole ไม่ได้ ก็ยังวัดได้จาก geometry
    """
    import numpy as np
    import cv2
    import pandas as pd

    try:
        outerCenter, outerRadius, _, _ = detect_phantom_advanced(hu, px)
    except Exception as e:
        return None, f"❌ Detection error: {e}"

    cx, cy = int(outerCenter[0]), int(outerCenter[1])

    nominal = 50.0
    template_mm = 55.0

    side_px = float(nominal / px)
    half = side_px / 2.0

    # geometry baseline
    C4_geom = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half]
    ], dtype=float)

    # ROI สำหรับหา hole
    roi_half = int(round(template_mm / px / 2.0))
    x1 = max(0, cx - roi_half)
    x2 = min(hu.shape[1] - 1, cx + roi_half)
    y1 = max(0, cy - roi_half)
    y2 = min(hu.shape[0] - 1, cy + roi_half)

    roi = hu[y1:y2, x1:x2]
    if roi.size == 0:
        return None, "❌ ROI empty — phantom too close to edge"

    use_detection = False
    C4 = C4_geom.copy()

    # ─────────────────────────
    # optional hole detection
    # ─────────────────────────
    try:
        img = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        img = cv2.GaussianBlur(img, (5, 5), 1.2)

        # small dark holes -> invert
        img_inv = 255 - img

        r0 = max(1.0, 1.5 / px)
        minR = max(2, int(round(r0 * 0.5)))
        maxR = max(minR + 2, int(round(r0 * 2.2)))

        circles = None
        for p2 in [18, 16, 14, 12, 10, 9, 8, 7]:
            c_try = cv2.HoughCircles(
                img_inv,
                cv2.HOUGH_GRADIENT,
                dp=1.1,
                minDist=max(8, int(r0 * 3.5)),
                param1=100,
                param2=p2,
                minRadius=minR,
                maxRadius=maxR
            )
            if c_try is not None and len(c_try[0]) >= 4:
                circles = np.round(c_try[0]).astype(int)
                break

        if circles is not None:
            centers = np.array([[c[0] + x1, c[1] + y1] for c in circles], dtype=float)

            chosen = []
            used = set()

            for k in range(4):
                d = np.linalg.norm(centers - C4_geom[k], axis=1)
                order = np.argsort(d)
                picked = None
                for idx_c in order:
                    if idx_c not in used:
                        picked = idx_c
                        break
                if picked is not None:
                    chosen.append(picked)
                    used.add(picked)

            if len(chosen) == 4:
                C4 = centers[chosen].astype(float)
                use_detection = True

    except Exception:
        use_detection = False
        C4 = C4_geom.copy()

    # ─────────────────────────
    # measure
    # ─────────────────────────
    tol2 = nominal * 0.02
    rows = []
    pass_all = []
    side_labels = ["Top", "Right", "Bottom", "Left"]

    img_full = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = cv2.cvtColor(img_full, cv2.COLOR_GRAY2BGR)

    # ขยาย 3x ก่อน putText → ตัวเลขคมชัด
    SCALE = 3
    out = cv2.resize(out, (out.shape[1]*SCALE, out.shape[0]*SCALE),
                     interpolation=cv2.INTER_LINEAR)
    x1 *= SCALE; x2 *= SCALE; y1 *= SCALE; y2 *= SCALE
    C4  = C4 * SCALE
    Pts = np.vstack([C4, C4[:1]])

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
    for (px_x, px_y) in C4.astype(int):
        cv2.circle(out, (px_x, px_y), 6*SCALE, (0, 255, 255), 1)

    for i in range(4):
        p1v = Pts[i]
        p2v = Pts[i + 1]

        dist_mm = float(np.linalg.norm(p2v - p1v) * px / SCALE)
        diff = abs(dist_mm - nominal)
        ok = diff <= tol2
        pass_all.append(ok)

        rows.append([
            side_labels[i],
            f"{dist_mm:.2f}",
            "50.00",
            "±2%",
            "PASS" if ok else "FAIL"
        ])

        col_cv = (0, 200, 0) if ok else (0, 0, 255)
        cv2.line(out, tuple(p1v.astype(int)), tuple(p2v.astype(int)), col_cv, 2)

        mid = ((p1v + p2v) / 2).astype(int)
        cv2.putText(
            out,
            f"{dist_mm:.1f}mm",
            (mid[0] - 40, mid[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    overallPass = all(pass_all)

    margin_z = max(5, int(round(0.10 * (template_mm / px)))) * SCALE
    zx1 = max(0, x1 - margin_z)
    zx2 = min(out.shape[1] - 1, x2 + margin_z)
    zy1 = max(0, y1 - margin_z)
    zy2 = min(out.shape[0] - 1, y2 + margin_z)
    zoom = out[zy1:zy2, zx1:zx2]

    df = pd.DataFrame(
        rows,
        columns=["Side", "Measured (mm)", "Nominal (mm)", "Tolerance", "Status"]
    )

    warn = "" if use_detection else " (geometry mode — hole detection not used)"
    return {"image": zoom, "df": df, "pass": overallPass, "warn": warn}, None

def run_linearity_qc(hu, px):
    """
    CT Number Linearity QC
    """
    try:
        outerCenter, outerRadius, innerCenter, innerRadius = detect_phantom_advanced(hu, px)
    except Exception as e:
        return None, f"❌ Detection error: {e}"

    cx, cy = int(outerCenter[0]), int(outerCenter[1])
    R = int(outerRadius)
    innerR = int(innerRadius)

    SCALE = 3
    img_norm = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    output = cv2.resize(output, (output.shape[1]*SCALE, output.shape[0]*SCALE),
                        interpolation=cv2.INTER_LINEAR)

    cx  *= SCALE;  cy  *= SCALE
    R   *= SCALE;  innerR *= SCALE

    cv2.circle(output, (cx, cy), R,      (255, 0, 0),     2)
    cv2.circle(output, (cx, cy), innerR, (255, 255, 255),  2)

    inserts = detect_linearity_inserts(hu, px)

    if not inserts:
        return None, "❌ ตรวจไม่พบ insert — ลองเลือก slice CTP404 ที่ถูกต้อง"

    materialRanges = {
        "Air":         (-1100, -850),
        "PMP":         (-260,  -140),
        "LDPE":        (-130,   -60),
        "Polystyrene": ( -80,    15),
        "Water":       ( -25,    25),
        "Acrylic":     (  70,   170),
        "Delrin":      ( 230,   450),
        "Teflon":      ( 750,  1150),
    }

    rows = []
    yy, xx = np.mgrid[0:hu.shape[0], 0:hu.shape[1]]

    for ins in inserts:
        ix, iy = int(ins["cx"]), int(ins["cy"])
        ir = int(ins["r_px"])

        roi_r = max(3, int(ir * 0.85))
        mask = (xx - ix)**2 + (yy - iy)**2 <= roi_r**2
        pixels = hu[mask]

        if len(pixels) == 0:
            continue

        meanHU_val = float(np.mean(pixels))
        sdHU = float(np.std(pixels))

        material = "Unknown"
        best_dist = np.inf
        for name, (lo, hi) in materialRanges.items():
            mid_ref = (lo + hi) / 2
            d = abs(meanHU_val - mid_ref)
            if lo <= meanHU_val <= hi and d < best_dist:
                material = name
                best_dist = d

        rows.append([ins["label"], material, round(meanHU_val, 2), round(sdHU, 2)])

        ix *= SCALE;  iy *= SCALE;  roi_r *= SCALE

        cv2.circle(output, (ix, iy), roi_r, (0, 255, 255), 2)
        cv2.putText(output, material[:5],
                    (ix - 60, iy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"{meanHU_val:.1f} HU",
                    (ix - 60, iy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    if not rows:
        return None, "❌ ไม่สามารถ sample ROI ได้"

    df = pd.DataFrame(rows, columns=["ROI", "Material", "HU", "SD"])

    referenceMaterial = ['Air', 'PMP', 'LDPE', 'Water', 'Polystyrene', 'Acrylic', 'Delrin', 'Teflon']
    density = np.array([0.00, 0.83, 0.92, 1.00, 1.03, 1.18, 1.42, 2.16])
    HU_ref_mid = np.array([-1016, -196, -104, 0, -47, 114, 365, 1000])
    mu_water = 0.170
    referenceMU = mu_water * (1 + HU_ref_mid / 1000)

    measuredHU = []
    for mat in referenceMaterial:
        idx_m = df.index[df["Material"] == mat]
        measuredHU.append(float(df.loc[idx_m[0], "HU"]) if len(idx_m) > 0 else np.nan)
    measuredHU = np.array(measuredHU)
    valid = ~np.isnan(measuredHU)

    R2 = None
    R2_density = None
    fig1 = None
    fig2 = None

    if valid.sum() >= 3:
        x_mu = referenceMU[valid]
        y_m = measuredHU[valid]

        p = np.polyfit(x_mu, y_m, 1)
        fit = np.polyval(p, x_mu)
        SSres = np.sum((y_m - fit)**2)
        SStot = np.sum((y_m - np.mean(y_m))**2)
        R2 = float(1 - SSres / SStot) if SStot > 0 else 0.0

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.set_facecolor("#0a1520")
        fig1.patch.set_facecolor("#0a1520")
        ax1.plot(x_mu, fit, color="#7b61ff", lw=2, label='Linear Fit')
        ax1.plot(x_mu, y_m, 'o', color="#00e5ff", ms=7, label='Measured')
        for xi, yi, lbl in zip(x_mu, y_m, [referenceMaterial[j] for j, v in enumerate(valid) if v]):
            ax1.annotate(lbl, (xi, yi-100), textcoords="offset points",
                         xytext=(4, 4), fontsize=7, color="#7ac8e0")
        ax1.set_xlabel("µ (cm⁻¹)", color="#7ac8e0")
        ax1.set_ylabel("HU", color="#7ac8e0")
        ax1.set_title(f"CT Number Linearity   R²={R2:.4f}", color="#d8e8ff", fontsize=11)
        ax1.tick_params(colors="#4a6080")
        ax1.spines[:].set_color("#1e2a3a")
        ax1.legend(facecolor="#0e1520", labelcolor="#d8e8ff", fontsize=8)
        ax1.grid(True, color="#1e2a3a", alpha=0.5)

        x2 = density[valid]
        p2 = np.polyfit(x2, y_m, 1)
        fit2 = np.polyval(p2, x2)
        SSres2 = np.sum((y_m - fit2)**2)
        SStot2 = np.sum((y_m - np.mean(y_m))**2)
        R2_density = float(1 - SSres2 / SStot2) if SStot2 > 0 else 0.0

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.set_facecolor("#0a1520")
        fig2.patch.set_facecolor("#0a1520")
        ax2.plot(x2, fit2, color="#7b61ff", lw=2, label='Linear Fit')
        ax2.plot(x2, y_m, 'o', color="#00e5ff", ms=7, label='Measured')
        for xi, yi, lbl in zip(x2, y_m, [referenceMaterial[j] for j, v in enumerate(valid) if v]):
            ax2.annotate(lbl, (xi, yi-100), textcoords="offset points",
                         xytext=(4, 4), fontsize=7, color="#7ac8e0")
        ax2.set_xlabel("Density (g/cm³)", color="#7ac8e0")
        ax2.set_ylabel("HU", color="#7ac8e0")
        ax2.set_title(f"CT Number vs Density   R²={R2_density:.4f}", color="#d8e8ff", fontsize=11)
        ax2.tick_params(colors="#4a6080")
        ax2.spines[:].set_color("#1e2a3a")
        ax2.legend(facecolor="#0e1520", labelcolor="#d8e8ff", fontsize=8)
        ax2.grid(True, color="#1e2a3a", alpha=0.5)

    is_pass = (
        R2 is not None and
        R2_density is not None and
        R2 >= 0.99 and
        R2_density >= 0.99
    )

    return {
        "image": output,
        "df": df,
        "R2": R2,
        "R2_density": R2_density,
        "fig1": fig1,
        "fig2": fig2,
        "pass": is_pass
    }, None


def run_slice_thickness_qc(hu, px, nominal, mode="fine", offset_x=0, offset_y=0):
    """
    Slice Thickness Accuracy — Fine ramp (0.25 mm/bead) & Coarse ramp (1 mm/bead).
    แปลงจาก MATLAB FinerampButton / CoarserampButton.

    Steps:
      A. สร้าง vertical line profile ตรงกลางภาพ (ความยาว 90 mm fine / 35 mm coarse)
      B. หา peak envelope + Butterworth smooth
      C. Tail detection (adaptive threshold)
      D. FWHM จาก interpolation (H1/H2)
      E. นับ bead peaks ในช่วง FWHM → slice thickness = num_beads × pitch
      F. PASS/FAIL vs nominal ± tolerance

    Returns: (result_dict, error_str)
    """
    from scipy.signal import butter, filtfilt, find_peaks

    h, w = hu.shape
    cx = int(w // 2 + offset_x)
    cy = int(h // 2 + offset_y)

    cx = max(0, min(w-1, cx))
    cy = max(0, min(h-1, cy))

    # ── A. สร้าง line profile แนวตั้งตรงกลาง ──
    line_mm  = 90.0 if mode == "fine" else 35.0
    line_px  = int(line_mm / px)
    # 🔥 สร้างเส้นเอียง (ใช้ angle)
    theta = np.deg2rad(angle)

    line_len = int(line_px)   # ใช้ความยาวเดิม

    x1 = int(cx - line_len/2 * np.cos(theta))
    y1 = int(cy - line_len/2 * np.sin(theta))

    x2 = int(cx + line_len/2 * np.cos(theta))
    y2 = int(cy + line_len/2 * np.sin(theta))

    # 🔥 sample profile
    num = line_len
    xs = np.linspace(x1, x2, num).astype(np.float32)
    ys = np.linspace(y1, y2, num).astype(np.float32)

    valid = (xs >= 1) & (xs < w-1) & (ys >= 1) & (ys < h-1)
    xs = xs[valid]
    ys = ys[valid]

    p_raw = cv2.remap(
        hu.astype(np.float32),
        xs.reshape(1,-1),
        ys.reshape(1,-1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    ).flatten()

    N = len(p_raw)
    px_vec = np.arange(N, dtype=np.float64)

    y_start  = max(0, cy - line_px // 2)
    y_end    = min(h - 1, cy + line_px // 2)


    if N < 10:
        return None, "❌ Profile too short — ภาพเล็กเกิน"

    px_vec = np.arange(N, dtype=np.float64)

    # ── B. Peak envelope + smooth ──
    peak_idx_raw = int(np.argmax(p_raw))
    win_l = max(0, peak_idx_raw - int(0.30 * N))
    win_r = min(N, peak_idx_raw + int(0.30 * N))
    p_crop = p_raw[win_l:win_r].copy()

    # Rolling max (MATLAB movmax)
    win_peak = 5
    peak_env = np.array([np.max(p_crop[max(0,i-win_peak//2):min(len(p_crop),i+win_peak//2+1)])
                          for i in range(len(p_crop))])

    # Normalize
    base_env  = np.min(peak_env)
    peak_norm = peak_env - base_env
    if np.max(peak_norm) == 0:
        return None, "❌ Flat profile — ไม่พบ ramp structure"
    peak_norm = peak_norm / np.max(peak_norm)

    # Butterworth lowpass
    cutoff = 0.08 if mode == "fine" else 0.12
    order  = 3     if mode == "fine" else 4
    try:
        b_f, a_f = butter(order, cutoff, btype="low")
        smooth_env = filtfilt(b_f, a_f, peak_norm)
    except Exception:
        smooth_env = peak_norm.copy()

    # Scale back to HU
    baseline_target = np.min(p_raw)
    peak_val_raw    = p_raw[peak_idx_raw]
    smooth_crop_hu  = smooth_env * (peak_val_raw - baseline_target) + baseline_target

    smooth_hu = np.empty(N)
    smooth_hu[:win_l]  = smooth_crop_hu[0]
    smooth_hu[win_l:win_r] = smooth_crop_hu
    smooth_hu[win_r:]  = smooth_crop_hu[-1]

    # ── C. Tail detection ──
    peak_val, peak_idx = float(np.max(smooth_hu)), int(np.argmax(smooth_hu))

    left_base  = float(np.median(smooth_hu[:max(1, int(0.15 * N))]))
    right_base = float(np.median(smooth_hu[min(N-1, int(0.85 * N)):]))
    tail_margin = 0.04 * (peak_val - min(left_base, right_base))
    th_tail     = min(left_base, right_base) + tail_margin

    # Left tail
    search_start_l = min(10, peak_idx)
    yL_search = smooth_hu[search_start_l:peak_idx + 1]
    below_l   = np.where(yL_search < th_tail)[0]
    left_tail  = int(below_l[0]  + search_start_l) if len(below_l)  > 0 else int(np.argmin(yL_search) + search_start_l)

    # Right tail
    yR_search = smooth_hu[peak_idx:]
    below_r   = np.where(yR_search < th_tail)[0]
    right_tail = int(below_r[0] + peak_idx) if len(below_r) > 0 else int(np.argmin(yR_search) + peak_idx)

    # ── D. Baseline + FWHM ──
    left_base2  = float(np.median(smooth_hu[:max(1, int(0.15 * N))]))
    right_base2 = float(np.median(smooth_hu[min(N-1, int(0.85 * N)):]))
    baseline    = (left_base2 + right_base2) / 2.0
    fwhm_level  = baseline + 0.5 * (peak_val - baseline)

    # Left crossing (H1)
    yL = smooth_hu[:peak_idx + 1]
    below_fwhm_l = np.where(yL < fwhm_level)[0]
    if len(below_fwhm_l) == 0:
        return None, "❌ Cannot find left FWHM crossing"
    idx_below_l = int(below_fwhm_l[-1])
    idx_above_l = idx_below_l + 1
    if idx_above_l >= len(yL):
        return None, "❌ Left FWHM crossing out of range"
    fwhm_left = idx_below_l + (fwhm_level - yL[idx_below_l]) * \
                (1.0) / (yL[idx_above_l] - yL[idx_below_l] + 1e-12)

    # Right crossing (H2)
    yR = smooth_hu[peak_idx:]
    below_fwhm_r = np.where(yR < fwhm_level)[0]
    if len(below_fwhm_r) == 0 or below_fwhm_r[0] == 0:
        return None, "❌ Cannot find right FWHM crossing"
    idx_below_r = int(below_fwhm_r[0])
    idx_above_r = idx_below_r - 1
    fwhm_right = (peak_idx + idx_above_r) + (fwhm_level - yR[idx_above_r]) * \
                 (1.0) / (yR[idx_below_r] - yR[idx_above_r] + 1e-12)

    fwhm_px   = fwhm_right - fwhm_left
    fwhm_mm   = fwhm_px * px

    # ── E. Peak detection in bead region → count beads in FWHM ──
    start_idx = max(0, left_tail)
    end_idx   = min(N - 1, right_tail)
    if end_idx <= start_idx:
        return None, "❌ Tail detection error — ปรับ line ให้ตัด ramp"

    p_bead   = p_raw[start_idx:end_idx + 1]
    min_prom = max(p_bead) * 0.02
    locs_rel, _ = find_peaks(p_bead, prominence=min_prom, distance=1)
    locs_global = locs_rel + start_idx

    beads_in_fwhm = locs_global[(locs_global >= fwhm_left) & (locs_global <= fwhm_right)]
    num_beads = len(beads_in_fwhm)

    # Pitch per mode
    pitch_mm = 0.25 if mode == "fine" else 1.0
    slice_mm = num_beads * pitch_mm if num_beads >= 1 else float("nan")

    # ── F. PASS / FAIL ──
    nom = float(nominal)

    # tolerance (เหมือน MATLAB)
    if nom <= 1.0:
        tol_min = 0.0
        tol_max = nom + 0.5
    elif nom <= 2.0:
        tol_min = 0.5 * nom
        tol_max = 1.5 * nom
    else:
        tol_min = nom - 1.0
        tol_max = nom + 1.0

    # ใช้ bead-based เท่านั้น (สำคัญ!)
    if num_beads >= 1:
        slice_mm_bead = num_beads * pitch_mm
    else:
        slice_mm_bead = float("nan")

    # PASS / FAIL
    if not np.isnan(slice_mm_bead) and (tol_min <= slice_mm_bead <= tol_max):
        is_pass = True
    else:
        is_pass = False

    # ── G. Matplotlib figure ──
    dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    # Left: image with profile line
    ax_img = axes[0]
    # WL=300 WW=1500 (เหมือน MATLAB)
    img_disp = wl_ww(hu, 300, 1500)
    ax_img.imshow(img_disp, cmap="gray", vmin=0, vmax=255)
    ax_img.plot([x1, x2], [y1, y2], '--', color="#00e5ff", lw=1.5, label="Profile line")
    ax_img.axhline(y=y_start, color="#ffb74d", lw=0.8, ls=":")
    ax_img.axhline(y=y_end,   color="#ffb74d", lw=0.8, ls=":")
    ax_img.set_title(f"{'Fine' if mode=='fine' else 'Coarse'} Ramp — Profile Line",
                     color=MUTED, fontsize=9)
    ax_img.axis("off")

    # Right: HU profile + FWHM annotations
    ax_p = axes[1]
    ax_p.set_facecolor(SURF)
    ax_p.plot(px_vec, p_raw,    color="#3a5060", lw=1,   alpha=0.7, label="Raw")
    ax_p.plot(px_vec, smooth_hu, color="#00e5ff", lw=1.8, label="Smoothed")
    ax_p.axhline(baseline,   color="white",   lw=0.8, ls="--", label="Baseline")
    ax_p.axhline(fwhm_level, color="#aaaaaa", lw=0.8, ls="--", label="FWHM level")

    ax_p.axvline(left_tail,  color="#00e676", lw=1,   ls="--", alpha=0.7)
    ax_p.axvline(right_tail, color="#00e676", lw=1,   ls="--", alpha=0.7, label="Tails")
    ax_p.axvline(fwhm_left,  color="#7b61ff", lw=1.5, ls="--")
    ax_p.axvline(fwhm_right, color="#7b61ff", lw=1.5, ls="--", label="FWHM")

    # FWHM bar
    ax_p.hlines(fwhm_level, fwhm_left, fwhm_right, color="#ff5252", lw=2.5)
    ax_p.annotate(f"H1={fwhm_left:.1f}", xy=(fwhm_left, fwhm_level),
                  xytext=(fwhm_left + 1, fwhm_level - (peak_val - baseline) * 0.12),
                  color="#7b61ff", fontsize=7, fontweight="bold")
    ax_p.annotate(f"H2={fwhm_right:.1f}", xy=(fwhm_right, fwhm_level),
                  xytext=(fwhm_right - 14, fwhm_level - (peak_val - baseline) * 0.12),
                  color="#7b61ff", fontsize=7, fontweight="bold")

    # Bead peaks inside FWHM
    if len(beads_in_fwhm) > 0:
        ax_p.plot(beads_in_fwhm, p_raw[beads_in_fwhm],
                  "ro", ms=5, lw=1.5, label=f"Beads ({num_beads})")

    mode_label = "Fine ramp (0.25 mm/bead)" if mode == "fine" else "Coarse ramp (1 mm/bead)"
    slice_txt  = f"{slice_mm:.2f} mm" if not np.isnan(slice_mm) else "N/A"
    ax_p.set_title(f"{mode_label}\nSlice Thickness = {slice_txt}",
                   color="#d8e8ff", fontsize=9)
    ax_p.set_xlabel("Pixel", color=MUTED, fontsize=8)
    ax_p.set_ylabel("HU", color=MUTED, fontsize=8)
    ax_p.tick_params(colors=MUTED)
    ax_p.spines[:].set_color("#1e2a3a")
    ax_p.legend(fontsize=7, facecolor=DARK, labelcolor="#d8e8ff", loc="upper right")
    ax_p.grid(True, color="#1e2a3a", alpha=0.5)

    plt.tight_layout(pad=0.8)

    # แปลง fig เป็น image array
    buf_st = io.BytesIO()
    fig.savefig(buf_st, format="png", dpi=150, bbox_inches="tight")
    buf_st.seek(0)
    from PIL import Image as _PIL
    _img_arr = np.array(_PIL.open(buf_st))
    buf_st.close()

    return {
        "fig":        fig,
        "img_arr":    _img_arr,    
        "mode":       mode,
        "slice_mm":   slice_mm_bead, 
        "fwhm_mm":    fwhm_mm,
        "num_beads":  num_beads,
        "fwhm_left":  fwhm_left,
        "fwhm_right": fwhm_right,
        "nominal":    nom,
        "tol_min":    tol_min,
        "tol_max":    tol_max,
        "pass":       is_pass,
        "pitch_mm":   pitch_mm,
        "y_start":    y_start,
        "y_end":      y_end,
    }, None

def reset_zoom():
    st.session_state["coarse_zoom"] = 1.0

st.number_input(
    "🔍 Zoom",
    min_value=1.0,
    max_value=8.0,
    step=0.5,
    key="coarse_zoom"
)

st.button("🔄 Reset Zoom", on_click=reset_zoom)

def step_icon(done):
    return "🟢 ✓" if done else "⚪"

def compute_spatial_resolution(app, lp_value):
    """
    lp_value = line pair per cm (เช่น 5, 6, 7)
    """

    if not hasattr(app, "profiles") or app.profiles is None:
        st.error("No profiles available")
        return

    profiles = app.profiles

    # ===== group selection =====
    group_size = 3
    idx_start = (lp_value - 1) * group_size
    idx_end   = min(lp_value * group_size, profiles.shape[0])

    group_profiles = profiles[idx_start:idx_end]

    # ===== mean profile =====
    y_raw = np.mean(group_profiles, axis=0)

    # smoothing (เหมือน MATLAB movmean)
    y_plot = np.convolve(y_raw, np.ones(3)/3, mode='same')

    # ===== Michelson contrast =====
    Imax = np.max(y_plot)
    Imin = np.min(y_plot)

    MTF_point = (Imax - Imin) / (Imax + Imin + 1e-8)

    # ===== plot profile =====
    fig1, ax1 = plt.subplots()
    ax1.plot(y_plot, 'b-')
    ax1.set_title(f'Profile ({lp_value} lp/cm) → MTF = {MTF_point:.3f}')
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('Intensity')
    ax1.grid()

    return {
        "MTF_point": MTF_point,
        "profile_fig": fig1
    }

def compute_mtf_curve(app, profile):

    pixel_size_mm = app.pixel_spacing  # จาก DICOM

    y = profile.copy()

    # ===== background removal =====
    N = len(y)
    edge = max(5, int(0.1 * N))
    bg = (np.mean(y[:edge]) + np.mean(y[-edge:])) / 2
    y = y - bg

    y = y - np.min(y)

    # ===== window =====
    win = np.hanning(N)
    y = y * win

    # ===== FFT =====
    Nfft = 4 * N
    Y = np.abs(np.fft.fft(y, Nfft))
    Y = Y[:Nfft//2]

    mtf = Y / Y[0]

    # ===== frequency axis =====
    f_cyc_mm = np.arange(Nfft//2) / (Nfft * pixel_size_mm)
    f_lp_cm = f_cyc_mm * 10

    return f_lp_cm, mtf

def find_mtf_metrics(f, mtf):

    def find_point(level):
        idx = np.where(mtf <= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        return np.interp(level, [mtf[i-1], mtf[i]], [f[i-1], f[i]])

    f50 = find_point(0.5)
    f10 = find_point(0.1)

    return f50, f10
# ══════════════════════════════════════════════════
#  UNIFORMITY / CT ACCURACY / NOISE QC  (from MATLAB)
# ══════════════════════════════════════════════════

def _draw_label_box(img, lines_txt, x, y, bg_color=(0,0,0), txt_color=(0,255,255)):
    """วาด multi-line label box บนภาพ CV2"""
    fsc, thick = 1.0, 3
    line_h = 40
    # measure max width
    max_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fsc, thick)[0][0]
                for t in lines_txt)
    box_h = line_h * len(lines_txt) + 4
    bx = x - max_w // 2 - 2
    by = y - box_h + 2
    cv2.rectangle(img, (bx, by), (bx + max_w + 4, y + 4), bg_color, -1)
    for j, txt in enumerate(lines_txt):
        ty = by + 4 + line_h * (j + 1) - 2
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fsc, thick)
        tx = x - tw // 2
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fsc, txt_color, thick)

def refine_outer_radius_strong(hu, cx, cy, r_init):

    h, w = hu.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    r_vals = np.arange(int(r_init*0.7), int(r_init*1.2))

    profile = []
    for r in r_vals:
        band = (rr > r-1) & (rr < r+1)
        if np.sum(band) < 50:
            profile.append(0)
        else:
            profile.append(np.mean(hu[band]))

    profile = np.array(profile)

    # 🔥 หา gradient (edge position)
    grad = np.abs(np.gradient(profile))

    best_r = r_vals[np.argmax(grad)]

    return int(best_r)

def _find_inner_radius_robust(hu, cx, cy, r_phantom):
    """
    Robust inner-ring detection using multi-angle radial gradient.
    Returns inner radius in pixels.
    """
    import numpy as np
    import cv2

    h, w = hu.shape
    img = cv2.GaussianBlur(hu.astype(np.float32), (5, 5), 1.5)

    r_min = int(r_phantom * 0.45)
    r_max = int(r_phantom * 0.90)

    if r_max <= r_min + 5:
        return int(r_phantom * 0.75)

    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    radii_found = []

    for th in angles:
        rs = np.arange(r_min, r_max, 1, dtype=np.float32)
        xs = cx + rs * np.cos(th)
        ys = cy + rs * np.sin(th)

        valid = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
        if np.sum(valid) < 20:
            continue

        xs = xs[valid]
        ys = ys[valid]
        rs = rs[valid]

        vals = cv2.remap(
            img,
            xs.reshape(1, -1).astype(np.float32),
            ys.reshape(1, -1).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        ).flatten()

        if len(vals) < 10:
            continue

        vals_s = cv2.GaussianBlur(vals.reshape(-1, 1), (1, 9), 0).flatten()
        grad = np.abs(np.gradient(vals_s))

        idx = int(np.argmax(grad))
        if grad[idx] < 2.0:
            continue

        radii_found.append(rs[idx])

    if len(radii_found) < 20:
        return int(r_phantom * 0.75)

    radii_found = np.array(radii_found, dtype=np.float32)

    med = np.median(radii_found)
    mad = np.median(np.abs(radii_found - med)) + 1e-6
    keep = np.abs(radii_found - med) < 3.0 * mad

    good = radii_found[keep]
    if len(good) < 10:
        r_final = med
    else:
        r_final = np.median(good)

    if r_final < r_phantom * 0.45 or r_final > r_phantom * 0.95:
        return int(r_phantom * 0.75)

    return int(round(r_final))

def run_ct_accuracy_qc(hu, px):
    """
    CT Number Accuracy + Uniformity — layout เหมือน MATLAB:
    - Outer boundary: วง cyan
    - Inner disk boundary: วงขาว
    - Noise ROI: วงแดง 40% ของ inner_r
    - ROI1 center: วง cyan เล็ก
    - ROI2-5 peripheral: วงขาวเล็ก 4 ทิศ ที่ 80% ของ inner_r
    """
    import math
    try:
        cx, cy, r_phantom = _basic_find_phantom(hu)
    except Exception as e:
        return None, f"❌ Phantom detection error: {e}"

    cx, cy, r_phantom = int(cx), int(cy), int(r_phantom)
    r_phantom = refine_outer_radius_strong(hu, cx, cy, r_phantom)
    inner_r = _find_inner_radius_robust(hu, cx, cy, r_phantom)

    h, w    = hu.shape
    roi_r   = max(6, int(10.0 / px))       # ROI radius ~20 mm
    offset  = int(0.8 * inner_r)           # 80% inner radius offset
    noise_r = max(10, int(inner_r * 0.40)) # Noise ROI = 40% inner radius

    # MATLAB angles = [0 -pi/2 pi pi/2] → Right / Top / Left / Bottom
    rois = [
        ("ROI1", cx,           cy,           (0,255,255), (0,0,0)),
        ("ROI2", cx + offset,  cy,           (255,255,255),(0,0,0)),
        ("ROI3", cx,           cy + offset,  (255,255,255),(0,0,0)),
        ("ROI4", cx - offset,  cy,           (255,255,255),(0,0,0)),
        ("ROI5", cx,           cy - offset,  (255,255,255),(0,0,0)),
    ]

    SCALE = 3
    img_norm = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output   = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    output   = cv2.resize(output, (output.shape[1]*SCALE, output.shape[0]*SCALE),
                          interpolation=cv2.INTER_LINEAR)

    cx *= SCALE;  cy *= SCALE
    r_phantom *= SCALE;  inner_r *= SCALE;  noise_r *= SCALE
    roi_r *= SCALE;  offset *= SCALE

    cv2.circle(output, (cx, cy), r_phantom, (255, 255, 0),  SCALE)
    cv2.circle(output, (cx, cy), inner_r,   (255, 255, 255), SCALE)

    rows    = []
    hu_vals = {}

    for name, rx, ry, circle_col, bg_col in rois:
        # rx, ry ยังเป็น original coordinate → scale ก่อน
        rx_s = int(rx * SCALE)
        ry_s = int(ry * SCALE)
        rx_s = max(roi_r, min(output.shape[1] - 1 - roi_r, rx_s))
        ry_s = max(roi_r, min(output.shape[0] - 1 - roi_r, ry_s))

        # sample HU จาก original coordinate
        rx_orig    = int(rx)
        ry_orig    = int(ry)
        roi_r_orig = roi_r // SCALE
        
        mask   = np.zeros_like(hu, dtype=np.uint8)
        cv2.circle(mask, (rx_orig, ry_orig), roi_r_orig, 1, -1)
        pixels = hu[mask == 1]
        if len(pixels) == 0:
            continue
        mu = float(np.mean(pixels))
        sd = float(np.std(pixels))
        hu_vals[name] = {"mean": mu, "sd": sd}
        rows.append([name, round(mu, 2), round(sd, 2)])

        cv2.circle(output, (rx_s, ry_s), roi_r, circle_col, SCALE)
        label_offset = int(roi_r * 1.4)
        _draw_label_box(output,
                [name, f"Mean:{mu:.2f}", f"SD:{sd:.2f}"],
                rx_s, ry_s+130 + label_offset,
                bg_color=(0, 0, 0),
                txt_color=(0, 255, 255))

    df = pd.DataFrame(rows, columns=["ROI", "Mean HU", "SD"])

    center_hu = (hu_vals.get("ROI1") or {}).get("mean", np.nan)
    tol_acc   = 20.0
    acc_pass  = (abs(center_hu) <= tol_acc) if not np.isnan(center_hu) else False

    peripherals = [(hu_vals.get(k) or {}).get("mean", np.nan)
                   for k in ("ROI2","ROI3","ROI4","ROI5")]
    peripherals = [p for p in peripherals if not np.isnan(p)]
    if peripherals and not np.isnan(center_hu):
        max_diff = float(max(abs(p - center_hu) for p in peripherals))
        uni_pass = max_diff <= 5.0
    else:
        max_diff = np.nan
        uni_pass = False

    return {
        "image":     output,
        "df":        df,
        "hu_vals":   hu_vals,
        "center_hu": center_hu,
        "max_diff":  max_diff,
        "acc_pass":  acc_pass,
        "uni_pass":  uni_pass,
        "tol_acc":   tol_acc,
        "inner_r":   inner_r,
    }, None

def refine_outer_radius(hu, cx, cy, r_init):
    h, w = hu.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    band = (rr > r_init*0.85) & (rr < r_init*1.15)
    values = hu[band]

    # ใช้ gradient-based threshold
    thresh = np.percentile(values, 85)

    edge_mask = (hu >= thresh)
    edge_rr = rr[edge_mask]

    if len(edge_rr) == 0:
        return r_init

    return int(np.median(edge_rr))

def run_noise_qc(hu, px, baseline_noise=None):
    """
    Noise QC — ใช้ outer + inner detection แบบ robust
    """

    import math
    import numpy as np
    import cv2

    # ─────────────────────────
    # 1. Detect phantom (initial)
    # ─────────────────────────
    try:
        cx, cy, r_phantom = _basic_find_phantom(hu)
    except Exception as e:
        return None, f"❌ Phantom detection error: {e}"

    cx, cy, r_phantom = int(cx), int(cy), int(r_phantom)

    # ─────────────────────────
    # 2. 🔵 refine outer (สำคัญ)
    # ─────────────────────────
    r_phantom = refine_outer_radius_strong(hu, cx, cy, r_phantom)

    # ─────────────────────────
    # 3. ⚪ detect inner
    # ─────────────────────────
    inner_r = _find_inner_radius_robust(hu, cx, cy, r_phantom)

    # ─────────────────────────
    # 4. Noise ROI
    # ─────────────────────────
    noise_r = max(10, int(inner_r * 0.40))

    cx = max(noise_r, min(hu.shape[1] - 1 - noise_r, cx))
    cy = max(noise_r, min(hu.shape[0] - 1 - noise_r, cy))

    mask = np.zeros_like(hu, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), noise_r, 1, -1)

    pixels = hu[mask == 1]
    if len(pixels) == 0:
        return None, "❌ Noise ROI empty"

    measured_noise = float(np.std(pixels))
    mean_noise     = float(np.mean(pixels))

    # ─────────────────────────
    # 5. Visualization
    # ─────────────────────────
    SCALE = 3
    img_norm = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output   = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    output   = cv2.resize(output, (output.shape[1]*SCALE, output.shape[0]*SCALE),
                          interpolation=cv2.INTER_LINEAR)

    cxs = cx * SCALE;  cys = cy * SCALE
    r_phantom_s = r_phantom * SCALE
    inner_r_s   = inner_r   * SCALE
    noise_r_s   = noise_r   * SCALE

    cv2.circle(output, (cxs, cys), r_phantom_s, (255,255,0),   SCALE)
    cv2.circle(output, (cxs, cys), inner_r_s,   (255,255,255), SCALE)
    cv2.circle(output, (cxs, cys), noise_r_s,   (0,0,255),     SCALE)
    # ─────────────────────────
    # 6. Label
    # ─────────────────────────
    angle_5 = math.radians(-90)
    nx = int(cxs + inner_r_s * math.cos(angle_5))
    ny = int(cys - inner_r_s * math.sin(angle_5))
    _draw_label_box(
        output,
        ["Noise ROI", f"Mean: {mean_noise:.2f} HU", f"SD: {measured_noise:.2f} HU"],
        nx, ny-100,
        bg_color=(180, 0, 0),
        txt_color=(255,255,255)
    )
    # ─────────────────────────
    # 7. Evaluation
    # ─────────────────────────
    if baseline_noise is None or baseline_noise <= 0:
        return {
            "image": output,
            "measured": measured_noise,
            "mean": mean_noise,
            "baseline": None,
            "abs_diff": None,
            "rel_diff": None,
            "min_accept": None,
            "max_accept": None,
            "pass": None,
            "inner_r": inner_r,
            "outer_r": r_phantom,
        }, None

    tol        = 0.25
    tol_hu     = tol * baseline_noise
    min_accept = baseline_noise - tol_hu
    max_accept = baseline_noise + tol_hu

    abs_diff   = measured_noise - baseline_noise
    rel_diff   = 100 * abs_diff / baseline_noise

    noise_pass = min_accept <= measured_noise <= max_accept

    return {
        "image": output,
        "measured": measured_noise,
        "mean": mean_noise,
        "baseline": baseline_noise,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "min_accept": min_accept,
        "max_accept": max_accept,
        "pass": noise_pass,
        "inner_r": inner_r,
        "outer_r": r_phantom,
    }, None


def compute_nps_patch(roi, px):

    roi = roi.astype(np.float32)
    # remove DC
    roi = roi - np.mean(roi)
    # Hanning window
    w = np.hanning(roi.shape[0])
    window = np.outer(w, w)
    roi = roi * window
    fft = np.fft.fftshift(np.fft.fft2(roi))
    ps = np.abs(fft)**2
    N = roi.shape[0]
    nps = (px**2 / (N*N)) * ps

    return nps

def get_nps_rois(img, patch):

    rois = []

    h, w = img.shape
    cx = int(w // 2 + offset_x)
    cy = int(h // 2 + offset_y)

    cx = max(0, min(w-1, cx))
    cy = max(0, min(h-1, cy))

    step = patch

    offsets = [
        (0,0),        # center
        (-step,0),    # left
        (step,0),     # right
        (0,-step),    # up
        (0,step)      # down
    ]

    for dx,dy in offsets:

        x = cx + dx - patch//2
        y = cy + dy - patch//2

        roi = img[y:y+patch, x:x+patch]

        if roi.shape == (patch,patch):
            rois.append((roi,(x,y)))

    return rois

# ─────────────────────────
# Detect phantom circle
# ─────────────────────────
def detect_phantom_inner(hu):

    img = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=120,
        maxRadius=300
    )

    if circles is None:
        return None

    x, y, r = np.round(circles[0][0]).astype(int)
    return x, y, r


# ─────────────────────────────────────────
# refine_insert_radius
# ─────────────────────────────────────────
def refine_insert_radius(hu, cx, cy, px):
    """
    หา edge radius โดย scan ทุกทิศ แล้วเฉลี่ย
    ใช้ gradient normalized เพื่อจับ insert contrast ต่ำด้วย
    """
    max_r = int(22 / px)
    angles = np.linspace(0, 2 * np.pi, 72)  # ละเอียดขึ้น
    edges = []

    for ang in angles:
        profile = []
        for r in range(2, max_r):
            x = int(cx + r * np.cos(ang))
            y = int(cy + r * np.sin(ang))
            if x < 0 or y < 0 or x >= hu.shape[1] or y >= hu.shape[0]:
                break
            profile.append(hu[y, x])

        if len(profile) < 6:
            continue

        profile = np.array(profile, dtype=np.float32)

        # normalize gradient ด้วย local std → จับ contrast ต่ำได้
        grad = np.abs(np.gradient(profile))
        local_std = np.std(profile) + 1e-6
        grad_norm = grad / local_std

        edge_r = int(np.argmax(grad_norm)) + 2
        edges.append(edge_r)

    if len(edges) < 8:
        return int(6 / px)  # fallback

    # ตัด outlier ออก (วงที่ไม่ใช่ insert จริง)
    edges = np.array(edges)
    med = np.median(edges)
    edges = edges[np.abs(edges - med) < med * 0.4]

    return int(np.median(edges)) if len(edges) > 0 else int(6 / px)

# ─────────────────────────────────────────
# refine_insert_center
# ─────────────────────────────────────────
def refine_insert_center(hu, x0, y0, r):

    y, x = np.ogrid[:hu.shape[0], :hu.shape[1]]
    mask = (x - x0) ** 2 + (y - y0) ** 2 <= (r * 1.5) ** 2

    roi = hu.copy()
    roi[~mask] = hu.mean()

    img = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(img, 50, 150)
    pts = np.column_stack(np.where(edges > 0))

    if len(pts) < 20:
        return x0, y0

    cy_r, cx_r = pts.mean(axis=0)
    return int(cx_r), int(cy_r)


import numpy as np
import cv2


# ─────────────────────────
# Detect phantom circle
# ─────────────────────────
def detect_linearity_inserts(hu, px):
    """
    Detect all 8 CTP404 linearity inserts.

    Fixes:
    1. ใช้ Hough Circle แทน SimpleBlobDetector
       → จับได้ทั้ง insert สว่าง (Teflon/Delrin) และมืด (Air/PMP)
    2. ขยาย ring distance range จาก (0.45–0.75) → (0.35–0.85)
       → ไม่ตัด insert ที่อยู่ขอบนอก/ในเกินไป
    3. เลือก 8 วงโดย spatial uniqueness แทน |HU| ranking
       → ไม่พลาดวงที่ HU ใกล้ background
    4. fallback multi-threshold blob สำหรับวงที่ Hough พลาด
    """

    phantom = detect_phantom_inner(hu)
    if phantom is None:
        return []

    cx, cy, r = phantom

    # ── preprocess ──────────────────────────────────────────
    img = cv2.normalize(hu.astype(np.float32), None, 0, 255,
                        cv2.NORM_MINMAX).astype(np.uint8)

    # ✅ CLAHE เพิ่ม contrast ให้วง contrast ต่ำมองเห็นได้
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    img_blur  = cv2.GaussianBlur(img_clahe, (5, 5), 1)

    # ── insert radius ประมาณ 12.2 mm ────────────────────────
    insert_r_px = (12.2 / 2) / px
    r_min = max(4, int(insert_r_px * 0.5))
    r_max = int(insert_r_px * 1.6)

    # ── Hough Circle ─────────────────────────────────────────
    # ✅ FIX: param2=12 (ต่ำพอจับวง contrast ต่ำ) แทน default สูง
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(insert_r_px * 1.4),
        param1=50,
        param2=12,
        minRadius=r_min,
        maxRadius=r_max
    )

    candidates = []

    if circles is not None:
        for (x0, y0, rc) in circles[0]:
            x0, y0 = int(x0), int(y0)
            dist = np.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2)

            # ✅ FIX: ขยาย ring zone จาก 0.45–0.75 → 0.35–0.85
            if not (0.35 * r <= dist <= 0.85 * r):
                continue

            candidates.append((x0, y0, rc))

    # ── fallback: multi-threshold blob ───────────────────────
    # ✅ FIX: ใช้ threshold HU หลายระดับจับวงที่ Hough พลาด
    candidates = _add_blob_candidates(hu, px, cx, cy, r,
                                      insert_r_px, candidates)

    # ── fallback 2: local variance map ──────────────────────
    # ✅ NEW: จับวง contrast ต่ำ (Polystyrene ~-35 HU)
    #         ที่ทั้ง Hough และ threshold blob ตรวจไม่เจอ
    if len(candidates) < 8:
        candidates = _detect_by_variance(hu, px, cx, cy, r,
                                         insert_r_px, candidates)

    # ── NMS: กำจัดวงซ้ำ ─────────────────────────────────────
    candidates = _nms(candidates, min_dist=insert_r_px * 1.3)

    # ── เลือก 8 วง โดย prefer วงที่อยู่ใน insert ring ────────
    # ✅ FIX: sort โดย distance ห่างจาก expected ring radius
    #         แทนการ sort โดย |HU| ที่จะพลาดวง HU ใกล้ background
    expected_ring_px = 0.58 * r  # insert อยู่ที่ ~58% ของ phantom radius
    candidates = sorted(
        candidates,
        key=lambda d: abs(np.sqrt((d[0]-cx)**2 + (d[1]-cy)**2) - expected_ring_px)
    )[:8]

    # ── คำนวณ HU stats ───────────────────────────────────────
    roi_r_px = int(8.5 / px)
    yy, xx = np.mgrid[0:hu.shape[0], 0:hu.shape[1]]
    inserts = []

    for (x0, y0, rc) in candidates:
        r_edge = refine_insert_radius(hu, x0, y0, px)
        xc, yc = refine_insert_center(hu, x0, y0, r_edge)

        roi_r = min(roi_r_px, max(3, int(r_edge * 0.75)))
        mask  = (xx - xc) ** 2 + (yy - yc) ** 2 <= roi_r ** 2
        vals  = hu[mask]

        if vals.size == 0:
            continue

        inserts.append(dict(
            cx=xc,
            cy=yc,
            r_px=r_edge,
            hu_mean=float(vals.mean()),
            hu_sd=float(vals.std()),
            label=0
        ))

    # ── เรียง clockwise จาก 12 นาฬิกา ─────────────────────────
    inserts = sorted(
        inserts,
        key=lambda d: (np.arctan2(d["cy"] - cy, d["cx"] - cx) + np.pi / 2) % (2 * np.pi)
    )

    for i, ins in enumerate(inserts):
        ins["label"] = i + 1

    return inserts


# ──────────────────────────────────────────────────────────
# Helper: Multi-threshold blob (fallback)
# ──────────────────────────────────────────────────────────
def _add_blob_candidates(hu, px, cx, cy, phantom_r, insert_r_px, existing):
    """เพิ่ม candidate จาก HU threshold เพื่อจับวงที่ Hough พลาด"""
    from scipy import ndimage

    area_min = np.pi * (insert_r_px * 0.4) ** 2
    area_max = np.pi * (insert_r_px * 1.6) ** 2

    thresholds = [
        hu < -700,                          # Air
        (hu > -500) & (hu < -50),           # PMP / LDPE
        (hu > 50)   & (hu < 200),           # Acrylic
        hu > 250,                            # Delrin / Teflon
        # ✅ NEW: Polystyrene ~-35 HU ใกล้ background มาก
        #         ใช้ range กว้างขึ้นเพื่อครอบ
        (hu > -80)  & (hu < 10),            # Polystyrene / low-contrast
    ]

    extras = []

    for binary in thresholds:
        labeled, n = ndimage.label(binary)
        for lid in range(1, n + 1):
            region = labeled == lid
            area   = region.sum()
            if not (area_min <= area <= area_max):
                continue

            ys, xs = np.where(region)
            bx, by = float(xs.mean()), float(ys.mean())

            dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
            if not (0.35 * phantom_r <= dist <= 0.85 * phantom_r):
                continue

            # ตรวจซ้ำ
            too_close = any(
                np.sqrt((bx - ex[0]) ** 2 + (by - ex[1]) ** 2) < insert_r_px * 1.3
                for ex in existing + extras
            )
            if not too_close:
                extras.append((bx, by, insert_r_px))

    return existing + extras


# ──────────────────────────────────────────────────────────
# Helper: Local Variance Map (สำหรับวง contrast ต่ำมาก)
# ──────────────────────────────────────────────────────────
def _detect_by_variance(hu, px, cx, cy, phantom_r, insert_r_px, existing):
    """
    ใช้ local variance map ตรวจหา insert ที่ HU ใกล้ background
    เช่น Polystyrene (~-35 HU) ที่ Hough และ threshold พลาด

    วิธี: วง insert มี texture/noise ต่างจาก background
          → local variance สูงกว่าบริเวณโดยรอบ
    """
    from scipy.ndimage import uniform_filter

    # คำนวณ local variance ในหน้าต่างขนาด insert
    win = max(3, int(insert_r_px * 0.8))
    if win % 2 == 0:
        win += 1

    hu_f   = hu.astype(np.float64)
    mean_  = uniform_filter(hu_f,        size=win)
    mean2_ = uniform_filter(hu_f ** 2,   size=win)
    var_map = mean2_ - mean_ ** 2
    var_map = np.sqrt(np.clip(var_map, 0, None))  # std map

    # normalize → uint8
    var_norm = cv2.normalize(var_map, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE เพิ่ม contrast ของ variance map
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    var_clahe = clahe.apply(var_norm)
    var_blur  = cv2.GaussianBlur(var_clahe, (5, 5), 1)

    # Hough Circle บน variance map
    r_min = max(4, int(insert_r_px * 0.5))
    r_max = int(insert_r_px * 1.6)

    circles = cv2.HoughCircles(
        var_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(insert_r_px * 1.4),
        param1=40,
        param2=10,       # ต่ำมากเพื่อจับ edge อ่อนๆ
        minRadius=r_min,
        maxRadius=r_max
    )

    extras = []

    if circles is not None:
        for (x0, y0, rc) in circles[0]:
            x0, y0 = float(x0), float(y0)
            dist = np.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2)

            if not (0.35 * phantom_r <= dist <= 0.85 * phantom_r):
                continue

            # ไม่ซ้ำกับที่มีอยู่แล้ว
            too_close = any(
                np.sqrt((x0 - ex[0]) ** 2 + (y0 - ex[1]) ** 2) < insert_r_px * 1.3
                for ex in existing + extras
            )
            if not too_close:
                extras.append((x0, y0, rc))

    return existing + extras


# ──────────────────────────────────────────────────────────
# Helper: Non-Maximum Suppression
# ──────────────────────────────────────────────────────────
def _nms(candidates, min_dist):
    """กำจัด candidate ที่ซ้อนกัน เลือกตัวที่ radius ใหญ่กว่าไว้"""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda d: d[2], reverse=True)
    kept  = []
    used  = [False] * len(candidates)

    for i, (x1, y1, r1) in enumerate(candidates):
        if used[i]:
            continue
        kept.append((x1, y1, r1))
        for j, (x2, y2, r2) in enumerate(candidates):
            if i != j and not used[j]:
                if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < min_dist:
                    used[j] = True

    return kept


# ─────────────────────────────────────────
# ส่วนที่ไม่เปลี่ยน (คงเดิม)
# ─────────────────────────────────────────
def detect_linearity_slices(slices):

    scores = []

    for i, s in enumerate(slices):
        hu   = s["hu_orig"]
        gx   = np.gradient(hu, axis=1)
        gy   = np.gradient(hu, axis=0)
        grad = np.mean(np.sqrt(gx ** 2 + gy ** 2))
        var  = np.var(hu)
        score = grad * 0.7 + var * 0.3
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:15]]


def refine_circle_edge(hu, cx, cy, r_init, px):

    max_search = int(6 / px)
    r_vals     = np.arange(r_init - max_search, r_init + max_search, 1)
    profile    = []

    for r in r_vals:
        mask = np.zeros_like(hu, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r), 1, 1)
        vals = hu[mask == 1]
        profile.append(np.mean(vals) if len(vals) > 0 else 0)

    profile = np.array(profile)
    grad    = np.abs(np.gradient(profile))
    idx     = np.argmax(grad)
    return r_vals[idx]


def detect_inner_module(hu, cx, cy, r):

    img   = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.GaussianBlur(img, (5, 5), 1.5)

    mask = np.zeros_like(img)
    cv2.circle(mask, (cx, cy), int(r * 0.50), 255, -1)
    img_mask = cv2.bitwise_and(img, img, mask=mask)

    circles = cv2.HoughCircles(
        img_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=28,
        param1=80,
        param2=8,
        minRadius=6,
        maxRadius=16
    )

    if circles is None:
        return []

    return np.round(circles[0]).astype(int)

# ─────────────────────────────────────────────────────────────
# sigmoid — บังคับ k > 0 เสมอ (ผ่าน bounds)
# ─────────────────────────────────────────────────────────────
def sigmoid(x, a, x0, k, b):
    """
    ESF sigmoid model: a / (1 + exp(-(x-x0)/k)) + b

    k ต้องเป็น positive เสมอ (enforce ผ่าน bounds ตอน curve_fit)
    → ป้องกัน PVE_width ติดลบ
    """
    k = np.abs(k)   # safety clamp กัน overflow
    return a / (1 + np.exp(-(x - x0) / k)) + b


# ─────────────────────────────────────────────────────────────
# radial_profile — FIXED
# ─────────────────────────────────────────────────────────────
def radial_profile(hu, cx, cy, max_r):
    """
    Full 360° radial ESF จาก insert center ออกไป
    sample ทุก pixel ในรัศมี max_r แล้วเฉลี่ยต่อ bin
    """
    h, w = hu.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - float(cx))**2 + (yy - float(cy))**2)

    profile_sum   = np.zeros(max_r, dtype=np.float64)
    profile_count = np.zeros(max_r, dtype=np.float64)

    mask  = dist < max_r
    r_idx = dist[mask].astype(np.int32)
    vals  = hu[mask]

    np.add.at(profile_sum,   r_idx, vals)
    np.add.at(profile_count, r_idx, 1)

    valid = profile_count > 0
    profile = np.where(
        valid,
        profile_sum / np.where(valid, profile_count, 1),
        np.nan
    )

    if np.any(~valid):
        x_all = np.arange(max_r)
        profile = np.interp(x_all, x_all[valid], profile[valid])

    return profile


# ─────────────────────────────────────────────────────────────
# compute_pve  — คำนวณ PVE width, MTF50, MTF10 ต่อ insert
# ─────────────────────────────────────────────────────────────
def compute_pve(hu, cx, cy, ps, r_insert_px=None, max_r=60):
    if r_insert_px is not None:
        max_r = int(r_insert_px * 2.5) + 10
    """
    Parameters
    ----------
    hu    : 2D HU array
    cx,cy : insert center (pixel)
    ps    : pixel spacing (mm/px)
    max_r : profile length (pixel)

    Returns
    -------
    dict: pve_width_mm, mtf50, mtf10, profile_s, lsf, freq, mtf
    """

    profile = radial_profile(hu, cx, cy, max_r)
    x       = np.arange(len(profile))

    # ── smooth ESF ──────────────────────────────────────────
    profile_s = gaussian_filter1d(profile, sigma=2)

    # ── PVE width จาก sigmoid fit ───────────────────────────
    pve_width = float("nan")

    try:
        # initial guess
        a_g  = profile_s[-1] - profile_s[0]
        x0_g = len(profile_s) // 2
        k_g  = 2.0
        b_g  = profile_s[0]

        # bounds: k > 0 เสมอ → ป้องกัน PVE ติดลบ
        lower = [-np.inf,  0,    0.1, -np.inf]
        upper = [ np.inf,  max_r, 20,  np.inf]

        popt, _ = curve_fit(
            sigmoid, x, profile_s,
            p0=[a_g, x0_g, k_g, b_g],
            bounds=(lower, upper),
            maxfev=8000
        )

        a, x0, k, b = popt

        # 10–90% transition width
        r10 = x0 - 2.197 * k
        r90 = x0 + 2.197 * k
        pve_width = (r90 - r10) * ps   # mm, ต้อง > 0 เสมอ

    except (RuntimeError, ValueError):
        pve_width = float("nan")

    # ── LSF = gradient of smoothed ESF ──────────────────────
    # smooth อีกรอบหลัง gradient เพื่อลด noise
    lsf = np.gradient(profile_s)
    lsf = gaussian_filter1d(lsf, sigma=1)

    # ── MTF = |FFT(LSF)| normalized ─────────────────────────
    # zero-pad เพื่อเพิ่ม frequency resolution
    n_pad  = len(lsf) * 4
    lsf_w  = lsf * np.hanning(len(lsf))   # Hanning window ลด spectral leakage
    mtf    = np.abs(np.fft.fft(lsf_w, n=n_pad))
    mtf   /= mtf[0] if mtf[0] > 0 else 1  # normalize ที่ f=0

    freq   = np.fft.fftfreq(n_pad, d=ps)
    half   = n_pad // 2
    freq_p = freq[:half]
    mtf_p  = mtf[:half]

    # MTF50 / MTF10 — หาจุดแรกที่ MTF ลดลงถึง threshold
    def _find_freq(mtf_arr, freq_arr, threshold):
        # หา region ที่ MTF ลดลงผ่าน threshold (monotonic drop)
        # ข้าม spike ช่วงแรก (f < 0.05 lp/mm) ที่เกิดจาก DC
        start_idx = np.searchsorted(freq_arr, 0.05)
        sub_mtf   = mtf_arr[start_idx:]
        sub_freq  = freq_arr[start_idx:]

        indices = np.where(sub_mtf <= threshold)[0]
        if len(indices) == 0:
            return float("nan")

        idx = indices[0]
        if idx == 0:
            return float(sub_freq[0])

        m1, m0 = sub_mtf[idx - 1], sub_mtf[idx]
        f1, f0 = sub_freq[idx - 1], sub_freq[idx]

        if m0 == m1:
            return float(f0)

        t = (threshold - m1) / (m0 - m1)
        return float(f1 + t * (f0 - f1))

    mtf50 = _find_freq(mtf_p, freq_p, 0.5)
    mtf10 = _find_freq(mtf_p, freq_p, 0.1)

    return {
        "pve_width_mm": pve_width,
        "mtf50":        mtf50,
        "mtf10":        mtf10,
        "profile_s":    profile_s,   # ESF smoothed
        "lsf":          lsf,
        "freq":         freq_p,
        "mtf":          mtf_p,
    }


# ══════════════════════════════════════════════════════════════════════════
#  BASIC QC PAGE — Geometry / Square / CT Number Linearity
# ══════════════════════════════════════════════════════════════════════════
if _page == "basicqc":
    # Slice selector
    _cur_basic = st.session_state.current
    _sl_basic  = st.session_state.slices[_cur_basic]
    hu_basic   = _sl_basic.get("hu_mod") if _sl_basic.get("hu_mod") is not None else _sl_basic["hu_orig"]
    px_basic   = _sl_basic["pixel_spacing"]

    st.markdown('<div class="secLabel">เลือก Slice สำหรับ Basic QC</div>', unsafe_allow_html=True)
    _n_sl = len(st.session_state.slices)
    _basic_slice_sel = st.slider(
        "Basic QC Slice",
        0, _n_sl - 1,
        _cur_basic, 1,
        key="basic_qc_slice",
        label_visibility="collapsed"
    )
    if _basic_slice_sel != _cur_basic:
        st.session_state.current = _basic_slice_sel
        st.rerun()

    _sl_basic = st.session_state.slices[_basic_slice_sel]
    hu_basic  = _sl_basic.get("hu_mod") if _sl_basic.get("hu_mod") is not None else _sl_basic["hu_orig"]
    px_basic  = _sl_basic["pixel_spacing"]

    # Preview thumbnail
    _thumb = wl_ww(hu_basic, st.session_state.wl, st.session_state.ww)
    dark_style()
    _fig_th, _ax_th = plt.subplots(figsize=(3, 3), facecolor=DARK)
    _ax_th.imshow(_thumb, cmap="gray", vmin=0, vmax=255)
    _ax_th.axis("off")
    _ax_th.set_title(
        f"{_sl_basic['name']}  |  ps={px_basic:.3f}mm",
        fontsize=7, color=MUTED, pad=2
    )
    plt.tight_layout(pad=0.1)
    st.image(fig2img(_fig_th), use_container_width=True)
    plt.close(_fig_th)

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # 3 คอลัมน์: Geometry | Square 50mm | CT Linearity
    # ══════════════════════════════════════════════════
    col_geo, col_sq, col_lin = st.columns(3)

    # ─────────────────────────
    # COL 1 · Geometry QC
    # ─────────────────────────
    with col_geo:
        st.markdown("""
        <div class="qc-card">
          <div class="qc-title">① Geometry QC</div>
          <div class="qc-desc">INNER DISK DIAMETER · NOMINAL 150 mm · TOL ±2%</div>
        </div>""", unsafe_allow_html=True)

        if st.button("▶ Run Geometry QC", use_container_width=True, key="btn_geo"):
            with st.spinner("Detecting phantom..."):
                _geo_res, _geo_err = run_geometry_qc(hu_basic, px_basic)
            if _geo_err:
                st.markdown(f'<div class="warnbox">{_geo_err}</div>', unsafe_allow_html=True)
            else:
                st.session_state.basic_geometry_result = _geo_res
                slog("Basic QC: Geometry done", "ok")

        if st.session_state.get("basic_geometry_result"):
            _gr = st.session_state.basic_geometry_result
            # image
            dark_style()
            _fig_g, _ax_g = plt.subplots(figsize=(4, 4), facecolor=DARK)
            _ax_g.imshow(_gr["image"][..., ::-1])
            _ax_g.set_title("Inner Disk Geometry", fontsize=9, color=MUTED)
            _ax_g.axis("off")
            plt.tight_layout(pad=0.2)
            st.image(fig2img(_fig_g), use_container_width=True)
            plt.close(_fig_g)

            # metrics
            _diff_abs = abs(_gr["diff"])
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-item">
                <div class="metric-val">{_gr['diameter']:.2f}</div>
                <div class="metric-lbl">Measured (mm)</div>
              </div>
              <div class="metric-item">
                <div class="metric-val">{_diff_abs:.2f}</div>
                <div class="metric-lbl">Diff (mm)</div>
              </div>
            </div>""", unsafe_allow_html=True)

            _badge = '<span class="pass-badge">✓ PASS</span>' if _gr["pass"] else '<span class="fail-badge">✗ FAIL</span>'
            st.markdown(f"Nominal: **{_gr['nominal']} ± {_gr['tol']:.1f} mm** &nbsp; {_badge}", unsafe_allow_html=True)
            st.dataframe(_gr["df"], use_container_width=True, hide_index=True)

    # ─────────────────────────
    # COL 2 · Square 50mm QC
    # ─────────────────────────
    with col_sq:
        st.markdown("""
        <div class="qc-card">
          <div class="qc-title">② Square 50 mm QC</div>
          <div class="qc-desc">4-CORNER HOLE DISTANCE · NOMINAL 50 mm · TOL ±2%</div>
        </div>""", unsafe_allow_html=True)

        if st.button("▶ Run Square QC", use_container_width=True, key="btn_sq"):
            with st.spinner("Detecting square holes..."):
                _sq_res, _sq_err = run_square_qc(hu_basic, px_basic)
            if _sq_err:
                st.markdown(f'<div class="warnbox">{_sq_err}</div>', unsafe_allow_html=True)
            else:
                st.session_state.basic_square_result = _sq_res
                slog("Basic QC: Square done", "ok")

        if st.session_state.get("basic_square_result"):
            _sqr = st.session_state.basic_square_result
            dark_style()
            _fig_s, _ax_s = plt.subplots(figsize=(6, 6), facecolor=DARK)
            _ax_s.imshow(_sqr["image"][..., ::-1])
            _ax_s.set_title("Square 50mm QC", fontsize=9, color=MUTED)
            _ax_s.axis("off")
            plt.tight_layout(pad=0.2)
            st.image(fig2img(_fig_s), use_container_width=True)
            plt.close(_fig_s)

            if _sqr.get("warn"):
                st.markdown(f'<div class="warnbox">{_sqr["warn"]}</div>', unsafe_allow_html=True)
            _badge_sq = '<span class="pass-badge">✓ OVERALL PASS</span>' if _sqr["pass"] else '<span class="fail-badge">✗ OVERALL FAIL</span>'
            st.markdown(_badge_sq, unsafe_allow_html=True)
            st.dataframe(_sqr["df"], use_container_width=True, hide_index=True)

    # ─────────────────────────
    # COL 3 · CT Number Linearity
    # ─────────────────────────
    with col_lin:
        st.markdown("""
        <div class="qc-card">
          <div class="qc-title">③ CT Number Linearity</div>
          <div class="qc-desc">8 MATERIAL INSERTS · R² ≥ 0.99 · HU vs µ / DENSITY</div>
        </div>""", unsafe_allow_html=True)

        if st.button("▶ Run CT Linearity", use_container_width=True, key="btn_lin"):
            with st.spinner("Detecting inserts & computing linearity..."):
                _lin_res, _lin_err = run_linearity_qc(hu_basic, px_basic)
            if _lin_err:
                st.markdown(f'<div class="warnbox">{_lin_err}</div>', unsafe_allow_html=True)
            else:
                st.session_state.basic_linearity_result = _lin_res
                slog("Basic QC: Linearity done", "ok")

        if st.session_state.get("basic_linearity_result"):
            _lr = st.session_state.basic_linearity_result
            # annotation image
            dark_style()
            _fig_l, _ax_l = plt.subplots(figsize=(4, 4), facecolor=DARK)
            _ax_l.imshow(_lr["image"][..., ::-1])
            _ax_l.set_title("CT Number Linearity ROIs", fontsize=9, color=MUTED)
            _ax_l.axis("off")
            plt.tight_layout(pad=0.2)
            st.image(fig2img(_fig_l), use_container_width=True)
            plt.close(_fig_l)

            # R² badges
            _r2 = _lr.get("R2")
            _r2d = _lr.get("R2_density")
            if _r2 is not None:
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-item">
                    <div class="metric-val">{"✓" if _r2>=0.99 else "✗"} {_r2:.4f}</div>
                    <div class="metric-lbl">R² vs µ</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-val">{"✓" if _r2d>=0.99 else "✗"} {_r2d:.4f}</div>
                    <div class="metric-lbl">R² vs Density</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            _lin_badge = '<span class="pass-badge">✓ PASS (R²≥0.99)</span>' if _lr["pass"] else '<span class="fail-badge">✗ FAIL (R²<0.99)</span>'
            st.markdown(_lin_badge, unsafe_allow_html=True)
            st.dataframe(_lr["df"], use_container_width=True, hide_index=True)

            # Linearity plots
            if _lr.get("fig1"):
                st.pyplot(_lr["fig1"])
                plt.close(_lr["fig1"])
            if _lr.get("fig2"):
                st.pyplot(_lr["fig2"])
                plt.close(_lr["fig2"])

    # ── Second row: CT Accuracy/Uniformity + Noise ──
    st.markdown("---")
    col_acc, col_noise = st.columns(2)

    # ─── CT Number Accuracy + Uniformity ───
    with col_acc:
        st.markdown("""
        <div class="qc-card">
          <div class="qc-title">④ CT Accuracy &amp; Uniformity</div>
          <div class="qc-desc">
            CENTER ROI ±20 HU (water) &nbsp;·&nbsp; MAX PERIPHERAL DIFF ≤5 HU
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("▶ Run Accuracy + Uniformity", use_container_width=True, key="btn_acc"):
            with st.spinner("Computing ROIs…"):
                _acc_res, _acc_err = run_ct_accuracy_qc(hu_basic, px_basic)
            if _acc_err:
                st.markdown(f'<div class="warnbox">{_acc_err}</div>', unsafe_allow_html=True)
            else:
                st.session_state.basic_accuracy_result = _acc_res
                slog("Basic QC: Accuracy+Uniformity done", "ok")

        if st.session_state.get("basic_accuracy_result"):
            _ar = st.session_state.basic_accuracy_result

            # Image
            dark_style()
            _fig_a, _ax_a = plt.subplots(figsize=(4, 4), facecolor=DARK)
            _ax_a.imshow(_ar["image"][..., ::-1])
            _ax_a.set_title("CT Accuracy & Uniformity ROIs", fontsize=9, color=MUTED)
            _ax_a.axis("off")
            plt.tight_layout(pad=0.2)
            st.image(fig2img(_fig_a), use_container_width=True)
            plt.close(_fig_a)

            # CT Accuracy result
            _c_hu  = _ar["center_hu"]
            _c_col = "#70ffc0" if _ar["acc_pass"] else "#ff8a80"
            _a_badge = '<span class="pass-badge">✓ PASS</span>' if _ar["acc_pass"] else '<span class="fail-badge">✗ FAIL</span>'
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-item">
                <div class="metric-val" style="color:{_c_col};">{_c_hu:.2f}</div>
                <div class="metric-lbl">Center HU (water)</div>
              </div>
              <div class="metric-item">
                <div class="metric-val">{_ar['tol_acc']:.0f}</div>
                <div class="metric-lbl">Tolerance (HU)</div>
              </div>
            </div>
            <div style="margin:4px 0 8px;">CT Accuracy {_a_badge}</div>""",
            unsafe_allow_html=True)

            # Uniformity result
            _md   = _ar["max_diff"]
            _u_badge = '<span class="pass-badge">✓ PASS</span>' if _ar["uni_pass"] else '<span class="fail-badge">✗ FAIL</span>'
            _u_col   = "#70ffc0" if _ar["uni_pass"] else "#ff8a80"
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-item">
                <div class="metric-val" style="color:{_u_col};">{_md:.2f}</div>
                <div class="metric-lbl">Max Diff (HU)</div>
              </div>
              <div class="metric-item">
                <div class="metric-val">5.00</div>
                <div class="metric-lbl">Tolerance (HU)</div>
              </div>
            </div>
            <div style="margin:4px 0 8px;">Uniformity {_u_badge}</div>""",
            unsafe_allow_html=True)

            st.dataframe(_ar["df"], use_container_width=True, hide_index=True)

    # ─── Noise QC ───
    with col_noise:
        st.markdown("""
        <div class="qc-card">
          <div class="qc-title">⑤ Noise QC (SD)</div>
          <div class="qc-desc">
            SD ของ ROI กลาง 40% radius &nbsp;·&nbsp; เทียบกับ baseline ±25%
          </div>
        </div>""", unsafe_allow_html=True)

        # Baseline input
        _cur_baseline = st.session_state.get("basic_noise_baseline") or 0.0
        _new_baseline = st.number_input(
            "Baseline Noise (HU SD)",
            min_value=0.0, max_value=500.0,
            value=float(_cur_baseline), step=0.5,
            key="noise_baseline_input",
            help="กรอก baseline noise จากวันก่อน (0 = ยังไม่มี baseline)")
        st.session_state.basic_noise_baseline = _new_baseline

        if st.button("▶ Run Noise QC", use_container_width=True, key="btn_noise"):
            _bl = _new_baseline if _new_baseline > 0 else None
            with st.spinner("Computing noise…"):
                _nz_res, _nz_err = run_noise_qc(hu_basic, px_basic, baseline_noise=_bl)
            if _nz_err:
                st.markdown(f'<div class="warnbox">{_nz_err}</div>', unsafe_allow_html=True)
            else:
                # Auto-update baseline if pass
                if _nz_res.get("pass") is True:
                    st.session_state.basic_noise_baseline = _nz_res["measured"]
                st.session_state.basic_noise_result = _nz_res
                slog("Basic QC: Noise done", "ok")

        if st.session_state.get("basic_noise_result"):
            _nr = st.session_state.basic_noise_result

            # Image
            dark_style()
            _fig_n, _ax_n = plt.subplots(figsize=(4, 4), facecolor=DARK)
            _ax_n.imshow(_nr["image"][..., ::-1])
            _ax_n.set_title("Noise ROI (40% radius)", fontsize=9, color=MUTED)
            _ax_n.axis("off")
            plt.tight_layout(pad=0.2)
            st.image(fig2img(_fig_n), use_container_width=True)
            plt.close(_fig_n)

            _msd = _nr["measured"]
            _bl2 = _nr.get("baseline")

            if _bl2 is None:
                # ยังไม่มี baseline — แค่แสดงค่า
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-item">
                    <div class="metric-val">{_msd:.2f}</div>
                    <div class="metric-lbl">Measured SD (HU)</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-val">{_nr["mean"]:.2f}</div>
                    <div class="metric-lbl">Mean HU</div>
                  </div>
                </div>
                <div class="infobox">ℹ กรอก Baseline Noise แล้วกด Run อีกครั้งเพื่อดู PASS/FAIL</div>""",
                unsafe_allow_html=True)
            else:
                _n_pass  = _nr["pass"]
                _n_badge = '<span class="pass-badge">✓ PASS</span>' if _n_pass else '<span class="fail-badge">✗ FAIL</span>'
                _n_col   = "#70ffc0" if _n_pass else "#ff8a80"
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-item">
                    <div class="metric-val" style="color:{_n_col};">{_msd:.2f}</div>
                    <div class="metric-lbl">Measured SD (HU)</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-val">{_bl2:.2f}</div>
                    <div class="metric-lbl">Baseline SD (HU)</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                _ad = _nr["abs_diff"]; _rd = _nr["rel_diff"]
                _mn = _nr["min_accept"]; _mx = _nr["max_accept"]
                st.markdown(f"""
                <div style="background:#0a0f18;border:1px solid #1a2a3a;border-radius:8px;
                  padding:10px 14px;font-size:11px;color:#7ac8e0;line-height:1.9;margin:8px 0;">
                  Absolute deviation: <b>{_ad:+.2f} HU</b><br>
                  Relative deviation: <b>{_rd:+.1f}%</b><br>
                  Acceptance range: <b>[{_mn:.2f} , {_mx:.2f}] HU</b> &nbsp;(±25%)
                </div>
                <div style="margin-top:4px;">Noise QC {_n_badge}</div>""",
                unsafe_allow_html=True)

    # ── Summary Banner ──
    st.markdown("---")
    _geo_ok  = (st.session_state.get("basic_geometry_result") or {}).get("pass", False)
    _sq_ok   = (st.session_state.get("basic_square_result")  or {}).get("pass", False)
    _lin_ok  = (st.session_state.get("basic_linearity_result") or {}).get("pass", False)
    _acc_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("acc_pass", False)
    _uni_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("uni_pass", False)
    _nz_ok   = (st.session_state.get("basic_noise_result") or {}).get("pass", False) or \
               (st.session_state.get("basic_noise_result") or {}).get("pass") is None

    _has_geo = st.session_state.get("basic_geometry_result") is not None
    _has_sq  = st.session_state.get("basic_square_result")  is not None
    _has_lin = st.session_state.get("basic_linearity_result") is not None
    _has_acc = st.session_state.get("basic_accuracy_result") is not None
    _has_nz  = st.session_state.get("basic_noise_result")   is not None

    # ── Third row: Slice Thickness ──
    st.markdown("---")
    st.markdown("""
    <div class="qc-card">
      <div class="qc-title">⑥ Slice Thickness Accuracy (Ramp Method)</div>
      <div class="qc-desc">
        FINE RAMP 0.25 mm/bead · COARSE RAMP 1 mm/bead · นับ bead peaks ในช่วง FWHM
      </div>
    </div>""", unsafe_allow_html=True)

    _st_nom = float(_sl_basic.get("slice_thickness", 1.0))
    hu_current = _sl_basic["hu_orig"]
    px_current = _sl_basic["pixel_spacing"]
    
    _sc1, _sc2,= st.columns([1, 1])

    with _sc1:
        _st_nom_input = st.number_input(
            "Nominal Slice Thickness (mm)",
            min_value=0.1, max_value=20.0,
            value=_st_nom, step=0.1,
            key="st_nominal_input",
            help="ดึงจาก DICOM SliceThickness อัตโนมัติ แก้ไขได้")
    with _sc2:
        _st_mode = st.radio("Ramp Mode", ["Fine (0.25 mm/bead)", "Coarse (1 mm/bead)"],
                            horizontal=True, key="st_mode_radio")
        _st_mode_key = "fine" if "Fine" in _st_mode else "coarse"
        

    st.markdown("### 🔧 Adjust Profile Line")

    if _st_mode_key == "fine":
            default_offset_x = -53
    else:
            default_offset_x = -95

    if "slice_offset_x" not in st.session_state:
            st.session_state.slice_offset_x = default_offset_x

    if st.session_state.get("last_mode") != _st_mode_key:
            st.session_state.slice_offset_x = default_offset_x
            st.session_state.last_mode = _st_mode_key

    _lc1, _lc2, _lc3 = st.columns(3)

    with _lc1:
            offset_x = st.slider("Shift X (px)", -100, 100, -53, 1, key="slice_offset_x")

    with _lc2:
            offset_y = st.slider("Shift Y (px)", -100, 100, 0, 1, key="slice_offset_y")

    with _lc3:
            angle = st.slider("Angle (deg)", -90, 90, 90, 1, key="slice_angle")
       
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        
    # ================= RUN =================
    _sterr = None
    _stres = None

    if st.button("▶ Run Slice Thickness", use_container_width=True, key="btn_slice"):
        with st.spinner("Computing profile…"):
            _stres, _sterr = run_slice_thickness_qc(
                hu_current,
                px_current,
                _st_nom_input,
                mode=_st_mode_key,
                offset_x=offset_x,
                offset_y=offset_y,
            )

    # ================= RESULT =================
    if _sterr:
        st.markdown(f'<div class="warnbox">{_sterr}</div>', unsafe_allow_html=True)

    elif _stres is not None:
        if _st_mode_key == "fine":
            st.session_state.basic_slice_fine = _stres
        else:
            st.session_state.basic_slice_coarse = _stres

        slog(f"Slice Thickness ({_st_mode_key}): {_stres['slice_mm']:.2f} mm", "ok")
    # =========================
    # Fine Result
    # =========================
    if st.session_state.get("basic_slice_fine"):
        st.markdown("### 🔵 Fine Ramp Result")
        _sr = st.session_state.basic_slice_fine

        if "fine_zoom" not in st.session_state:
            st.session_state["fine_zoom"] = 1.0

        def _reset_fine():
            st.session_state["fine_zoom"] = 1.0

        _fc1, _fc2 = st.columns([3, 1])
        with _fc1:
            _fine_zoom = st.number_input(
                "🔍 Zoom", min_value=1.0, max_value=8.0,
                step=0.5, key="fine_zoom")
        with _fc2:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            st.button("🔄 Reset Zoom", key="fine_reset",
                      on_click=_reset_fine,
                      use_container_width=True)

        _img = _sr["img_arr"]
        _ih, _iw = _img.shape[:2]

        _gray  = np.mean(_img, axis=(0, 2))
        _split = int(np.argmin(_gray[_iw//4 : 3*_iw//4]) + _iw//4)

        _img_left  = _img[:, :_split]
        _img_right = _img[:, _split:]

        _ih_l, _iw_l = _img_left.shape[:2]
        _cx_l = _iw_l / 2.0
        _cy_l = _ih_l / 2.0
        _hw_l = _cx_l / _fine_zoom
        _hh_l = _cy_l / _fine_zoom

        dark_style()
        _fig_f, (_ax_l, _ax_r) = plt.subplots(
            1, 2, figsize=(8, 4), facecolor=DARK,
            gridspec_kw={"width_ratios": [_split, _iw - _split]}
        )
        _ax_l.imshow(_img_left)
        _ax_l.set_xlim(_cx_l - _hw_l, _cx_l + _hw_l)
        _ax_l.set_ylim(_cy_l + _hh_l, _cy_l - _hh_l)
        _ax_l.axis("off")
        _ax_l.set_title("Profile Line", fontsize=8, color=MUTED)

        _ax_r.imshow(_img_right)
        _ax_r.axis("off")
        _ax_r.set_title("HU Profile", fontsize=8, color=MUTED)

        plt.tight_layout(pad=0.1)
        st.image(fig2img(_fig_f), use_container_width=True)
        plt.close(_fig_f)

        _sm   = _sr["slice_mm"]
        _sp   = _sr["pass"]
        _scol = "#70ffc0" if _sp else "#ff8a80"
        _sbadge = '<span class="pass-badge">✓ PASS</span>' if _sp else '<span class="fail-badge">✗ FAIL</span>'

        st.markdown(f"""
        <div class="metric-row">
        <div class="metric-item">
            <div class="metric-val" style="color:{_scol};">
            {"N/A" if np.isnan(_sm) else f"{_sm:.2f}"}
            </div>
            <div class="metric-lbl">Measured (mm)</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["nominal"]:.2f}</div>
            <div class="metric-lbl">Nominal (mm)</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["num_beads"]}</div>
            <div class="metric-lbl">Beads</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["slice_mm"]:.2f}</div>
            <div class="metric-lbl">Slice Thickness</div>
        </div>
        </div>
        <div>Fine {_sbadge}</div>
        """, unsafe_allow_html=True)

    # =========================
    # Coarse Result
    # =========================
    if st.session_state.get("basic_slice_coarse"):
        st.markdown("### 🟠 Coarse Ramp Result")
        _sr = st.session_state.basic_slice_coarse

        KEY = "coarse"

        # -------------------------
        # state init
        # -------------------------
        if f"{KEY}_zoom_value" not in st.session_state:
            st.session_state[f"{KEY}_zoom_value"] = 1.0

        if f"{KEY}_zoom_version" not in st.session_state:
            st.session_state[f"{KEY}_zoom_version"] = 0

        # current widget key (เปลี่ยนเมื่อ reset)
        zoom_widget_key = f"{KEY}_zoom_widget_{st.session_state[f'{KEY}_zoom_version']}"

        # -------------------------
        # reset callback
        # -------------------------
        def reset_coarse_zoom():
            st.session_state[f"{KEY}_zoom_value"] = 1.0
            st.session_state[f"{KEY}_zoom_version"] += 1

        # -------------------------
        # widget
        # -------------------------
        _coarse_zoom = st.number_input(
            "🔍 Zoom",
            min_value=1.0,
            max_value=8.0,
            value=float(st.session_state[f"{KEY}_zoom_value"]),
            step=0.5,
            key=zoom_widget_key,
        )

        # เก็บค่าปัจจุบันไว้ใช้งาน
        st.session_state[f"{KEY}_zoom_value"] = _coarse_zoom
        zoom_val = _coarse_zoom

        # -------------------------
        # reset button
        # -------------------------
        st.button(
            "🔄 Reset Zoom",
            key=f"{KEY}_reset_btn",
            on_click=reset_coarse_zoom,
        )

        # ================= IMAGE =================
        _img = _sr["img_arr"]
        _ih, _iw = _img.shape[:2]

        _gray  = np.mean(_img, axis=(0, 2))
        _split = int(np.argmin(_gray[_iw//4 : 3*_iw//4]) + _iw//4)

        _img_left  = _img[:, :_split]
        _img_right = _img[:, _split:]

        _ih_l, _iw_l = _img_left.shape[:2]
        _cx_l = _iw_l / 2.0
        _cy_l = _ih_l / 2.0
        _hw_l = _cx_l / zoom_val
        _hh_l = _cy_l / zoom_val

        dark_style()
        _fig_c, (_ax_l, _ax_r) = plt.subplots(
            1, 2, figsize=(8, 4), facecolor=DARK,
            gridspec_kw={"width_ratios": [_split, _iw - _split]}
        )

        _ax_l.imshow(_img_left)
        _ax_l.set_xlim(_cx_l - _hw_l, _cx_l + _hw_l)
        _ax_l.set_ylim(_cy_l + _hh_l, _cy_l - _hh_l)
        _ax_l.axis("off")
        _ax_l.set_title("Profile Line", fontsize=8, color=MUTED)

        _ax_r.imshow(_img_right)
        _ax_r.axis("off")
        _ax_r.set_title("HU Profile", fontsize=8, color=MUTED)

        plt.tight_layout(pad=0.1)
        st.image(fig2img(_fig_c), use_container_width=True)
        plt.close(_fig_c)

        # ================= METRIC =================
        _sm   = _sr["slice_mm"]
        _sp   = _sr["pass"]
        _scol = "#70ffc0" if _sp else "#ff8a80"
        _sbadge = '<span class="pass-badge">✓ PASS</span>' if _sp else '<span class="fail-badge">✗ FAIL</span>'

        st.markdown(f"""
        <div class="metric-row">
        <div class="metric-item">
            <div class="metric-val" style="color:{_scol};">
            {"N/A" if np.isnan(_sm) else f"{_sm:.2f}"}
            </div>
            <div class="metric-lbl">Measured (mm)</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["nominal"]:.2f}</div>
            <div class="metric-lbl">Nominal (mm)</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["num_beads"]}</div>
            <div class="metric-lbl">Beads</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{_sr["fwhm_mm"]:.2f}</div>
            <div class="metric-lbl">FWHM</div>
        </div>
        </div>
        <div>Coarse {_sbadge}</div>
        """, unsafe_allow_html=True)

    # ── Summary Banner ──
    st.markdown("---")
    _geo_ok  = (st.session_state.get("basic_geometry_result") or {}).get("pass", False)
    _sq_ok   = (st.session_state.get("basic_square_result")  or {}).get("pass", False)
    _lin_ok  = (st.session_state.get("basic_linearity_result") or {}).get("pass", False)
    _acc_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("acc_pass", False)
    _uni_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("uni_pass", False)
    _fine_ok   = (st.session_state.get("basic_slice_fine")   or {}).get("pass", False)
    _coarse_ok = (st.session_state.get("basic_slice_coarse") or {}).get("pass", False)
    _lc_ok   = st.session_state.get("lc_detect_v2", False)
    _has_lc  = st.session_state.get("lc_detect_v2", False) or \
                st.session_state.get("lc_not_detect_v2", False)
    _has_fine   = st.session_state.get("basic_slice_fine")   is not None
    _has_coarse = st.session_state.get("basic_slice_coarse") is not None

    _st_ok = _fine_ok and _coarse_ok
    _has_st = _has_fine or _has_coarse

    _has_geo = st.session_state.get("basic_geometry_result") is not None
    _has_sq  = st.session_state.get("basic_square_result")  is not None
    _has_lin = st.session_state.get("basic_linearity_result") is not None
    _has_acc = st.session_state.get("basic_accuracy_result") is not None
    _has_nz  = st.session_state.get("basic_noise_result")   is not None
    

    def _icon(has, ok):
        if not has: return "⚪"
        return "🟢" if ok else "🔴"

    _any_done = any([_has_geo, _has_sq, _has_lin, _has_acc, _has_nz, _has_st, _has_lc])
    if _any_done:
        _nz_pass_flag = (st.session_state.get("basic_noise_result") or {}).get("pass")
        _nz_disp_ok   = _nz_pass_flag is True or _nz_pass_flag is None

        st.markdown(f"""
        <div class="qc-card" style="margin-top:4px;">
          <div class="qc-title" style="font-size:13px;">📋 Basic QC Summary</div>
          <div style="display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;font-size:12px;color:#d8e8ff;">
            <span>{_icon(_has_geo,_geo_ok)} Geometry QC</span>
            <span>{_icon(_has_sq,_sq_ok)} Square 50mm</span>
            <span>{_icon(_has_lin,_lin_ok)} CT Linearity</span>
            <span>{_icon(_has_acc,_acc_ok)} CT Accuracy</span>
            <span>{_icon(_has_acc,_uni_ok)} Uniformity</span>
            <span>{_icon(_has_nz,_nz_disp_ok)} Noise QC</span>
            <span>{_icon(_has_st,_st_ok)} Slice Thickness</span>
            <span>{_icon(_has_lc,_lc_ok)} Low Contrast</span>
          </div>
        </div>""", unsafe_allow_html=True)

        _all_done = all([_has_geo, _has_sq, _has_lin, _has_acc, _has_nz, _has_st, _has_lc])
        if _all_done:
            _nz_real_pass = (st.session_state.get("basic_noise_result") or {}).get("pass")
            _all_ok = all([_geo_ok, _sq_ok, _lin_ok, _acc_ok, _uni_ok, _st_ok, _lc_ok,
                           (_nz_real_pass is True or _nz_real_pass is None)])
            if _all_ok:
                st.markdown('<div class="okbox" style="font-size:13px;font-weight:700;">🟢 ALL BASIC QC TESTS PASSED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warnbox" style="font-size:13px;font-weight:700;">🔴 SOME TESTS FAILED — ตรวจสอบผลลัพธ์ด้านบน</div>', unsafe_allow_html=True)
   
def reset_spatial_state():
    for _k in ["spatial_slices", "spatial_img", "spatial_px",
               "spatial_profiles", "spatial_groups", "spatial_line_coords",
               "spatial_last_upload"]:
        st.session_state[_k] = None
    # clear ผลการวิเคราะห์เก่าด้วย
    for _k in ["sr_detect_v2", "sr_not_detect_v2", "sr_pass", "sr_has"]:
        if _k in st.session_state:
            del st.session_state[_k] 

# ═══════════════════════════════════════════════════════════════════════════
# PASTE THIS BLOCK to replace the entire ⑦ Low Contrast Resolution section
# Replace from the comment below through to (NOT including):
#   st.markdown("### 📂 Spatial Resolution Input (DICOM Series)")
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────
# ⑦ Low Contrast Resolution  (uses already-loaded slices — no separate upload)
# ─────────────────────────

st.markdown("---")
st.markdown("""
<div class="qc-card">
  <div class="tool-card-accent accent-green" style="border-radius:10px 10px 0 0;height:3px;"></div>
  <div class="qc-title" style="margin-top:8px;">⑦ Low Contrast Resolution</div>
  <div class="qc-desc">
    IAEA STANDARD · DETECT ≤5 mm OBJECT @ 0.5% CONTRAST · VISUAL ASSESSMENT
  </div>
</div>
""", unsafe_allow_html=True)

# ── Use the SAME slice already selected in Basic QC (no separate upload needed) ──
_lc_sl   = st.session_state.slices[_basic_slice_sel]
_lc_hu   = _lc_sl.get("hu_mod") if _lc_sl.get("hu_mod") is not None else _lc_sl["hu_orig"]
_lc_px   = _lc_sl["pixel_spacing"]
_lc_h, _lc_w = _lc_hu.shape

st.markdown(
    f'<div class="infobox" style="font-size:11px;margin-bottom:10px;">'
    f'📋 Slice <b>{_basic_slice_sel + 1}</b> — <b>{_lc_sl["name"]}</b>'
    f' &nbsp;|&nbsp; ps={_lc_px:.3f} mm'
    f' &nbsp;|&nbsp; ใช้ slice slider ด้านบนเพื่อเปลี่ยน slice</div>',
    unsafe_allow_html=True
)

# ── Controls row ──
_lc_c1, _lc_c2, _lc_c3 = st.columns([1, 1, 1])
with _lc_c1:
    _lc_wl = st.number_input("WL", min_value=-5000, max_value=5000, 
                              value=100, step=10, key="lc_wl_v2",
                              help="Window Level — แนะนำ 0–60 HU สำหรับ low contrast")
with _lc_c2:
    _lc_ww = st.number_input("WW", min_value=-5000, max_value=5000, 
                              value=100, step=100, key="lc_ww_v2",
                              help="Window Width — แคบ (100–200 HU) เพื่อเพิ่ม contrast")
with _lc_c3:
    _lc_zoom = st.number_input("🔍 Zoom", min_value=1.0, max_value=8.0, 
                                value=1.0, step=0.5, key="lc_zoom_v2")

# ── Windowed + zoomed image ──
_lc_range = float(_lc_ww)

if _lc_range <= 0:
    # WW = 0 → binary threshold ที่ WL (ตาม DICOM Standard)
    _lc_disp = (_lc_hu >= _lc_wl).astype(np.float32)
else:
    _lc_lo   = _lc_wl - _lc_range / 2.0
    _lc_hi   = _lc_wl + _lc_range / 2.0
    _lc_disp = np.clip(_lc_hu, _lc_lo, _lc_hi)
    _lc_disp = (_lc_disp - _lc_lo) / _lc_range

dark_style()
_fig_lc, _ax_lc = plt.subplots(figsize=(5, 5), facecolor=DARK)
_ax_lc.imshow(_lc_disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

_lc_cx, _lc_cy = _lc_w / 2.0, _lc_h / 2.0
_lc_hw = (_lc_w / 2.0) / _lc_zoom
_lc_hh = (_lc_h / 2.0) / _lc_zoom
_ax_lc.set_xlim(_lc_cx - _lc_hw, _lc_cx + _lc_hw)
_ax_lc.set_ylim(_lc_cy + _lc_hh, _lc_cy - _lc_hh)
_ax_lc.set_title(
    f"Low Contrast  ·  WL {_lc_wl} / WW {_lc_ww}  ·  Zoom {_lc_zoom:.1f}×",
    fontsize=8, color=MUTED, pad=4
)
_ax_lc.axis("off")
plt.tight_layout(pad=0.2)
st.image(fig2img(_fig_lc), use_container_width=True)
plt.close(_fig_lc)

# ── Visual assessment checkboxes ──
st.markdown(
    '<div style="font-size:10px;letter-spacing:2px;color:#3a5060;'
    'text-transform:uppercase;margin:12px 0 6px;">Visual Assessment</div>',
    unsafe_allow_html=True
)
_lc_choice = st.radio(
    "Low Contrast Resolution Assessment",
    options=[
        "✅  Detected — ≤5 mm object visible at 0.5%  (PASS)",
        "❌  Not detected — object not visible  (FAIL)"
    ],
    horizontal=True,
    index=None,
    key="lc_choice_final"
)

if _lc_choice == "✅  Detected — ≤5 mm object visible at 0.5%  (PASS)":
    st.markdown("""
    <div class="okbox" style="display:flex;align-items:center;gap:12px;
      padding:12px 18px;margin-top:8px;">
      <span style="font-size:26px;line-height:1;">🟢</span>
      <div>
        <div style="font-weight:700;font-size:13px;color:#70ffc0;">
          PASS — Low Contrast Resolution
        </div>
        <div style="font-size:10px;color:#50e090;margin-top:3px;">
          Able to detect a ≤5 mm object at 0.5% target contrast level
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.session_state["lc_detect_v2"]     = True
    st.session_state["lc_not_detect_v2"] = False

elif _lc_choice == "❌  Not detected — object not visible  (FAIL)":
    st.markdown("""
    <div class="warnbox" style="display:flex;align-items:center;gap:12px;
      padding:12px 18px;margin-top:8px;">
      <span style="font-size:26px;line-height:1;">🔴</span>
      <div>
        <div style="font-weight:700;font-size:13px;color:#ff8a80;">
          FAIL — Low Contrast Resolution
        </div>
        <div style="font-size:10px;color:#f5c060;margin-top:3px;">
          Unable to detect a ≤5 mm object at 0.5% target contrast level
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.session_state["lc_detect_v2"]     = False
    st.session_state["lc_not_detect_v2"] = True

else:
    st.markdown("""
    <div style="background:#0a0f18;border:1px dashed #1e2a3a;border-radius:8px;
      padding:12px 18px;font-size:11px;color:#3a5060;
      text-align:center;margin-top:8px;">
      ⚪ &nbsp; Select assessment result to record
    </div>""", unsafe_allow_html=True)
    st.session_state["lc_detect_v2"]     = False
    st.session_state["lc_not_detect_v2"] = False

st.markdown("""
<div class="mathbox" style="margin-top:10px;">
  <b>IAEA — Low Contrast Criterion:</b><br>
  The 0.5% contrast section should resolve objects ≤5 mm in diameter.<br>
  <span style="color:#3a7090;">
    Tip: Use WL ≈ 0–60 HU, WW ≈ 100–150 HU and Zoom for better visibility
  </span>
</div>""", unsafe_allow_html=True)

# ── Basic QC Summary (แสดงหลัง Low Contrast) ──
st.markdown("---")
_geo_ok  = (st.session_state.get("basic_geometry_result") or {}).get("pass", False)
_sq_ok   = (st.session_state.get("basic_square_result")  or {}).get("pass", False)
_lin_ok  = (st.session_state.get("basic_linearity_result") or {}).get("pass", False)
_acc_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("acc_pass", False)
_uni_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("uni_pass", False)
_fine_ok   = (st.session_state.get("basic_slice_fine")   or {}).get("pass", False)
_coarse_ok = (st.session_state.get("basic_slice_coarse") or {}).get("pass", False)
_st_ok   = _fine_ok and _coarse_ok
_lc_ok   = st.session_state.get("lc_detect_v2", False)
_sr_ok   = st.session_state.get("sr_pass", False)

_has_geo    = st.session_state.get("basic_geometry_result")   is not None
_has_sq     = st.session_state.get("basic_square_result")     is not None
_has_lin    = st.session_state.get("basic_linearity_result")  is not None
_has_acc    = st.session_state.get("basic_accuracy_result")   is not None
_has_nz     = st.session_state.get("basic_noise_result")      is not None
_has_st     = st.session_state.get("basic_slice_fine") is not None or \
              st.session_state.get("basic_slice_coarse") is not None
_has_lc     = st.session_state.get("lc_detect_v2", False) or \
              st.session_state.get("lc_not_detect_v2", False)
_has_sr     = st.session_state.get("sr_has", False)

def _icon(has, ok):
    if not has: return "⚪"
    return "🟢" if ok else "🔴"

_nz_pass_flag = (st.session_state.get("basic_noise_result") or {}).get("pass")
_nz_disp_ok   = _nz_pass_flag is True or _nz_pass_flag is None

st.markdown(f"""
<div class="qc-card" style="margin-top:4px;">
  <div class="qc-title" style="font-size:13px;">📋 Basic QC Summary</div>
  <div style="display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;
    font-size:12px;color:#d8e8ff;">
    <span>{_icon(_has_geo,_geo_ok)} Geometry QC</span>
    <span>{_icon(_has_sq,_sq_ok)} Square 50mm</span>
    <span>{_icon(_has_lin,_lin_ok)} CT Linearity</span>
    <span>{_icon(_has_acc,_acc_ok)} CT Accuracy</span>
    <span>{_icon(_has_acc,_uni_ok)} Uniformity</span>
    <span>{_icon(_has_nz,_nz_disp_ok)} Noise QC</span>
    <span>{_icon(_has_st,_st_ok)} Slice Thickness</span>
    <span>{_icon(_has_lc,_lc_ok)} Low Contrast</span>
    <span>{_icon(_has_sr,_sr_ok)} Spatial Resolution</span>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────
# Spatial Resolution (MTF)
# ─────────────────────────
# ── UI header ──
st.markdown("""
    <div class="qc-card">
      <div class="qc-title">⑦ Spatial Resolution (MTF)</div>
      <div class="qc-desc">
            IAEA HUMAN HEALTH SERIES No.19 · CTP528 MODULE · ACCEPTANCE CRITERION &gt; 5 lp/cm
          </div>
    </div>""", unsafe_allow_html=True)

if st.session_state.get("sr_computed"):

        _lp_val  = st.session_state.get("sr_last_lp", 0)
        _mtf_val = st.session_state.get("sr_last_mtf", 0)

        st.markdown(f"""
        <div class="infobox" style="font-size:11px;margin-bottom:10px;">
          Last computed: &nbsp;
          <b>{_lp_val} lp/cm</b> &nbsp;|&nbsp;
          MTF (Michelson) = <b>{_mtf_val:.3f}</b> &nbsp;|&nbsp;
          Criterion: resolve <b>&gt; 5 lp/cm</b> visually
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="mathbox" style="margin-bottom:10px;">
          <b>IAEA Protocol — Step-by-step:</b><br>
          1. Load the Catphan CTP528 slice (high-contrast resolution module)<br>
          2. Adjust WL/WW to clearly visualise the line pair groups<br>
          3. Click <b>Detect Line Pairs</b> to auto-locate LP groups<br>
          4. Select the highest lp/cm group still visually resolvable<br>
          5. Click <b>Compute MTF</b> to calculate Michelson contrast<br>
          6. Mark <b>Resolved</b> if the line pairs are clearly separated<br>
          7. <b>Pass criterion:</b> able to resolve &gt; 5 lp/cm (IAEA HHS No.19 §4.3)
        </div>""", unsafe_allow_html=True)

st.markdown("### 📂 Spatial Resolution Input (DICOM Series)")

# ── init session keys ──
for _k in ["spatial_slices", "spatial_img", "spatial_px",
           "spatial_profiles", "spatial_groups", "spatial_line_coords",
           "spatial_last_upload"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

# key counter สำหรับ reset uploader
if "spatial_uploader_key" not in st.session_state:
    st.session_state.spatial_uploader_key = 0

def reset_spatial_state():
    for _k in ["spatial_slices", "spatial_img", "spatial_px",
               "spatial_profiles", "spatial_groups", "spatial_line_coords",
               "spatial_last_upload"]:
        st.session_state[_k] = None
    # เพิ่ม counter → uploader จะ clear ตัวเอง
    st.session_state.spatial_uploader_key += 1

# ── Upload + Reset ──
_sp_up_col, _sp_rst_col = st.columns([4, 1])
with _sp_up_col:
    files = st.file_uploader(
        "Upload DICOM folder (multiple files)",
        type=None, accept_multiple_files=True,
        key=f"spatial_folder_{st.session_state.spatial_uploader_key}",
        label_visibility="collapsed")
with _sp_rst_col:
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    if st.button("🔄 Reset", use_container_width=True, key="sp_reset"):
        reset_spatial_state()
        st.rerun()

# ── Load ──
if files:
    file_names = sorted(f.name for f in files)
    if st.session_state.spatial_last_upload != file_names:
        reset_spatial_state()  # ← clear ทุกอย่างก่อน
        st.session_state.spatial_last_upload = file_names
        _sp_loaded = []
        for _f in files:
            try:
                import pydicom as _pydc
                _ds = _pydc.dcmread(io.BytesIO(_f.read()), force=True)
                _arr = _ds.pixel_array.astype(np.float32)
                _arr = _arr * float(getattr(_ds, "RescaleSlope", 1)) \
                           + float(getattr(_ds, "RescaleIntercept", 0))
                _sp_loaded.append({
                    "img": _arr,
                    "z":   int(getattr(_ds, "InstanceNumber", 0)),
                    "px":  float(getattr(_ds, "PixelSpacing", [1, 1])[0])
                })
            except Exception:
                pass
        if _sp_loaded:
            _sp_loaded.sort(key=lambda x: x["z"])
            st.session_state.spatial_slices = _sp_loaded
            _default = min(97, len(_sp_loaded) - 1)
            st.session_state.spatial_img = _sp_loaded[_default]["img"]
            st.session_state.spatial_px  = _sp_loaded[_default]["px"]
        st.rerun()

# ── Slice selector ──
if st.session_state.spatial_slices:
    _slices = st.session_state.spatial_slices
    st.success(f"✅ Loaded {len(_slices)} slices")
    _sp_idx = st.slider("Select slice", 0, len(_slices) - 1,
                    min(170, len(_slices) - 1), key="spatial_slice_idx")
    st.session_state.spatial_img = _slices[_sp_idx]["img"]
    st.session_state.spatial_px  = _slices[_sp_idx]["px"]

# ── Main UI ──
if st.session_state.spatial_img is not None:
    img = st.session_state.spatial_img
    h, w = img.shape

    # ── Controls row ──
    _wc1, _wc2, = st.columns([1, 1])
    with _wc1:
        wl = st.number_input("WL", min_value=-5000, max_value=5000, 
                                value=500, step=10, key="wl_v2")                           
    with _wc2:
        ww = st.number_input("WW", min_value=-5000, max_value=5000, 
                                value=2000, step=10, key="ww_v2")

    img_disp = np.clip((img - (wl - ww / 2)) / (ww + 1e-8), 0, 1)

    # ── Controls row ──
    _cc1, _cc2 = st.columns([1, 1])
    with _cc1:
        zoom = st.number_input("🔍 Zoom", min_value=1.0, max_value=8.0, 
                                    value=1.0, step=0.5, key="spatial_zoom")
    with _cc2:
        show_lines = st.checkbox("👁 แสดงเส้น sampling", value=True, key="spatial_show_lines")

    # ── Live image preview (re-renders every time controls change) ──
    dark_style()
    _fig_prev, _ax_prev = plt.subplots(figsize=(5, 5), facecolor=DARK)
    _ax_prev.imshow(img_disp, cmap="gray", vmin=0, vmax=1)

    # draw sampling lines from stored coords
    if show_lines and st.session_state.spatial_line_coords:
        _colors_lp = ["#00e5ff","#7b61ff","#ff9800","#00e676","#ff5252",
                      "#e040fb","#40c4ff","#ffea00","#ff6d00","#b2ff59",
                      "#ea80fc","#69ffb0"]
        for (_lx, _ly, _gi) in st.session_state.spatial_line_coords:
            _ax_prev.plot(_lx, _ly, color=_colors_lp[_gi % len(_colors_lp)],
                          lw=0.7, alpha=0.85)

    # draw group labels
    if "spatial_groups" in st.session_state and st.session_state.spatial_groups:

        for _gi, _g in enumerate(st.session_state.spatial_groups[:12], start=1):

            _offset_lbl = 12  # pixel offset

            _lbl_x = _g["cx"] + _offset_lbl * np.cos(_g["ang"])
            _lbl_y = _g["cy"] + _offset_lbl * np.sin(_g["ang"])

            _ax_prev.text(
                _lbl_x,
                _lbl_y,
                f"G{_gi}",
                color="yellow",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.55, pad=1)
            )

    # apply zoom (crop center)
    _cx_z, _cy_z = w / 2, h / 2
    _hw = (w / 2) / zoom
    _hh = (h / 2) / zoom
    _ax_prev.set_xlim(_cx_z - _hw, _cx_z + _hw)
    _ax_prev.set_ylim(_cy_z + _hh, _cy_z - _hh)
    _ax_prev.set_title(
        f"Zoom {zoom:.1f}×  |  Lines {'ON' if show_lines else 'OFF'}",
        fontsize=8, color=MUTED)
    _ax_prev.axis("off")
    plt.tight_layout(pad=0.2)
    st.image(fig2img(_fig_prev), use_container_width=True)
    plt.close(_fig_prev)
    
    if st.button("▶ Detect Line Pairs", use_container_width=True, key="sp_detect"):
        from scipy.ndimage import map_coordinates as _map_coords

        _yy, _xx = np.ogrid[:h, :w]
        _rr = np.sqrt((_xx - w/2)**2 + (_yy - h/2)**2)
        _r_min = 0.18 * min(h, w)
        _r_max = 0.46 * min(h, w)
        _ann   = (_rr >= _r_min) & (_rr <= _r_max)

        _vals = img_disp[_ann]
        if _vals.size == 0:
            st.error("❌ ไม่พบ LP annulus")
        else:
            _thr = np.mean(_vals) + 2.0 * np.std(_vals)
            _bw  = (img_disp >= _thr) & _ann
            _bw8 = (_bw.astype(np.uint8) * 255)
            _k3  = np.ones((3, 3), np.uint8)
            _bw8 = cv2.morphologyEx(_bw8, cv2.MORPH_OPEN,  _k3)
            _bw8 = cv2.morphologyEx(_bw8, cv2.MORPH_CLOSE, _k3)

            _nlab, _lab, _stats, _cents = cv2.connectedComponentsWithStats(_bw8, connectivity=8)

            _cands = []
            for _ii in range(1, _nlab):
                _area = _stats[_ii, cv2.CC_STAT_AREA]
                if _area < 8 or _area > 400:
                    continue
                _cx_i, _cy_i = _cents[_ii]
                _r_i   = float(np.sqrt((_cx_i - w/2)**2 + (_cy_i - h/2)**2))
                _ang_i = float(np.arctan2(_cy_i - h/2, _cx_i - w/2))
                _ys_i, _xs_i = np.where(_lab == _ii)
                if len(_xs_i) < 6:
                    continue
                _pts_i = np.column_stack([_xs_i, _ys_i]).astype(np.float32)
                _cov_i = np.cov((_pts_i - _pts_i.mean(0)).T)
                if _cov_i.shape != (2, 2):
                    continue
                _ev, _evec = np.linalg.eigh(_cov_i)
                _maj = _evec[:, np.argmax(_ev)]
                _vx_i, _vy_i = float(_maj[0]), float(_maj[1])
                _nn_i = np.hypot(_vx_i, _vy_i) + 1e-8
                _cands.append({"cx": _cx_i, "cy": _cy_i, "r": _r_i, "ang": _ang_i,
                               "vx": _vx_i / _nn_i, "vy": _vy_i / _nn_i})

            if len(_cands) < 4:
                st.error(f"❌ detect ได้แค่ {len(_cands)} candidates — ปรับ WL/WW")
            else:
                _r_med = float(np.median([c["r"] for c in _cands]))
                _r_tol = max(15, 0.10 * _r_med)
                _ring  = sorted([c for c in _cands if abs(c["r"] - _r_med) <= _r_tol],
                                key=lambda c: c["ang"])

                _groups = []
                _used   = [False] * len(_ring)
                for _ii in range(len(_ring)):
                    if _used[_ii]: continue
                    _grp = [_ring[_ii]]; _used[_ii] = True
                    for _jj in range(_ii + 1, len(_ring)):
                        if _used[_jj]: continue
                        _da = abs(np.angle(np.exp(1j * (_ring[_jj]["ang"] - _ring[_ii]["ang"]))))
                        _dd = np.hypot(_ring[_jj]["cx"] - _ring[_ii]["cx"],
                                       _ring[_jj]["cy"] - _ring[_ii]["cy"])
                        if _da < np.deg2rad(14) and _dd < 32:
                            _grp.append(_ring[_jj]); _used[_jj] = True
                    _cxg = float(np.mean([z["cx"] for z in _grp]))
                    _cyg = float(np.mean([z["cy"] for z in _grp]))
                    _angs_g = np.array([z["ang"] for z in _grp])
                    _angg = float(np.angle(np.mean(np.exp(1j * _angs_g))))
                    _vxg = float(np.mean([z["vx"] for z in _grp]))
                    _vyg = float(np.mean([z["vy"] for z in _grp]))
                    _nv  = np.hypot(_vxg, _vyg) + 1e-8
                    _groups.append({"cx": _cxg, "cy": _cyg,
                                    "r": float(np.mean([z["r"] for z in _grp])),
                                    "ang": _angg,
                                    "vx": _vxg / _nv, "vy": _vyg / _nv})
                _groups.sort(key=lambda g: g["ang"])

                _profs       = []
                _line_coords = []
                _slen = 18; _ostep = 3; _nsamp = 90
                for _gi, _g in enumerate(_groups[:12]):
                    _gx, _gy  = _g["cx"], _g["cy"]
                    _ang_g    = _g["ang"]
                    _radial_x =  float(np.cos(_ang_g))
                    _radial_y =  float(np.sin(_ang_g))
                    _tang_x   = -_radial_y
                    _tang_y   =  _radial_x
                    _nx, _ny  = _tang_x, _tang_y
                    _tx, _ty  = _radial_x, _radial_y
                    for _k in (-1, 0, 1):
                        _t  = np.linspace(-_slen, _slen, _nsamp)
                        _xl = _gx + _t * _nx + _k * _ostep * _tx
                        _yl = _gy + _t * _ny + _k * _ostep * _ty
                        _vd = (_xl >= 0) & (_xl <= w-1) & (_yl >= 0) & (_yl <= h-1)
                        _xl, _yl = _xl[_vd], _yl[_vd]
                        if len(_xl) < 15: continue
                        _vp = _map_coords(img, [_yl, _xl], order=1, mode="nearest")
                        _vp = (_vp - _vp.min()) / (_vp.max() - _vp.min() + 1e-8)
                        _profs.append(_vp)
                        _line_coords.append((_xl, _yl, _gi))

                st.session_state.spatial_groups      = _groups
                st.session_state.spatial_profiles    = _profs
                st.session_state.spatial_line_coords = _line_coords
                st.success(f"✅ Detected {len(_groups)} LP groups, {len(_profs)} profiles")
                st.rerun()

    lp_value = st.selectbox(
        "Line Pair (lp/cm)",
        [1,2,3,4,5,6,7,8],
        key="spatial_lp_main"
    )

    if "spatial_profiles" not in st.session_state or not st.session_state.spatial_profiles:
        st.info("กด Detect ก่อน")

    else:
        if st.button("▶ Compute MTF", use_container_width=True, key="sp_compute"):

            _profs_all = st.session_state.spatial_profiles
            _idx_s = (lp_value - 1) * 3
            _idx_e = min(lp_value * 3, len(_profs_all))

            if _idx_s >= len(_profs_all):
                st.error(f"❌ ไม่มี group สำหรับ {lp_value} lp/cm")
                st.stop()

            _sel = _profs_all[_idx_s:_idx_e]

            if len(_sel) == 0:
                st.error(f"❌ ไม่พบ profiles ของ {lp_value}")
                st.stop()

            _min_len = min(len(p) for p in _sel)
            _sel_eq = np.asarray([p[:_min_len] for p in _sel], dtype=np.float32)
            _y_avg = np.mean(_sel_eq, axis=0)

            _Imax = float(np.max(_y_avg))
            _Imin = float(np.min(_y_avg))
            _mtf_point = (_Imax - _Imin) / (_Imax + _Imin + 1e-8)

            _px_val = st.session_state.get("spatial_px", 1.0)
            _px = float(_px_val[0]) if isinstance(_px_val, (list, tuple, np.ndarray)) else float(_px_val)

            _y_fft = _y_avg.copy()
            _N = len(_y_fft)
            _edge = max(5, int(0.1 * _N))
            _bg = (np.mean(_y_fft[:_edge]) + np.mean(_y_fft[-_edge:])) / 2.0
            _y_fft = _y_fft - _bg
            _y_fft = _y_fft - np.min(_y_fft)
            _y_fft = _y_fft * np.hanning(_N)

            _Nfft = 4 * _N
            _Y = np.abs(np.fft.fft(_y_fft, _Nfft))[:_Nfft // 2]

            if _Y[0] <= 1e-12:
                st.error("❌ FFT normalization failed")
                st.stop()

            _mtf_curve = _Y / _Y[0]
            _f_lp_cm = np.arange(_Nfft // 2) / (_Nfft * _px) * 10.0
            _mtf_curve = np.maximum.accumulate(_mtf_curve[::-1])[::-1]

            def find_level(level):
                for i in range(1, len(_mtf_curve)):
                    if _mtf_curve[i] <= level:
                        return np.interp(
                            level,
                            [_mtf_curve[i-1], _mtf_curve[i]],
                            [_f_lp_cm[i-1], _f_lp_cm[i]]
                        )
                return np.nan

            _f50 = find_level(0.5)
            _f10 = find_level(0.1)

            # ── บันทึกผลทั้งหมดลง session state ──
            st.session_state["sr_last_lp"]      = lp_value
            st.session_state["sr_last_mtf"]     = _mtf_point
            st.session_state["sr_last_Imax"]    = _Imax
            st.session_state["sr_last_Imin"]    = _Imin
            st.session_state["sr_last_mean"]    = float(np.mean(_y_avg))
            st.session_state["sr_last_f50"]     = _f50
            st.session_state["sr_last_f10"]     = _f10
            st.session_state["sr_last_sel_eq"]  = _sel_eq
            st.session_state["sr_last_y_avg"]   = _y_avg
            st.session_state["sr_last_f_lp_cm"] = _f_lp_cm
            st.session_state["sr_last_mtf_curve"] = _mtf_curve
            st.session_state["sr_computed"]     = True

        # ── แสดงผล (ข้างนอก button block → คงอยู่หลัง rerun) ──
        if st.session_state.get("sr_computed"):

            _lp_val    = st.session_state["sr_last_lp"]
            _mtf_val   = st.session_state["sr_last_mtf"]
            _Imax      = st.session_state["sr_last_Imax"]
            _Imin      = st.session_state["sr_last_Imin"]
            _mean_val  = st.session_state["sr_last_mean"]
            _f50       = st.session_state["sr_last_f50"]
            _f10       = st.session_state["sr_last_f10"]
            _sel_eq    = st.session_state["sr_last_sel_eq"]
            _y_avg     = st.session_state["sr_last_y_avg"]
            _f_lp_cm   = st.session_state["sr_last_f_lp_cm"]
            _mtf_curve = st.session_state["sr_last_mtf_curve"]

            st.write("MTF50:", _f50)
            st.write("MTF10:", _f10)

            _c1, _c2 = st.columns(2)

            with _c1:
                fig1, ax1 = plt.subplots()
                for _p in _sel_eq:
                    ax1.plot(_p, color="#9aa5b1", alpha=0.4, lw=0.8)
                ax1.plot(_y_avg, color="#00e5ff", lw=2, label="Mean profile")
                ax1.axhline(_Imax, color="#ff5252", lw=1.2, ls="--", label=f"Imax = {_Imax:.3f}")
                ax1.axhline(_Imin, color="#7b61ff", lw=1.2, ls="--", label=f"Imin = {_Imin:.3f}")
                ax1.axhline(_mean_val, color="#ffd54f", lw=1.0, ls=":", label=f"Mean = {_mean_val:.3f}")
                txt = (
                    f"LP = {_lp_val} lp/cm\n"
                    f"MTF = {_mtf_val:.3f}\n"
                    f"Imax = {_Imax:.3f}\n"
                    f"Imin = {_Imin:.3f}"
                )
                ax1.text(0.02, 0.95, txt, transform=ax1.transAxes, fontsize=8,
                         verticalalignment='top',
                         bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=4))
                ax1.set_title(f"Profile @ {_lp_val} lp/cm")
                ax1.set_xlabel("Sample")
                ax1.set_ylabel("Normalized HU")
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=7)
                st.pyplot(fig1)
                plt.close(fig1)

            with _c2:
                fig2, ax2 = plt.subplots()
                ax2.plot(_f_lp_cm, _mtf_curve, lw=2)
                ax2.axhline(0.5, ls="--")
                ax2.axhline(0.1, ls="--")
                if not np.isnan(_f50):
                    ax2.axvline(_f50, ls="--", label=f"MTF50={_f50:.2f}")
                if not np.isnan(_f10):
                    ax2.axvline(_f10, ls="--", label=f"MTF10={_f10:.2f}")
                ax2.set_xlim(0, 12)
                ax2.set_ylim(0, 1.05)
                ax2.set_title(f"MTF Curve @ {_lp_val} lp/cm")
                ax2.legend()
                ax2.grid()
                st.pyplot(fig2)
                plt.close(fig2)

            st.metric(f"MTF @ {_lp_val} lp/cm", f"{_mtf_val:.3f}")

            st.markdown("---")

            # ── Assessment ──
            _sr_choice = st.radio(
                "Spatial Resolution Assessment",
                options=[
                    "✅  Resolved ≥ 5 lp/cm  (PASS)",
                    "❌  Not resolved ≥ 5 lp/cm  (FAIL)"
                ],
                horizontal=True,
                index=None,
                key="sr_choice_final"
            )

            if _sr_choice == "✅  Resolved ≥ 5 lp/cm  (PASS)":
                st.markdown(f"""
                <div class="okbox" style="display:flex;align-items:center;gap:12px;
                  padding:12px 18px;margin-top:8px;">
                  <span style="font-size:26px;line-height:1;">🟢</span>
                  <div>
                    <div style="font-weight:700;font-size:13px;color:#70ffc0;">
                      PASS — Spatial Resolution
                    </div>
                    <div style="font-size:10px;color:#50e090;margin-top:3px;">
                      Resolved ≥ 5 lp/cm at <b>{_lp_val} lp/cm</b> —
                      meets IAEA HHS No.19 criterion
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.session_state["sr_pass"] = True
                st.session_state["sr_has"]  = True

            elif _sr_choice == "❌  Not resolved ≥ 5 lp/cm  (FAIL)":
                st.markdown(f"""
                <div class="warnbox" style="display:flex;align-items:center;gap:12px;
                  padding:12px 18px;margin-top:8px;">
                  <span style="font-size:26px;line-height:1;">🔴</span>
                  <div>
                    <div style="font-weight:700;font-size:13px;color:#ff8a80;">
                      FAIL — Spatial Resolution
                    </div>
                    <div style="font-size:10px;color:#f5c060;margin-top:3px;">
                      Not resolved ≥ 5 lp/cm at <b>{_lp_val} lp/cm</b> —
                      does not meet IAEA HHS No.19 criterion
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.session_state["sr_pass"] = False
                st.session_state["sr_has"]  = True

            else:
                st.markdown("""
                <div style="background:#0a0f18;border:1px dashed #1e2a3a;
                  border-radius:8px;padding:12px 18px;font-size:11px;
                  color:#3a5060;text-align:center;margin-top:8px;">
                  ⚪ &nbsp; Select assessment result to record
                </div>""", unsafe_allow_html=True)
                st.session_state["sr_pass"] = False
                st.session_state["sr_has"]  = False

    # ── Basic QC Summary ──
    st.markdown("---")
    _geo_ok  = (st.session_state.get("basic_geometry_result") or {}).get("pass", False)
    _sq_ok   = (st.session_state.get("basic_square_result")  or {}).get("pass", False)
    _lin_ok  = (st.session_state.get("basic_linearity_result") or {}).get("pass", False)
    _acc_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("acc_pass", False)
    _uni_ok  = (st.session_state.get("basic_accuracy_result") or {}).get("uni_pass", False)
    _fine_ok   = (st.session_state.get("basic_slice_fine")   or {}).get("pass", False)
    _coarse_ok = (st.session_state.get("basic_slice_coarse") or {}).get("pass", False)
    _st_ok   = _fine_ok and _coarse_ok
    _lc_ok   = st.session_state.get("lc_detect_v2", False)
    _sr_ok   = st.session_state.get("sr_pass", False)

    _has_geo = st.session_state.get("basic_geometry_result")  is not None
    _has_sq  = st.session_state.get("basic_square_result")    is not None
    _has_lin = st.session_state.get("basic_linearity_result") is not None
    _has_acc = st.session_state.get("basic_accuracy_result")  is not None
    _has_nz  = st.session_state.get("basic_noise_result")     is not None
    _has_st  = (st.session_state.get("basic_slice_fine")   is not None or
                st.session_state.get("basic_slice_coarse") is not None)
    _has_lc  = (st.session_state.get("lc_detect_v2",     False) or
                st.session_state.get("lc_not_detect_v2", False))
    _has_sr  = st.session_state.get("sr_has", False)

    def _icon(has, ok):
        if not has: return "⚪"
        return "🟢" if ok else "🔴"

    _nz_pass_flag = (st.session_state.get("basic_noise_result") or {}).get("pass")
    _nz_disp_ok   = _nz_pass_flag is True or _nz_pass_flag is None

    st.markdown(f"""
    <div class="qc-card" style="margin-top:4px;">
      <div class="qc-title" style="font-size:13px;">📋 Basic QC Summary</div>
      <div style="display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;
        font-size:12px;color:#d8e8ff;">
        <span>{_icon(_has_geo,_geo_ok)} Geometry QC</span>
        <span>{_icon(_has_sq,_sq_ok)} Square 50mm</span>
        <span>{_icon(_has_lin,_lin_ok)} CT Linearity</span>
        <span>{_icon(_has_acc,_acc_ok)} CT Accuracy</span>
        <span>{_icon(_has_acc,_uni_ok)} Uniformity</span>
        <span>{_icon(_has_nz,_nz_disp_ok)} Noise QC</span>
        <span>{_icon(_has_st,_st_ok)} Slice Thickness</span>
        <span>{_icon(_has_lc,_lc_ok)} Low Contrast</span>
        <span>{_icon(_has_sr,_sr_ok)} Spatial Resolution</span>
      </div>
    </div>""", unsafe_allow_html=True)

    _all_done = all([_has_geo, _has_sq, _has_lin, _has_acc,
                     _has_nz, _has_st, _has_lc, _has_sr])
    if _all_done:
        _nz_real_pass = (st.session_state.get("basic_noise_result") or {}).get("pass")
        _all_ok = all([_geo_ok, _sq_ok, _lin_ok, _acc_ok, _uni_ok,
                       _st_ok, _lc_ok, _sr_ok,
                       (_nz_real_pass is True or _nz_real_pass is None)])
        if _all_ok:
            st.markdown(
                '<div class="okbox" style="font-size:13px;font-weight:700;">'
                '🟢 ALL BASIC QC TESTS PASSED</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warnbox" style="font-size:13px;font-weight:700;">'
                '🔴 SOME TESTS FAILED — Review results above</div>',
                unsafe_allow_html=True)
# ─────────────────────────── VIEWER PAGE ──────────────────────────────────
if _page == "viewer":
    _topbar("DICOM Viewer", "📷")
    import base64

    # ── Handle wheel scroll via query param ──
    qp = st.query_params
    if "slice" in qp:
        try:
            new_idx = int(qp["slice"])
            if 0 <= new_idx < n and new_idx != st.session_state.current:
                st.session_state.current = new_idx
                st.query_params.clear()
                st.rerun()
        except Exception:
            st.query_params.clear()

    col_img, col_ctrl = st.columns([4, 1])

    with col_img:
        # ── Always use current slice (updated by wheel/sidebar) ──
        _cur_idx = st.session_state.current
        _sl      = st.session_state.slices[_cur_idx]
        hu_cur   = _sl["hu_mod"] if _sl.get("hu_mod") is not None else _sl["hu_orig"]
        img_arr  = wl_ww(hu_cur, wl, ww_val)   # uint8 grayscale

        rows_px, cols_px = img_arr.shape

        # ── Render: PIL → base64 PNG (no overlays — show raw inserted image) ──
        from PIL import Image as _PIL
        import base64

        pil_rgb = _PIL.fromarray(img_arr).convert("RGB")

        _buf = io.BytesIO()
        pil_rgb.save(_buf, format="PNG")
        b64_plot = base64.b64encode(_buf.getvalue()).decode()

        # ── iframe height = exact pixel height of image ──
        # col width ≈ 4/5 × 960px = 768px; scale height proportionally
        est_w    = 768
        est_h    = int(est_w * rows_px / cols_px)
        iframe_h = est_h + 30   # +30 for HUD

        # Series-aware navigation: wheel scrolls within active series only
        _ser_idxs_view = [st.session_state.slices.index(s)
                          for s in get_active_slices(st.session_state.slices,
                                                     st.session_state.active_series)]
        _ser_pos = _ser_idxs_view.index(_cur_idx) if _cur_idx in _ser_idxs_view else 0
        _ser_n   = len(_ser_idxs_view)
        # Pass global index list to JS as JSON
        import json as _json
        _idx_json = _json.dumps(_ser_idxs_view)

        # Build wheel HTML with correct f-string formatting
        _hud_txt  = f"Slice {_ser_pos+1} / {_ser_n}  |  🖱 scroll เพื่อเลื่อน"
        _title_txt = f"{_sl['name']}  |  WL {wl} / WW {ww_val}"
        wheel_html = (
            "<style>"
            "html,body{margin:0;padding:0;background:#080b10;overflow:hidden;}"
            "#vwrap{position:relative;width:100%;line-height:0;}"
            "#vimg{width:100%;height:auto;display:block;border-radius:4px;}"
            "#hud{position:absolute;bottom:6px;left:6px;"
            "background:rgba(0,0,0,.72);color:#00e5ff;"
            "font:12px monospace;padding:3px 10px;"
            "border-radius:4px;pointer-events:none;}"
            "#title{color:#3a5060;font:10px monospace;padding:2px 6px;background:#080b10;}"
            "</style>"
            f'<div id="title">{_title_txt}</div>'
            f'<div id="vwrap">'
            f'  <img id="vimg" src="data:image/png;base64,{b64_plot}">'
            f'  <div id="hud">{_hud_txt}</div>'
            "</div>"
            "<script>(function(){"
            f"var idxs={_idx_json};"
            f"var pos={_ser_pos};"
            "var hud=document.getElementById('hud'), t=null;"
            "document.getElementById('vwrap').addEventListener('wheel',function(e){"
            "e.preventDefault();"
            "pos=Math.max(0,Math.min(idxs.length-1,pos+(e.deltaY>0?1:-1)));"
            "hud.textContent='Slice '+(pos+1)+' / '+idxs.length+'  |  🖱 scroll เพื่อเลื่อน';"
            "clearTimeout(t);"
            "t=setTimeout(function(){"
            "var u=new URL(window.parent.location.href);"
            "u.searchParams.set('slice',idxs[pos]);"
            "window.parent.history.replaceState(null,'',u.toString());"
            "window.parent.dispatchEvent(new PopStateEvent('popstate'));"
            "},220);"
            "},{passive:false});"
            "})();</script>"
        )
        import streamlit.components.v1 as components
        components.html(
            f"""
        <div style="display:flex; flex-direction:row; gap:8px">

        <div style="flex:1">
        {wheel_html}
        </div>

        <div style="
        width:16px;
        height:{iframe_h}px;
        overflow-y:scroll;
        background:#0e1520;
        border:1px solid #1e2a3a;
        ">

        <div style="height:{n*40}px"></div>

        </div>

        </div>
        """,
        height=iframe_h,
        scrolling=False
        )
    # ── Step 2: Scroll slice ──
    st.markdown('<div class="secLabel">Step 2 · Browse Slices</div>', unsafe_allow_html=True)
    st.caption("ใช้ slider หรือ scroll ใน viewer เพื่อเลื่อน")
    cur_view = st.slider("Slice", 0, max(0, n_view-1),
                             _view_pos, 1,
                             label_visibility="collapsed")
    st.session_state.current = _global_idxs[cur_view] if _global_idxs else 0
    with col_ctrl:

        st.markdown("### Slice info")

        u_flag = "✓ Uniformity" if _sl["is_uniformity"] else "· Not uniform"

        st.code(
            f"File  : {_sl['name']}\n"
            f"Type  : {u_flag}\n"
            f"Size  : {_sl['rows']}×{_sl['cols']}\n"
            f"ps    : {_sl['pixel_spacing']:.4f} mm\n"
            f"ST    : {_sl['slice_thickness']:.2f} mm\n"
            f"kVp   : {_sl.get('kvp','')}\n"
            f"mAs   : {_sl.get('mas','')}\n"
            f"Kernel: {_sl.get('kernel','')}\n"
            f"Series: {_sl.get('series_desc','')}\n"
            f"Recon : {_sl.get('recon_method','')}\n"
            f"IR/DL : {_sl.get('iter_desc','') or _sl.get('dlir_level','')}\n"
            f"Lesions: {len(_sl['lesions'])}",
            language="text"
        )

    # ─────────────────────────────────────────
    # 1. Uniformity Detection
    # ─────────────────────────────────────────
    st.divider()
    st.markdown("## 1. Uniformity Detection")

    # ── Step 3: Detect Uniformity ──
    st.markdown('<div class="secLabel">Step 3 · Detect Uniformity Module</div>', unsafe_allow_html=True)

    det_thresh = st.slider("Detection threshold", 10, 120, 55, 5,
                               help="Max=120 | ลดถ้าหาไม่เจอ | เพิ่มถ้าได้ module ผิด")

    if st.button("Scan phantom modules"):

        detected = scan_module_slices(st.session_state.slices)

        st.session_state.detected_slices = detected

    if "detected_slices" in st.session_state:

        idx = st.selectbox(
            "Select phantom slice",
            st.session_state.detected_slices
        )

        sl = st.session_state.slices[idx]

        st.image(
            wl_ww(sl["hu_orig"],0,400),
            clamp=True
        )

    if st.button("🔍 Auto-detect Uniformity Slices", use_container_width=True):
            # Detect only within active series
            _active = get_active_slices(st.session_state.slices,
                                        st.session_state.active_series)
            _active_global_idxs = [st.session_state.slices.index(s) for s in _active]
            with st.spinner(f"Scoring {len(_active)} slices in series…"):
                # rank_uniformity_slices returns (local_idx, score, breakdown)
                # but we pass subset → remap to global idx
                ranked_local = rank_uniformity_slices(_active)
                ranked = [(_active_global_idxs[li], sc, bd)
                          for li, sc, bd in ranked_local]
                for idx, sc, bd in ranked:
                    st.session_state.slices[idx]["uni_score"] = sc
                    st.session_state.slices[idx]["uni_bd"]    = bd
                idxs = [idx for idx, sc, bd in ranked if sc >= det_thresh]
            for i, s in enumerate(st.session_state.slices):
                s["is_uniformity"] = (i in idxs)
            st.session_state.uniformity_idx    = idxs
            st.session_state.uniformity_ranked = ranked
            st.session_state.qc_status["uniformity"] = (len(idxs) > 0)
            slog(f"Uniformity: {len(idxs)} slices (series={st.session_state.active_series}, threshold={det_thresh})", "ok")
            st.rerun()

    uni_idxs = st.session_state.uniformity_idx
    ranked   = st.session_state.get("uniformity_ranked", [])

    # ── Score table top-15 ──
    if ranked:
            import pandas as pd
            top15 = ranked[:15]
            df_rank = pd.DataFrame([{
                "Slice": st.session_state.slices[idx]["name"],
                "Score": sc,
                "mu":   st.session_state.slices[idx].get("uni_bd",{}).get("mu",""),
                "std":  st.session_state.slices[idx].get("uni_bd",{}).get("inner_std",""),
                "✓":    "✓" if idx in uni_idxs else ""
            } for idx, sc, bd in top15])
            st.dataframe(df_rank, use_container_width=True,
                         height=min(240, len(top15)*35+38), hide_index=True)

# ── Preview slider for ALL detected slices ──
uni_idxs = st.session_state.uniformity_idx
ranked   = st.session_state.get("uniformity_ranked", [])

if uni_idxs:
    st.markdown(
        f'<div class="okbox">✓ พบ {len(uni_idxs)} uniformity slices</div>',
        unsafe_allow_html=True
    )

    # ── Preview detected slices ──
    if len(uni_idxs) > 1:
        st.markdown("**Preview detected slices**")

        if "uni_preview_slider" not in st.session_state:
            st.session_state.uni_preview_slider = 0

        preview_pos = st.slider(
            "Navigate Uniformity Slices",
            0,
            len(uni_idxs) - 1,
            st.session_state.uni_preview_slider,
            1,
            key="uni_preview_slider"
        )

        prev_idx = uni_idxs[preview_pos]
        prev_sl  = st.session_state.slices[prev_idx]
        prev_hu  = prev_sl["hu_orig"]
        prev_img = wl_ww(prev_hu, st.session_state.wl, st.session_state.ww)

        dark_style()
        fig_p, ax_p = plt.subplots(figsize=(3, 3), facecolor=DARK)
        ax_p.imshow(prev_img, cmap="gray", vmin=0, vmax=255)
        ax_p.axis("off")

        sc_val = prev_sl.get("uni_score", 0)
        bd_val = prev_sl.get("uni_bd", {})

        ax_p.set_title(
            f"{prev_sl['name']}\n"
            f"Score={sc_val:.0f}  μ={bd_val.get('mu','?')}  σ={bd_val.get('inner_std','?')}",
            fontsize=7,
            color=MUTED,
            pad=3
        )

        plt.tight_layout(pad=0.2)
        st.image(fig2img(fig_p), use_container_width=True)
        plt.close(fig_p)

    else:
        prev_idx = uni_idxs[0]
        prev_sl  = st.session_state.slices[prev_idx]
        prev_img = wl_ww(
            prev_sl["hu_orig"],
            st.session_state.wl,
            st.session_state.ww
        )
        st.image(
            prev_img,
            use_container_width=True,
            caption=f"Score={prev_sl.get('uni_score',0):.0f}"
        )

elif ranked:
    st.markdown(
        '<div class="warnbox">ไม่พบ slice ที่ผ่าน threshold — ลด threshold หรือเลือก manual</div>',
        unsafe_allow_html=True
    )

    top_idxs = [idx for idx, sc, bd in ranked[:15]]

    sel_manual = st.selectbox(
        "Manual select (top scored)",
        options=top_idxs,
        format_func=lambda i: (
            f"{st.session_state.slices[i]['name']} "
            f"[{st.session_state.slices[i].get('uni_score',0):.0f}]"
        ),
        key="sel_manual_box"
    )

    if st.button("✓ ใช้ Slice นี้ (Manual)", type="primary", use_container_width=True):
        st.session_state.slices[sel_manual]["is_uniformity"] = True
        st.session_state.uniformity_idx = [sel_manual]
        st.session_state.current = sel_manual
        slog(f"Manual: {st.session_state.slices[sel_manual]['name']}", "ok")
        st.rerun()

# ── Uniformity summary ──
if uni_idxs:
    _cur_sl = st.session_state.slices[st.session_state.current]
    _bd = _cur_sl.get("uni_bd", {})

    if _cur_sl.get("is_uniformity", False):
        c_uni1, c_uni2, c_uni3 = st.columns(3)
        c_uni1.metric("Uniformity score", f"{_cur_sl.get('uni_score', 0):.1f}")
        c_uni2.metric("Mean HU", f"{_bd.get('mu', '—')}")
        c_uni3.metric("Noise SD", f"{_bd.get('inner_std', '—')}")

# ── Range selection (uniformity only + realtime preview) ──
_uni = st.session_state.uniformity_idx

if _uni:
    st.markdown("### Select Insert Range (Uniformity slices only)")

    # เก็บตำแหน่ง range เป็น index ของ _uni ไม่ใช่ slice index จริง
    if "uni_range_slider" not in st.session_state:
        st.session_state.uni_range_slider = (0, len(_uni) - 1)

    # กันกรณี rerun แล้วจำนวน _uni เปลี่ยน
    _r0, _r1 = st.session_state.uni_range_slider
    _r0 = max(0, min(_r0, len(_uni) - 1))
    _r1 = max(0, min(_r1, len(_uni) - 1))
    if _r0 > _r1:
        _r0, _r1 = _r1, _r0
    st.session_state.uni_range_slider = (_r0, _r1)

    idx_range = st.slider(
        "Uniformity range",
        0,
        len(_uni) - 1,
        st.session_state.uni_range_slider,
        key="uni_range_slider"
    )

    slice_ids = _uni[idx_range[0]: idx_range[1] + 1]

    from_slice = slice_ids[0]
    to_slice   = slice_ids[-1]

    # sync กับ sidebar
    st.session_state.insert_from = from_slice
    st.session_state.insert_to   = to_slice

    st.markdown(
        f'<div class="okbox">'
        f'Insert range: <b>{from_slice+1}</b> → <b>{to_slice+1}</b> '
        f'({len(slice_ids)} slices)'
        f'</div>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**From slice preview**")
        sl_from = st.session_state.slices[from_slice]
        img_from = wl_ww(sl_from["hu_orig"], st.session_state.wl, st.session_state.ww)
        st.image(img_from, use_container_width=True)
        st.caption(f"[{from_slice+1}] {sl_from['name']}")

    with p2:
        st.markdown("**End slice preview**")
        sl_to = st.session_state.slices[to_slice]
        img_to = wl_ww(sl_to["hu_orig"], st.session_state.wl, st.session_state.ww)
        st.image(img_to, use_container_width=True)
        st.caption(f"[{to_slice+1}] {sl_to['name']}")

# ── End of viewer page ──
if _page == "viewer":
    st.stop()


# ─────────────────────────────────────────
# ADVANCED QC PAGE  (NPS + PVE + PSF + Simulation)
# ─────────────────────────────────────────
if _page in ("advqc", "simulation"):
    if _page == "advqc":
        _topbar("Advanced QC", "📈")
    else:
        _topbar("Image-Domain Simulation", "🔬")

    # Inline tabs inside the page for sub-sections
    _adv_tabs = st.tabs(["📊 NPS", "🔬 PVE", "🧬 PSF Bead", "🎯 Simulation"])
    _tab_nps, _tab_pv, _tab_psf, _tab_sim = _adv_tabs

    # also need shared insert sidebar for advqc
    with st.sidebar:
        st.markdown('<div class="secLabel">Insert Lesions</div>', unsafe_allow_html=True)
        insert_method = st.radio("Method", ["Image-domain", "Projection-domain"], index=0, key="sb_ins_method")
        use_proj = insert_method == "Projection-domain"
        contrast_val = st.slider("Contrast (HU)", -500, 100, -50, step=5, key="sb_contrast")
        diam_val = st.selectbox("Diameter (mm)", list(range(2,21)), index=15, key="sb_diam")
        method_key = "projection" if use_proj else "image"
        if st.button("⊕ Insert 5 Lesions", type="primary", use_container_width=True,
                     disabled=(not st.session_state.uniformity_idx), key="sb_insert_btn"):
            _insert_idxs = list(range(
                st.session_state.get("insert_from",0),
                st.session_state.get("insert_to",0)+1))
            with st.spinner(f"Inserting into {len(_insert_idxs)} slices..."):
                for idx in _insert_idxs:
                    insert_clock_lesions(st.session_state.slices[idx], contrast_val, diam_val, method=method_key)
            st.session_state.inserted_set1 = True
            slog(f"Inserted: contrast={contrast_val} diam={diam_val}mm", "ok")
            st.rerun()
        if st.button("✖ Clear Lesions", use_container_width=True, key="sb_clear_les"):
            for _s in st.session_state.slices:
                _s["hu_mod"] = None; _s["lesions"] = []
            st.rerun()

    # Open whichever sub-tab matches
    _default_adv = 3 if _page == "simulation" else 0

    # ═══════════════════════════════════════════════
    # TAB 1: NPS
    # ═══════════════════════════════════════════════
    with _tab_nps:
        st.caption("ส่วนนี้จะใช้ slice uniformity range ที่เลือกไว้ เพื่อคำนวณ Noise Power Spectrum (NPS)")

        # ── Slice range selector ──
        _uni_idx = st.session_state.uniformity_idx
        _n_total = len(st.session_state.slices)

        if _uni_idx:
            st.markdown('<div class="okbox" style="font-size:11px;">✓ ใช้ range จาก Uniformity Detection อัตโนมัติ</div>', unsafe_allow_html=True)
            _nps_from = st.session_state.get("insert_from", min(_uni_idx))
            _nps_to   = st.session_state.get("insert_to",   max(_uni_idx))
        else:
            st.markdown('<div class="warnbox" style="font-size:11px;">⚠ ยังไม่ detect uniformity — เลือก range ด้วยตนเอง</div>', unsafe_allow_html=True)
            _nps_from = 0
            _nps_to   = max(0, _n_total - 1)

        _nps_range = st.slider(
            "NPS slice range",
            0, max(0, _n_total - 1),
            (_nps_from, _nps_to),
            key="nps_range_slider",
            help="เลือก slice range สำหรับคำนวณ NPS (แนะนำ uniformity slices)"
        )
        _nps_from2, _nps_to2 = _nps_range

        # Preview start/end
        _pc1, _pc2 = st.columns(2)
        with _pc1:
            _sl_f = st.session_state.slices[_nps_from2]
            _img_f = wl_ww(_sl_f["hu_orig"], st.session_state.wl, st.session_state.ww)
            dark_style()
            _fig_f, _ax_f = plt.subplots(figsize=(3,3), facecolor=DARK)
            _ax_f.imshow(_img_f, cmap="gray"); _ax_f.axis("off")
            _ax_f.set_title(f"[{_nps_from2+1}] {_sl_f['name']}", fontsize=7, color=MUTED)
            plt.tight_layout(pad=0.1)
            st.image(fig2img(_fig_f), use_container_width=True)
            plt.close(_fig_f)
        with _pc2:
            _sl_t = st.session_state.slices[_nps_to2]
            _img_t = wl_ww(_sl_t["hu_orig"], st.session_state.wl, st.session_state.ww)
            dark_style()
            _fig_t, _ax_t = plt.subplots(figsize=(3,3), facecolor=DARK)
            _ax_t.imshow(_img_t, cmap="gray"); _ax_t.axis("off")
            _ax_t.set_title(f"[{_nps_to2+1}] {_sl_t['name']}", fontsize=7, color=MUTED)
            plt.tight_layout(pad=0.1)
            st.image(fig2img(_fig_t), use_container_width=True)
            plt.close(_fig_t)

        _nps_ids = list(range(_nps_from2, _nps_to2 + 1))
        st.markdown(f'<div class="infobox">📋 {len(_nps_ids)} slices selected (index {_nps_from2+1}–{_nps_to2+1})</div>', unsafe_allow_html=True)

        if len(_nps_ids) < 3:
            st.markdown('<div class="warnbox">AAPM TG-233 แนะนำใช้ ≥3 slices</div>', unsafe_allow_html=True)

        nps_col1, nps_col2 = st.columns([1, 1])
        with nps_col1:
            px_nps = st.session_state.slices[_nps_from2]["pixel_spacing"]
            nps_patch_mm = st.number_input(
                "NPS patch size (mm)",
                min_value=5.0, max_value=100.0, value=30.0, step=5.0,
                key="nps_patch_mm"
            )
            nps_patch_size = int(round(nps_patch_mm / px_nps))
            if nps_patch_size % 2 == 1:
                nps_patch_size += 1
            run_nps = st.button("▶ Run NPS", use_container_width=True, key="run_nps_btn")

        with nps_col2:
            st.markdown(
                '<div class="infobox">'
                'NPS uses a central square ROI on the selected uniformity slice range.<br>'
                'จะแสดง ROI overlay, noise texture, noise power map และ radial NPS plot'
                '</div>',
                unsafe_allow_html=True
            )

        st.caption("AAPM TG-233 แนะนำใช้ ≥3 slices เพื่อให้ NPS มีความเสถียร")

        # ── แสดงผลเดิม (ถ้ามี) ──
        if not run_nps and "nps_result_display" in st.session_state:
            _nps_disp = st.session_state["nps_result_display"]
            st.markdown(f'<div class="okbox">✓ NPS คำนวณแล้ว — {_nps_disp["n_slices"]} slices | กด ▶ Run NPS เพื่อคำนวณใหม่</div>', unsafe_allow_html=True)

        if run_nps:
            slice_ids = _nps_ids
            if len(slice_ids) == 0:
                st.warning("No slices selected for NPS")
            else:
                with st.spinner(f"Computing NPS from {len(slice_ids)} slices…"):
                    nps_stack = []
                    roi_positions = []
                    roi_stack = []

                    for idx in slice_ids:
                        sl_nps = st.session_state.slices[idx]
                        hu_nps = sl_nps["hu_orig"]
                        px_sl  = sl_nps["pixel_spacing"]
                        rois_nps = get_nps_rois(hu_nps, nps_patch_size)
                        for roi_n, (x_n, y_n) in rois_nps:
                            nps_stack.append(compute_nps_patch(roi_n, px_sl))
                            roi_stack.append(roi_n)
                            roi_positions.append((x_n, y_n))

                nps_mean = np.mean(nps_stack, axis=0)
                roi_mean = np.mean(roi_stack, axis=0)
                st.session_state["_nps_mean"]      = nps_mean
                st.session_state["_nps_roi_mean"]  = roi_mean
                st.session_state["_nps_roi_stack"] = roi_stack
                st.session_state["_nps_slice_ids"] = slice_ids
                st.session_state["_nps_px"]        = px_nps
                st.session_state["_nps_patch_size"] = nps_patch_size

                # ROI overlay
                ref_sl = st.session_state.slices[slice_ids[0]]
                hu_ref = ref_sl["hu_orig"]
                dark_style()
                fig_ov, ax_ov = plt.subplots(figsize=(5,5), facecolor=DARK)
                ax_ov.imshow(wl_ww(hu_ref, 0, 400), cmap="gray")
                for roi_n, (xr, yr) in get_nps_rois(hu_ref, nps_patch_size):
                    ax_ov.add_patch(plt.Rectangle((xr,yr), nps_patch_size, nps_patch_size,
                                                  edgecolor="red", linewidth=1.5, fill=False))
                ax_ov.set_title("AAPM NPS ROIs (+ pattern)", fontsize=9, color=MUTED)
                ax_ov.axis("off")
                plt.tight_layout(pad=0.2)
                st.image(fig2img(fig_ov), use_container_width=True)
                plt.close(fig_ov)

                c_img1, c_img2 = st.columns(2)
                with c_img1:
                    dark_style()
                    fig_nt, ax_nt = plt.subplots(figsize=(4,4), facecolor=DARK)
                    ax_nt.imshow(roi_mean, cmap="gray")
                    ax_nt.set_title("Noise texture", fontsize=9, color=MUTED)
                    ax_nt.axis("off")
                    st.image(fig2img(fig_nt), use_container_width=True)
                    plt.close(fig_nt)
                with c_img2:
                    dark_style()
                    fig_nm, ax_nm = plt.subplots(figsize=(4,4), facecolor=DARK)
                    ax_nm.imshow(np.log(nps_mean+1), cmap="inferno")
                    ax_nm.set_title("2D NPS (log)", fontsize=9, color=MUTED)
                    ax_nm.axis("off")
                    st.image(fig2img(fig_nm), use_container_width=True)
                    plt.close(fig_nm)

                # Radial NPS (AAPM TG-233)
                N_nps = nps_patch_size
                nps_accum = np.zeros((N_nps, N_nps), dtype=np.float64)
                n_rois_nps = 0
                hann_2d   = np.outer(np.hanning(N_nps), np.hanning(N_nps))
                hann_norm = float((hann_2d**2).mean())
                yy_nps, xx_nps = np.mgrid[0:N_nps, 0:N_nps]
                A_dt = np.column_stack([xx_nps.ravel(), yy_nps.ravel(), np.ones(N_nps*N_nps)])

                for hu_s in [st.session_state.slices[i]["hu_orig"] for i in slice_ids]:
                    rows_s, cols_s = hu_s.shape
                    try:
                        cx_p, cy_p, r_p = find_phantom_circle(hu_s)
                    except Exception:
                        cx_p, cy_p, r_p = cols_s//2, rows_s//2, min(rows_s,cols_s)//3
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            oy = int(cy_p + dr * N_nps)
                            ox = int(cx_p + dc * N_nps)
                            r0 = oy - N_nps//2; r1 = r0 + N_nps
                            c0 = ox - N_nps//2; c1 = c0 + N_nps
                            if r0 < 0 or c0 < 0 or r1 > rows_s or c1 > cols_s:
                                continue
                            if any(np.sqrt((x-cx_p)**2+(y-cy_p)**2) > r_p*0.70
                                   for x,y in [(ox-N_nps//2,oy-N_nps//2),(ox+N_nps//2,oy-N_nps//2),
                                               (ox-N_nps//2,oy+N_nps//2),(ox+N_nps//2,oy+N_nps//2)]):
                                continue
                            patch_s = hu_s[r0:r1, c0:c1].astype(np.float64)
                            cf, _, _, _ = np.linalg.lstsq(A_dt, patch_s.ravel(), rcond=None)
                            patch_dt = patch_s - (cf[0]*xx_nps + cf[1]*yy_nps + cf[2])
                            F = np.fft.fft2(patch_dt * hann_2d)
                            nps_accum += np.abs(F)**2 / hann_norm
                            n_rois_nps += 1

                if n_rois_nps == 0:
                    st.error("ไม่มี ROI")
                else:
                    nps_2d_final = np.fft.fftshift(nps_accum / n_rois_nps) * (px_nps**2) / (N_nps**2)
                    cy_c, cx_c = N_nps//2, N_nps//2
                    fy_r, fx_r = np.indices(nps_2d_final.shape)
                    r_int = np.round(np.sqrt((fx_r-cx_c)**2+(fy_r-cy_c)**2)).astype(int).ravel()
                    max_r_nps = min(cy_c, cx_c)
                    mask_r = r_int <= max_r_nps
                    tbin  = np.bincount(r_int[mask_r], nps_2d_final.ravel()[mask_r])
                    cnt_r = np.where(np.bincount(r_int[mask_r])==0, 1, np.bincount(r_int[mask_r]))
                    radial = tbin / cnt_r

                    window_sg = min(9, len(radial)-1)
                    if window_sg % 2 == 0: window_sg -= 1
                    radial_smooth = savgol_filter(radial, window_length=window_sg, polyorder=2)

                    df_nps_freq = 1.0 / (N_nps * px_nps)
                    freq_nps = np.arange(len(radial)) * df_nps_freq
                    nyquist_nps = 1.0 / (2.0 * px_nps)

                    fp_idx = int(np.argmax(radial_smooth[1:]) + 1)
                    fp = freq_nps[fp_idx]
                    w_nps = radial[1:]; fw_nps = freq_nps[1:]
                    fA = float((fw_nps * w_nps).sum() / w_nps.sum()) if w_nps.sum() > 0 else fp

                    dark_style()
                    fig_nps, ax_nps = plt.subplots(figsize=(7, 4), facecolor=DARK)
                    ax_nps.set_facecolor(DARK)
                    ax_nps.plot(freq_nps, radial, color="#3a5060", lw=1.0, alpha=0.5, label="NPS")
                    ax_nps.plot(freq_nps, radial_smooth, color="#00e5ff", lw=1.8, label="NPS (smoothed)")
                    ax_nps.axvline(fp,         color="white",   ls="--", lw=0.8, label=f"f_peak={fp:.3f} mm⁻¹")
                    ax_nps.axvline(fA,         color="#00ffaa", ls="-",  lw=0.8, label=f"f_avg={fA:.3f} mm⁻¹")
                    ax_nps.axvline(nyquist_nps,color="#ff9800", ls=":",  lw=1.5, label=f"Nyquist={nyquist_nps:.3f} mm⁻¹")
                    ax_nps.set_title("Radially averaged NPS (AAPM TG-233)", color=MUTED, fontsize=10)
                    ax_nps.set_xlabel("Spatial Frequency [mm⁻¹]", color=MUTED)
                    ax_nps.set_ylabel("NPS [HU²·mm²]", color=MUTED)
                    ax_nps.set_xlim(0, nyquist_nps)
                    ax_nps.set_ylim(0, radial_smooth.max() * 1.15)
                    ax_nps.tick_params(colors=MUTED)
                    ax_nps.legend(fontsize=8, labelcolor="white", facecolor=DARK, edgecolor=MUTED)
                    for sp in ax_nps.spines.values(): sp.set_edgecolor(MUTED)
                    st.pyplot(fig_nps)
                    plt.close(fig_nps)

                    nps_area = float(np.trapz(radial_smooth, freq_nps))
                    noise_sd_nps = float(np.std(np.concatenate([r.flatten() for r in roi_stack])))

                    m1n, m2n, m3n = st.columns(3)
                    m1n.metric("Integrated NPS", f"{nps_area:.5f}")
                    m2n.metric("Noise SD (HU)", f"{noise_sd_nps:.2f}")
                    m3n.metric("ROIs used", str(n_rois_nps))

                    nps_result = {
                        "Slices": len(slice_ids),
                        "ROI size (px)": int(N_nps),
                        "Noise SD (HU)": round(noise_sd_nps, 3),
                        "Int. NPS": round(nps_area, 6),
                        "f_peak (mm⁻¹)": round(float(fp), 4),
                        "f_avg (mm⁻¹)": round(float(fA), 4),
                        "px (mm)": round(float(px_nps), 4)
                    }
                    st.session_state.nps_results.append(nps_result)
                    st.session_state.qc_status["nps"] = True
                    st.session_state["nps_cache"] = nps_2d_final
                    st.session_state["nps_result_display"] = {"n_slices": len(slice_ids)}
                    slog(f"NPS done: {len(slice_ids)} slices, f_peak={fp:.3f} mm⁻¹", "ok")

        if st.session_state.nps_results:
            st.markdown("### NPS Results Table")
            df_nps_t = pd.DataFrame(st.session_state.nps_results)
            st.dataframe(df_nps_t, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════
    # TAB 2: PVE
    # ═══════════════════════════════════════════════
    with _tab_pv:
        st.caption("เลือก slice ที่เป็น CTP404 Linearity Module (มี insert วัสดุต่างๆ)")

        # ── Slice selector ──
        _pve_range = st.slider(
            "Select slice range",
            0, n_view - 1, (0, n_view - 1),
            key="slice_range_manual"
        )
        start_local, end_local = _pve_range
        start_slice_pv = _active_idxs[start_local]
        end_slice_pv   = _active_idxs[end_local]
        st.session_state.insert_from = start_slice_pv
        st.session_state.insert_to   = end_slice_pv

        st.markdown("### Window Settings")
        col_wl_pv, col_ww_pv = st.columns(2)
        with col_wl_pv:
            st.session_state.wl = st.slider("Window Level (WL)", -1000, 1000, st.session_state.get("wl", 100), key="pve_wl_sl")
        with col_ww_pv:
            st.session_state.ww = st.slider("Window Width (WW)", 1, 1000, st.session_state.get("ww", 400), key="pve_ww_sl")

        col_pv1, col_pv2 = st.columns(2)
        with col_pv1:
            st.markdown("**Start slice**")
            sl_pv_s = _active[start_local]
            st.image(wl_ww(sl_pv_s["hu_orig"], st.session_state.wl, st.session_state.ww), use_container_width=True)
        with col_pv2:
            st.markdown("**End slice**")
            sl_pv_e = _active[end_local]
            st.image(wl_ww(sl_pv_e["hu_orig"], st.session_state.wl, st.session_state.ww), use_container_width=True)

        st.markdown(f'<div class="okbox">Series: <b>{start_local+1}</b> → <b>{end_local+1}</b> &nbsp;|&nbsp; Global: {start_slice_pv} → {end_slice_pv}</div>', unsafe_allow_html=True)

        selected_slice_pv = st.selectbox(
            "Select slice for PVE",
            list(range(start_slice_pv, end_slice_pv + 1)),
            format_func=lambda i: f"Slice {i+1}",
            key="pve_slice_sel"
        )
        st.session_state.current = selected_slice_pv
        sl_pv = st.session_state.slices[selected_slice_pv]

        dark_style()
        fig_pv0, ax_pv0 = plt.subplots(figsize=(5, 5), facecolor=DARK)
        ax_pv0.imshow(wl_ww(sl_pv["hu_orig"], st.session_state.wl, st.session_state.ww), cmap="gray")
        if "inserts" in sl_pv:
            roi_r_pv = int(8.5 / sl_pv["pixel_spacing"])
            for ins_pv in sl_pv["inserts"]:
                ax_pv0.add_patch(plt.Circle((ins_pv["cx"], ins_pv["cy"]), roi_r_pv,
                                            fill=False, edgecolor="red", lw=0.8))
                ax_pv0.text(ins_pv["cx"], ins_pv["cy"], f"{ins_pv['label']}",
                            color="yellow", fontsize=6, ha="center", va="center")
            ax_pv0.set_title(f"Detected {len(sl_pv['inserts'])} inserts", fontsize=9, color=MUTED)
        ax_pv0.axis("off")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_pv0)
        plt.close(fig_pv0)

        st.markdown(
            '<div class="mathbox">'
            '<b>Concept:</b><br>'
            'Measured signal = weighted average of materials within a voxel.<br><br>'
            '<b>Simple model:</b><br>'
            'HU_measured = f · HU_object + (1-f) · HU_background<br><br>'
            '<b>วัดจาก:</b> ESF ข้าม edge ของ insert → transition width 10–90%'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button("▶ Run Partial Volume", use_container_width=True, key="run_pve_btn"):
            with st.spinner("Detecting inserts…"):
                inserts_pv = detect_linearity_inserts(sl_pv["hu_orig"], sl_pv["pixel_spacing"])
            st.session_state["inserts"] = inserts_pv
            sl_pv["inserts"] = inserts_pv
            slog(f"PVE: {len(inserts_pv)} inserts detected", "ok")

        if "inserts" in st.session_state:
            inserts_pv = st.session_state["inserts"]
            materials_pv = {"Air":-1000,"PMP":-200,"LDPE":-100,"Polystyrene":-35,
                            "Acrylic":120,"Delrin":340,"Teflon":990}
            for ins_pv in inserts_pv:
                ins_pv["material"] = min(materials_pv, key=lambda k: abs(materials_pv[k] - ins_pv["hu_mean"]))

            rows_pv = []
            ps_pv = sl_pv["pixel_spacing"]
            for i_pv, ins_pv in enumerate(inserts_pv):
                profile_pv = radial_profile(sl_pv["hu_orig"], ins_pv["cx"], ins_pv["cy"], 60)
                x_pv = np.arange(len(profile_pv))
                try:
                    popt_pv, _ = curve_fit(sigmoid, x_pv, profile_pv, maxfev=5000)
                    a_pv, x0_pv, k_pv, b_pv = popt_pv
                    pve_width = (x0_pv + 2.197*k_pv - (x0_pv - 2.197*k_pv)) * ps_pv
                except RuntimeError:
                    pve_width = float("nan")

                profile_s_pv = gaussian_filter1d(profile_pv, sigma=1)
                lsf_pv = np.gradient(profile_s_pv)
                mtf_pv_raw = np.abs(np.fft.fft(lsf_pv))
                mtf_pv_raw = mtf_pv_raw / (np.max(mtf_pv_raw) + 1e-12)
                freq_pv = np.fft.fftfreq(len(lsf_pv), d=ps_pv)
                fp_pv = freq_pv[:len(freq_pv)//2]
                mp_pv = mtf_pv_raw[:len(mtf_pv_raw)//2]
                mtf50_pv = find_mtf_freq(mp_pv, fp_pv, 0.5)
                mtf10_pv = find_mtf_freq(mp_pv, fp_pv, 0.1)

                rows_pv.append({
                    "ROI": i_pv+1, "Material": ins_pv["material"],
                    "HU": round(ins_pv["hu_mean"],2), "SD": round(ins_pv["hu_sd"],2),
                    "PVE width (mm)": round(pve_width,3) if not np.isnan(pve_width) else "N/A",
                    "MTF50": round(mtf50_pv,4) if np.isfinite(mtf50_pv) else "N/A",
                    "MTF10": round(mtf10_pv,4) if np.isfinite(mtf10_pv) else "N/A",
                })

            df_pve = pd.DataFrame(rows_pv)
            st.markdown("### Partial Volume Metrics")
            st.dataframe(df_pve, use_container_width=True, hide_index=True)
            st.session_state.qc_status["pve"] = True

            if len(inserts_pv) > 0:
                last = inserts_pv[-1]
                profile_s_last = gaussian_filter1d(radial_profile(sl_pv["hu_orig"], last["cx"], last["cy"], 60), sigma=1)
                lsf_last = np.gradient(profile_s_last)
                mtf_last = np.abs(np.fft.fft(lsf_last)); mtf_last /= (np.max(mtf_last)+1e-12)
                x_last = np.arange(len(profile_s_last))
                freq_last = np.fft.fftfreq(len(lsf_last), d=sl_pv["pixel_spacing"])
                h_last = len(freq_last)//2
                dark_style()
                col_e, col_l, col_m = st.columns(3)
                with col_e:
                    fig_e, ax_e = plt.subplots(figsize=(3,3), facecolor=DARK)
                    ax_e.plot(x_last, profile_s_last, color="#ff9800", lw=1)
                    ax_e.set_title("ESF (last insert)", fontsize=9, color=MUTED)
                    ax_e.set_xlabel("Radius (px)", fontsize=7, color=MUTED)
                    ax_e.tick_params(colors=MUTED); ax_e.set_facecolor(SURF)
                    plt.tight_layout(pad=0.2); st.pyplot(fig_e); plt.close(fig_e)
                with col_l:
                    fig_l, ax_l = plt.subplots(figsize=(3,3), facecolor=DARK)
                    ax_l.plot(x_last, lsf_last, color="#7b61ff", lw=1)
                    ax_l.set_title("LSF", fontsize=9, color=MUTED)
                    ax_l.set_xlabel("Radius (px)", fontsize=7, color=MUTED)
                    ax_l.tick_params(colors=MUTED); ax_l.set_facecolor(SURF)
                    plt.tight_layout(pad=0.2); st.pyplot(fig_l); plt.close(fig_l)
                with col_m:
                    fig_m, ax_m = plt.subplots(figsize=(3,3), facecolor=DARK)
                    ax_m.plot(freq_last[:h_last], mtf_last[:h_last], color=ACC, lw=1)
                    ax_m.axhline(0.5, color="red", ls="--", lw=0.7)
                    ax_m.axhline(0.1, color="#9c27b0", ls="--", lw=0.7)
                    ax_m.set_title("MTF", fontsize=9, color=MUTED)
                    ax_m.set_xlabel("Freq (mm⁻¹)", fontsize=7, color=MUTED)
                    ax_m.set_ylim(0, 1.1); ax_m.tick_params(colors=MUTED); ax_m.set_facecolor(SURF)
                    plt.tight_layout(pad=0.2); st.pyplot(fig_m); plt.close(fig_m)

    # ═══════════════════════════════════════════════
    # TAB 3: PSF BEAD
    # ═══════════════════════════════════════════════
    with _tab_psf:
        st.caption("เลือก slice ที่เป็น CTP Bead Module")

        if "psf_wl" not in st.session_state:
            st.session_state.psf_wl = 100
        if "psf_ww" not in st.session_state:
            st.session_state.psf_ww = 400

        _psf_range = st.slider(
            "Select slice range",
            0, max(0, len(st.session_state.slices) - 1),
            (0, max(0, len(st.session_state.slices) - 1)),
            key="slice_range_psf"
        )
        start_psf, end_psf = _psf_range

        st.markdown("### Window Settings")
        col_wl_psf, col_ww_psf = st.columns(2)
        with col_wl_psf:
            st.session_state.psf_wl = st.slider("WL", -1000, 1000, st.session_state.psf_wl, key="psf_wl_slider")
        with col_ww_psf:
            st.session_state.psf_ww = st.slider("WW", 1, 2000, st.session_state.psf_ww, key="psf_ww_slider")

        col_psf1, col_psf2 = st.columns(2)
        with col_psf1:
            st.markdown("**Start slice**")
            st.image(wl_ww(st.session_state.slices[start_psf]["hu_orig"],
                           st.session_state.psf_wl, st.session_state.psf_ww), use_container_width=True)
        with col_psf2:
            st.markdown("**End slice**")
            st.image(wl_ww(st.session_state.slices[end_psf]["hu_orig"],
                           st.session_state.psf_wl, st.session_state.psf_ww), use_container_width=True)

        st.markdown(f'<div class="okbox">Slices: <b>{start_psf+1}</b> → <b>{end_psf+1}</b></div>', unsafe_allow_html=True)

        selected_slice_psf = st.selectbox(
            "Select slice for PSF",
            list(range(start_psf, end_psf + 1)),
            format_func=lambda i: f"Slice {i+1}",
            key="psf_slice_select"
        )
        st.session_state.current = selected_slice_psf
        sl_psf = st.session_state.slices[selected_slice_psf]

        st.markdown("### PSF Analysis")
        col_pt1, col_pt2 = st.columns(2)
        with col_pt1:
            bead_thresh = st.number_input("Bead threshold (HU)", 100, 4000, 600, 50, key="bead_thresh_psf")
        with col_pt2:
            bead_patch_half = st.number_input("Patch half-size (px)", 3, 30, 3, 1, key="bead_patch_half_psf")

        bead_psf = detect_single_bead(sl_psf["hu_orig"], thresh=bead_thresh)
        dark_style()
        fig_psf_prev, ax_psf_prev = plt.subplots(figsize=(5,5), facecolor=DARK)
        ax_psf_prev.imshow(wl_ww(sl_psf["hu_orig"], st.session_state.psf_wl, st.session_state.psf_ww), cmap="gray")
        if bead_psf is not None:
            cx_b, cy_b = bead_psf
            half_b = int(bead_patch_half)
            ax_psf_prev.add_patch(plt.Rectangle((cx_b-half_b, cy_b-half_b), 2*half_b, 2*half_b,
                                                 fill=False, edgecolor="red", linewidth=2))
            ax_psf_prev.plot(cx_b, cy_b, "r+", markersize=14, markeredgewidth=2)
            ax_psf_prev.add_patch(plt.Circle((cx_b, cy_b), 3, fill=False, edgecolor="yellow", lw=1.5))
            hu_peak_psf = float(sl_psf["hu_orig"][cy_b, cx_b])
            ax_psf_prev.text(cx_b + half_b + 5, cy_b,
                             f"bead ({cx_b},{cy_b})\nHU={hu_peak_psf:.0f}",
                             color="yellow", fontsize=7, va="center",
                             bbox=dict(facecolor="black", alpha=0.6, pad=2))
            ax_psf_prev.set_title(f"PSF bead at ({cx_b},{cy_b})", fontsize=9, color=MUTED)
        else:
            ax_psf_prev.set_title("No bead detected — ลด Threshold", fontsize=9, color=MUTED)
        ax_psf_prev.axis("off")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_psf_prev)
        plt.close(fig_psf_prev)

        if st.button("▶ Run PSF Analysis", use_container_width=True, key="run_psf_btn"):
            hu_psf = sl_psf["hu_orig"]
            ps_psf = sl_psf["pixel_spacing"]
            bead_found = detect_single_bead(hu_psf, thresh=bead_thresh)
            if bead_found is None:
                st.warning("ไม่พบ bead — ลด threshold")
            else:
                cx_psf, cy_psf = bead_found
                half_psf = int(bead_patch_half) * 2
                patch_psf = hu_psf[cy_psf-half_psf:cy_psf+half_psf+1,
                                   cx_psf-half_psf:cx_psf+half_psf+1]
                center_psf = patch_psf.shape[0] // 2
                line_psf = patch_psf[center_psf].astype(float)
                line_psf -= np.min(line_psf)
                if np.max(line_psf) > 0: line_psf /= np.max(line_psf)
                mtf_psf = np.abs(np.fft.fft(line_psf))
                mtf_psf /= (np.max(mtf_psf) + 1e-12)
                freq_psf = np.fft.fftfreq(len(line_psf), d=ps_psf)
                h_psf = len(freq_psf)//2
                mtf50_psf = find_mtf_freq(mtf_psf[:h_psf], freq_psf[:h_psf], 0.5)
                mtf10_psf = find_mtf_freq(mtf_psf[:h_psf], freq_psf[:h_psf], 0.1)

                dark_style()
                fig_pa, axes_pa = plt.subplots(1, 3, figsize=(12, 4), facecolor=DARK)
                fig_pa.patch.set_facecolor(DARK)
                axes_pa[0].imshow(patch_psf, cmap="hot"); axes_pa[0].set_title("PSF patch", color=MUTED, fontsize=9); axes_pa[0].axis("off")
                axes_pa[1].plot(line_psf, color=ACC, lw=1.5); axes_pa[1].set_title("LSF (center line)", color=MUTED, fontsize=9)
                axes_pa[1].set_facecolor(SURF); axes_pa[1].tick_params(colors=MUTED)
                axes_pa[2].plot(freq_psf[:h_psf], mtf_psf[:h_psf], color=ACC2, lw=1.5)
                axes_pa[2].axhline(0.5, color="red", ls="--", lw=0.8)
                axes_pa[2].axhline(0.1, color="#9c27b0", ls="--", lw=0.8)
                if np.isfinite(mtf50_psf): axes_pa[2].axvline(mtf50_psf, color="red", lw=0.6)
                if np.isfinite(mtf10_psf): axes_pa[2].axvline(mtf10_psf, color="#9c27b0", lw=0.6)
                axes_pa[2].set_title("MTF", color=MUTED, fontsize=9); axes_pa[2].set_ylim(0, 1.1)
                axes_pa[2].set_facecolor(SURF); axes_pa[2].tick_params(colors=MUTED)
                plt.tight_layout(pad=0.8)
                st.pyplot(fig_pa); plt.close(fig_pa)

                m_p1, m_p2 = st.columns(2)
                m_p1.metric("MTF50 (mm⁻¹)", f"{mtf50_psf:.4f}" if np.isfinite(mtf50_psf) else "N/A")
                m_p2.metric("MTF10 (mm⁻¹)", f"{mtf10_psf:.4f}" if np.isfinite(mtf10_psf) else "N/A")

                fwhm_psf_px = compute_fwhm(line_psf, 1.0)
                fwhm_psf_mm = fwhm_psf_px * ps_psf if np.isfinite(fwhm_psf_px) else float("nan")
                st.session_state["mean_psf"] = patch_psf
                st.session_state["psf_metrics"] = {
                    "slice": selected_slice_psf + 1,
                    "pixel_spacing_mm": round(float(ps_psf), 4),
                    "bead_cx": int(cx_psf), "bead_cy": int(cy_psf),
                    "patch_size": f"{patch_psf.shape[0]}×{patch_psf.shape[1]} px",
                    "FWHM_mm":   round(float(fwhm_psf_mm), 4) if np.isfinite(fwhm_psf_mm) else "N/A",
                    "MTF50_lpmm": round(float(mtf50_psf), 4) if np.isfinite(mtf50_psf) else "N/A",
                    "MTF10_lpmm": round(float(mtf10_psf), 4) if np.isfinite(mtf10_psf) else "N/A",
                }
                st.session_state.qc_status["psf"] = True
                slog("PSF saved → ready for lesion insertion", "ok")

        if st.session_state.get("psf_metrics"):
            m_psf = st.session_state["psf_metrics"]
            st.markdown("### PSF Metrics")
            st.dataframe(pd.DataFrame([{
                "Slice": m_psf["slice"], "Pixel spacing (mm)": m_psf["pixel_spacing_mm"],
                "Bead (cx,cy)": f"({m_psf['bead_cx']},{m_psf['bead_cy']})",
                "Patch size": m_psf["patch_size"],
                "FWHM (mm)": m_psf["FWHM_mm"],
                "MTF50 (lp/mm)": m_psf["MTF50_lpmm"],
                "MTF10 (lp/mm)": m_psf["MTF10_lpmm"],
            }]), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════
    # TAB 4: SIMULATION
    # ═══════════════════════════════════════════════
    with _tab_sim:
        st.markdown("### Simulated Lesion (Image-domain)")

        sl_sim        = st.session_state.slices[st.session_state.current]
        background    = sl_sim["hu_orig"]
        pixel_spacing = sl_sim["pixel_spacing"]
        psf_kernel    = st.session_state.get("mean_psf", None)
        n_total_sl    = len(st.session_state.slices)
        h_sim, w_sim  = background.shape

        if psf_kernel is None:
            st.warning("⚠ ยังไม่มี PSF — กรุณา Run PSF Analysis ใน tab PSF Bead ก่อน")
        else:
            psf_m_sim = st.session_state.get("psf_metrics", {})
            if psf_m_sim:
                st.markdown(
                    f'<div class="infobox">PSF: Slice <b>{psf_m_sim["slice"]}</b> | '
                    f'FWHM <b>{psf_m_sim["FWHM_mm"]} mm</b> | '
                    f'MTF50 <b>{psf_m_sim["MTF50_lpmm"]} lp/mm</b></div>',
                    unsafe_allow_html=True
                )

        from scipy.ndimage import center_of_mass as _com
        _body_sim = background > -500
        if np.any(_body_sim):
            _cy_p, _cx_p = _com(_body_sim)
            if not (np.isfinite(_cx_p) and np.isfinite(_cy_p)):
                _cx_p, _cy_p = w_sim / 2.0, h_sim / 2.0
        else:
            _cx_p, _cy_p = w_sim / 2.0, h_sim / 2.0
        ph_cx, ph_cy     = int(round(_cx_p)), int(round(_cy_p))
        _yy_sim, _xx_sim = np.ogrid[:h_sim, :w_sim]
        _dist_sim        = np.sqrt((_xx_sim - ph_cx)**2 + (_yy_sim - ph_cy)**2)
        if np.any(_body_sim):
            ph_r = float(np.percentile(_dist_sim[_body_sim], 90))
        else:
            ph_r = float(min(h_sim, w_sim)) / 3.0

        insert_mode = st.radio(
            "Insertion mode",
            ["Single lesion (manual)", "ACR pattern (4/5/6 mm, +6 HU)"],
            horizontal=True, key="insert_mode_radio"
        )

        st.markdown("##### เลือก Slice ที่ต้องการ Insert")
        slice_apply = st.radio(
            "Apply to",
            ["Selected slice only", "Range of slices", "All slices"],
            horizontal=True, key="slice_apply_mode"
        )

        if slice_apply == "Selected slice only":
            t_sel = st.selectbox("Slice", list(range(n_total_sl)),
                                 index=st.session_state.current,
                                 format_func=lambda i: f"Slice {i+1}",
                                 key="target_slice_sel")
            target_slices = [t_sel]
        elif slice_apply == "Range of slices":
            c1r_sim, c2r_sim = st.columns(2)
            r_s_sim = c1r_sim.number_input("Start", 1, n_total_sl, 1, key="rng_s") - 1
            r_e_sim = c2r_sim.number_input("End",   1, n_total_sl, n_total_sl, key="rng_e") - 1
            target_slices = list(range(int(r_s_sim), int(r_e_sim)+1))
            st.caption(f"{len(target_slices)} slices selected")
        else:
            target_slices = list(range(n_total_sl))
            st.caption(f"All {n_total_sl} slices")

        st.markdown("---")

        if insert_mode == "Single lesion (manual)":
            c1s, c2s, c3s = st.columns(3)
            diam_mm_sim     = c1s.slider("Diameter (mm)", 2, 30, 10, key="s_diam")
            contrast_hu_sim = c2s.slider("Contrast (HU)", -500, 100, -10, key="s_con")
            add_noise_sim   = c3s.checkbox("Add noise", True, key="s_noise")
            cx_les = st.slider("Center X", 0, w_sim-1, w_sim//2, key="s_cx")
            cy_les = st.slider("Center Y", 0, h_sim-1, h_sim//2, key="s_cy")
            lesion_list_sim = [{"cx": cx_les, "cy": cy_les,
                                "diam_mm": diam_mm_sim, "contrast_hu": contrast_hu_sim, "label": "lesion"}]
        else:
            c1a_sim, c2a_sim = st.columns(2)
            add_noise_sim    = c1a_sim.checkbox("Add noise", True, key="acr_noise")
            acr_contrast_sim = c2a_sim.number_input("Contrast (HU)", -500, 100, 6, key="acr_con")
            ring_defs_sim = [
                {"diam_mm": 6, "ring_frac": 0.30},
                {"diam_mm": 5, "ring_frac": 0.40},
                {"diam_mm": 4, "ring_frac": 0.50},
            ]
            angles_4_sim   = [-np.pi/2, 0, np.pi/2, np.pi]
            clock_lbl_sim  = ["12","3","6","9"]
            lesion_list_sim = []
            for rdef_sim in ring_defs_sim:
                ring_r_sim = ph_r * rdef_sim["ring_frac"]
                for ang_sim, clk_sim in zip(angles_4_sim, clock_lbl_sim):
                    lesion_list_sim.append({
                        "cx": int(ph_cx + ring_r_sim * np.cos(ang_sim)),
                        "cy": int(ph_cy + ring_r_sim * np.sin(ang_sim)),
                        "diam_mm": rdef_sim["diam_mm"],
                        "contrast_hu": int(acr_contrast_sim),
                        "label": f'{rdef_sim["diam_mm"]}mm-{clk_sim}',
                    })
            diam_mm_sim = 5; contrast_hu_sim = int(acr_contrast_sim)

        dark_style()
        fig_prev_sim, ax_prev_sim = plt.subplots(figsize=(5, 5), facecolor=DARK)
        ax_prev_sim.imshow(wl_ww(background, st.session_state.wl, st.session_state.ww), cmap="gray")
        color_map_sim = {6: "cyan", 5: "lime", 4: "orange"}
        if insert_mode != "Single lesion (manual)":
            drawn_r = set()
            for rd in ring_defs_sim:
                if rd["diam_mm"] not in drawn_r:
                    ax_prev_sim.add_patch(plt.Circle((ph_cx, ph_cy), ph_r*rd["ring_frac"],
                                                     fill=False, edgecolor=color_map_sim[rd["diam_mm"]],
                                                     lw=0.8, ls="--", alpha=0.4))
                    drawn_r.add(rd["diam_mm"])
        for les_sim in lesion_list_sim:
            r_p_sim = (les_sim["diam_mm"]/2) / pixel_spacing
            ax_prev_sim.add_patch(plt.Circle((les_sim["cx"], les_sim["cy"]), r_p_sim,
                                             fill=False, edgecolor=color_map_sim.get(les_sim["diam_mm"],"white"), lw=1.5))
        ax_prev_sim.set_title("Preview", fontsize=9, color=MUTED)
        ax_prev_sim.axis("off")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_prev_sim)
        plt.close(fig_prev_sim)

        if psf_kernel is not None and st.button("▶ Insert Lesion(s)", use_container_width=True, key="btn_insert"):
            nps_2d_in_sim = st.session_state.get("nps_cache", None)
            for sl_clr in st.session_state.slices:
                sl_clr.pop("hu_sim", None)
            progress_sim = st.progress(0, text="Processing...")
            for t_idx_sim, sl_idx_sim in enumerate(target_slices):
                sl_i_sim  = st.session_state.slices[sl_idx_sim]
                base_hu_sim = sl_i_sim["hu_orig"].copy()
                sl_thick_sim = float(sl_i_sim.get("slice_thickness", sl_i_sim["pixel_spacing"]))
                ps_i_sim = sl_i_sim["pixel_spacing"]
                for les_sim in lesion_list_sim:
                    base_hu_sim, _ = insert_lesion_image_domain(
                        background_hu=base_hu_sim, cx=les_sim["cx"], cy=les_sim["cy"],
                        diam_mm=les_sim["diam_mm"], contrast_hu=les_sim["contrast_hu"],
                        pixel_spacing_mm=ps_i_sim, psf=psf_kernel,
                        slice_thickness_mm=sl_thick_sim, add_noise=add_noise_sim,
                        noise_sd=None, nps_2d=nps_2d_in_sim, seed=42+sl_idx_sim
                    )
                sl_i_sim["hu_sim"] = base_hu_sim
                progress_sim.progress((t_idx_sim+1)/len(target_slices),
                                      text=f"Slice {sl_idx_sim+1} ({t_idx_sim+1}/{len(target_slices)})")
            progress_sim.empty()
            st.session_state["sim_ready"]         = True
            st.session_state["sim_lesions"]       = lesion_list_sim
            st.session_state["sim_target_slices"] = target_slices
            st.session_state["sim_viewer_idx"]    = target_slices[0]
            slog(f"Inserted {len(lesion_list_sim)} lesion(s) into {len(target_slices)} slices", "ok")
            st.rerun()

        if st.session_state.get("sim_ready"):
            st.markdown("### Simulated Series Viewer")
            target_sl_sim = st.session_state.get("sim_target_slices", list(range(n_total_sl)))
            sim_idx = st.selectbox(
                "Sim Slice",
                options=target_sl_sim,
                format_func=lambda i: f"Slice {i+1}",
                key="sim_viewer_select"
            )
            sl_view_sim = st.session_state.slices[sim_idx]
            hu_view_sim = sl_view_sim.get("hu_sim", sl_view_sim["hu_orig"])
            dark_style()
            fig_sv, ax_sv = plt.subplots(figsize=(6, 6), facecolor=DARK)
            ax_sv.imshow(wl_ww(hu_view_sim, st.session_state.wl, st.session_state.ww), cmap="gray")
            ax_sv.set_title(f"Slice {sim_idx+1} / {n_total_sl}", fontsize=10, color=MUTED)
            ax_sv.axis("off")
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_sv)
            plt.close(fig_sv)


# ── End of advqc/simulation page ──
if _page in ("advqc", "simulation"):
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# CHO PAGE
# ──────────────────────────────────────────────────────────────────────────
if _page == "cho":
    _topbar("CHO d′ Observer", "🧪")

    _cho_tabs = st.tabs(["📈 MTF / TTF", "🧪 CHO d′"])
    _tab_mtf, _tab_cho = _cho_tabs

    with _tab_mtf:
        st.markdown('<div class="secLabel">MTF / TTF Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="infobox">
        เลือก lesion ROI ใน Viewer → ระบบจะคำนวณ ESF → LSF → FFT → MTF<br>
        ต้อง Insert lesions ก่อนใน Advanced QC / Simulation
        </div>""", unsafe_allow_html=True)

        _sl_mtf = st.session_state.slices[st.session_state.current]
        _les_mtf = _sl_mtf.get("lesions", [])
        if not _les_mtf:
            st.markdown('<div class="warnbox">⚠ ยังไม่มี lesion — ไปที่ Advanced QC เพื่อ Insert ก่อน</div>',
                        unsafe_allow_html=True)
        else:
            _les_opts = {f"Ø{l['diam_mm']}mm {l['contrast']:+d}HU @ {l['position']}": l
                         for l in _les_mtf}
            _sel_les = st.selectbox("เลือก lesion", list(_les_opts.keys()))
            if st.button("▶ Run MTF/TTF", use_container_width=True, type="primary"):
                with st.spinner("Computing MTF…"):
                    _res_mtf = run_mtf_analysis(_sl_mtf, _les_opts[_sel_les])
                st.session_state.mtf_result = _res_mtf
                slog(f"MTF done: f50={_res_mtf['f50']}", "ok")
            if st.session_state.get("mtf_result"):
                st.image(plot_mtf(st.session_state.mtf_result), use_container_width=True)
                _r = st.session_state.mtf_result
                c1m, c2m = st.columns(2)
                c1m.metric("f₅₀ (mm⁻¹)", f"{_r['f50']:.4f}" if _r['f50'] else "—")
                c2m.metric("f₁₀ (mm⁻¹)", f"{_r['f10']:.4f}" if _r['f10'] else "—")

    with _tab_cho:
        st.markdown('<div class="secLabel">Channelized Hotelling Observer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="mathbox">
        <b>d′</b> = √[ Δv̄ᵀ · K_v⁻¹ · Δv̄ ]<br>
        <b>AUC</b> = Φ(d′/√2)<br>
        Gabor channels: 4 freq × 2 orient × 2 phase = 16 channels
        </div>""", unsafe_allow_html=True)

        _sl_cho = st.session_state.slices[st.session_state.current]
        if not _sl_cho.get("lesions"):
            st.markdown('<div class="warnbox">⚠ ยังไม่มี lesion — ไปที่ Advanced QC เพื่อ Insert ก่อน</div>',
                        unsafe_allow_html=True)
        else:
            if st.button("▶ Run CHO d′", use_container_width=True, type="primary"):
                with st.spinner("Running CHO…"):
                    _cho_res = run_cho_from_clock(_sl_cho)
                if _cho_res:
                    st.session_state.cho_result = _cho_res
                    slog("CHO done", "ok")
                else:
                    st.warning("ต้องการ lesion อย่างน้อย 5 ตำแหน่ง")
            if st.session_state.get("cho_result"):
                _cho = st.session_state.cho_result
                st.image(plot_cho(_cho, _sl_cho), use_container_width=True)
                _rows_cho = [{"Position": r["position"], "Ø (mm)": r["diam_mm"],
                               "Contrast": r["contrast"], "d′": f"{r['dprime']:.3f}",
                               "AUC": f"{r['auc']:.3f}"} for r in _cho["results"]]
                st.dataframe(pd.DataFrame(_rows_cho), use_container_width=True, hide_index=True)

    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# REPORT PAGE
# ──────────────────────────────────────────────────────────────────────────
if _page == "report":
    _topbar("Report & Log", "📋")

    _r1, _r2 = st.columns([3, 2])

    with _r1:
        st.markdown('<div class="secLabel">QC Workflow Status</div>', unsafe_allow_html=True)
        _qc = st.session_state.qc_status
        _geo_d = st.session_state.basic_geometry_result is not None
        _sq_d  = st.session_state.basic_square_result  is not None
        _lin_d = st.session_state.basic_linearity_result is not None

        def _si2(done, ok=True):
            if not done: return "⚪"
            return "🟢" if ok else "🔴"

        status_rows = [
            ("Geometry QC",      _geo_d, (st.session_state.basic_geometry_result or {}).get("pass", False)),
            ("Square 50mm QC",   _sq_d,  (st.session_state.basic_square_result   or {}).get("pass", False)),
            ("CT Linearity",     _lin_d, (st.session_state.basic_linearity_result or {}).get("pass", False)),
            ("Uniformity",       _qc["uniformity"], _qc["uniformity"]),
            ("NPS",              _qc["nps"],        _qc["nps"]),
            ("Partial Volume",   _qc["pve"],        _qc["pve"]),
            ("PSF",              _qc["psf"],        _qc["psf"]),
        ]
        st.markdown("""
        <div style="background:#0a0f18;border:1px solid #1a2a3a;border-radius:10px;padding:14px 18px;">
        """ + "".join(
            f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
            f'border-bottom:1px solid #111b2a;font-size:12px;">'
            f'<span style="font-size:15px">{_si2(done,ok)}</span>'
            f'<span style="color:{"#70ffc0" if ok and done else "#4a7090" if not done else "#ff8a80"}">'
            f'{name}</span></div>'
            for name, done, ok in status_rows
        ) + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="secLabel" style="margin-top:16px;">Export</div>', unsafe_allow_html=True)
        _c1r, _c2r = st.columns(2)
        with _c1r:
            _csv = report_csv_bytes()
            st.download_button("⬇ CSV Report", data=_csv,
                file_name=f"ctqc_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)
        with _c2r:
            _pdf = report_pdf_bytes()
            st.download_button("⬇ PDF Report", data=_pdf,
                file_name=f"ctqc_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True)

    with _r2:
        st.markdown('<div class="secLabel">Activity Log</div>', unsafe_allow_html=True)
        _log = st.session_state.log[-40:][::-1]
        if _log:
            st.markdown(
                '<div style="background:#080b10;border:1px solid #1a2a3a;border-radius:8px;'
                'padding:10px 14px;font-family:monospace;font-size:10px;color:#4a8090;'
                'max-height:400px;overflow-y:auto;">'
                + "<br>".join(_log)
                + "</div>", unsafe_allow_html=True)
        else:
            st.caption("ยังไม่มี log")

    st.stop()
