# ---------- server.py (full file) ----------

import os, json, math
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app)  # allow requests from Bubble

# ----- OpenAI setup -----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# ----- Load memory (embeddings.json optional & safe) -----
try:
    with open("embeddings.json", "r", encoding="utf-8") as f:
        MEM = json.load(f)
    MEM_MAT = np.array([m["embedding"] for m in MEM], dtype=np.float32)
    MEM_MAT = MEM_MAT / (np.linalg.norm(MEM_MAT, axis=1, keepdims=True) + 1e-12)
except Exception:
    # If the file is missing or malformed, continue without memory
    MEM = []
    MEM_MAT = None


# ----- Helpers -----
def describe_image(url: str) -> str:
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": (
                "Describe the subject for fashion modeling suitability. "
                "Max 2 lines. Avoid sensitive attributes; focus on symmetry, bone structure, "
                "proportions, skin clarity, posture, and editorial/commercial vibe."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this applicant image."},
                {"type": "image_url", "image_url": {"url": url}}
            ]}
        ]
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=body, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def embed_text(text: str) -> np.ndarray:
    body = {"model": "text-embedding-3-large", "input": text}
    r = requests.post("https://api.openai.com/v1/embeddings", headers=HEADERS, json=body, timeout=60)
    r.raise_for_status()
    v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

defdef score_photos(photo_urls):
    details = []
    best_overall = 0.0
    for url in photo_urls:
        desc = describe_image(url)
        best = 0.0
        if MEM_MAT is not None:
            # Only compute similarity if memory is available
            vec = embed_text(desc)
            sims = MEM_MAT @ vec
            best = float(np.max(sims))
        # If no memory, best stays 0.0 (the app still works)
        best_overall = max(best_overall, best)
        details.append({"url": url, "desc": desc, "best_similarity": round(best, 3)})
    return best_overall, details


def parse_measurements(meas_str):
    # very tolerant parser: "84-60-89", "84/60/89", "32-24-35 in"
    import re
    if not meas_str or not isinstance(meas_str, str): return None
    s = meas_str.strip().lower()
    unit = "cm"
    if " in" in s or s.endswith("in"):
        unit = "in"
    s_clean = re.sub(r"[^0-9\-/ ,.]", " ", s)
    parts = [p for p in re.split(r"[-/ ,]+", s_clean) if p]
    if len(parts) < 3: return None
    vals = [float(parts[0]), float(parts[1]), float(parts[2])]
    if unit == "in":
        vals = [round(v*2.54, 1) for v in vals]
    return {"bust_or_chest": vals[0], "waist": vals[1], "hips": vals[2]}

def decide(gender, height_cm, age, best_sim, parsed):
    # hard rules
    if age is not None and (age < 16 or age > 23):
        return "REJECTED", "Age outside 16–23"
    if gender == "Male" and (height_cm is not None) and not (183 <= height_cm <= 190):
        return "REJECTED", "Male height outside 183–190"
    if gender == "Female" and (height_cm is not None) and not (175 <= height_cm <= 180):
        return "REJECTED", "Female height outside 175–180"
    # similarity thresholds
    if best_sim >= 0.78:
        return "SELECTED", ""
    if best_sim >= 0.60:
        return "NEEDS_REVIEW", ""
    return "REJECTED", "Low similarity to preferred look"

def _to_float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

# -----
