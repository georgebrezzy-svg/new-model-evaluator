import os, json, math
from flask import Flask, request, jsonify
import numpy as np
import requests

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# --- Load memory (embeddings.json MUST be present in the repo) ---
with open("embeddings.json", "r", encoding="utf-8") as f:
    MEM = json.load(f)
MEM_MAT = np.array([m["embedding"] for m in MEM], dtype=np.float32)
MEM_MAT = MEM_MAT / (np.linalg.norm(MEM_MAT, axis=1, keepdims=True) + 1e-12)

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

def score_photos(photo_urls):
    details = []
    best_overall = 0.0
    for url in photo_urls:
        desc = describe_image(url)
        vec = embed_text(desc)
        sims = MEM_MAT @ vec
        best = float(np.max(sims))
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
    # optional measurement guidance (not hard-reject unless you want it)
    # thresholds could be adjusted later
    if parsed:
        # Example soft hints, not hard rules:
        pass
    # similarity thresholds
    if best_sim >= 0.78:
        return "SELECTED", ""
    if best_sim >= 0.60:
        return "NEEDS_REVIEW", ""
    return "REJECTED", "Low similarity to preferred look"

@app.get("/ping")
def ping():
    return jsonify({"ok": True})

@app.post("/evaluate")
def evaluate():
    try:
        data = request.get_json(force=True)
        photos = data.get("photos") or []
        gender = data.get("gender")
        height_cm = data.get("height_cm")
        age = data.get("age")
        measurements = data.get("measurements", "")

        # validate
        if not isinstance(photos, list) or len(photos) < 1:
            return jsonify({"error": "photos must be a non-empty list"}), 400
        if any((not p) or (not isinstance(p, str)) for p in photos):
            return jsonify({"error": "photos list contains empty url"}), 400
        if gender not in ("Male", "Female"):
            return jsonify({"error": "gender must be 'Male' or 'Female'"}), 400
        try:
            height_cm = float(height_cm) if height_cm is not None else None
            age = float(age) if age is not None else None
        except Exception:
            return jsonify({"error": "height_cm/age must be numeric"}), 400

        best_sim, details = score_photos(photos[:5])  # first 1–5 photos
        parsed = parse_measurements(measurements)
        decision, reason = decide(gender, height_cm, age, best_sim, parsed)

        return jsonify({
            "decision": decision,
            "confidence": round(best_sim, 3),
            "reason": reason,  # always present so Bubble can map it
            "details": details,
            "parsed_measurements_cm": parsed
        })
    except requests.HTTPError as e:
        return jsonify({"error": "OpenAI error", "detail": str(e), "body": getattr(e.response, 'text', '')}), 502
    except Exception as e:
        return jsonify({"error": "server exception", "detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
