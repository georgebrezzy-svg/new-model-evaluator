# ---------- server.py (minimal working version) ----------

from flask import Flask, request, jsonify
from flask_cors import CORS
import json as _json

app = Flask(__name__)
CORS(app)  # allow Bubble to call this API

# Health check
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True})

# Minimal evaluator so Bubble can work end-to-end
@app.route("/evaluate", methods=["POST"])
def evaluate():
    # Read JSON safely (works even if Bubble sends empty strings)
    data = request.get_json(silent=True)
    if not data:
        raw = (request.data or b"").decode("utf-8", errors="ignore")
        try:
            data = _json.loads(raw) if raw else {}
        except Exception:
            data = {}
    if not data and request.form:
        data = request.form.to_dict(flat=True)
    data = data or {}

    photos = data.get("photos") or []
    if isinstance(photos, str):
        photos = [p.strip() for p in photos.split(",") if p.strip()]

    gender = (str(data.get("gender") or "")).strip()
    height_cm = str(data.get("height_cm") or "").strip()
    age = str(data.get("age") or "").strip()
    measurements = (str(data.get("measurements") or "")).strip()

    # Build a simple details_text so you can map it in Bubble
    details_text = "; ".join(photos[:2]) if photos else ""

    # Return a valid, predictable shape (you can wire this in Bubble now)
    return jsonify({
        "decision": "review",
        "confidence": 0.75,
        "reason": "Endpoint OK; mock evaluation response.",
        "details": [
            {"url": p, "desc": "provided", "best_similarity": 0.0}
            for p in photos[:2]
        ],
        "details_text": details_text,
        "parsed_measurements_cm": None
    }), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

# ---------- end ----------
