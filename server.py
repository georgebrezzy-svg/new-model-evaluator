from werkzeug.exceptions import BadRequest
import json

def _to_float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    # accept commas and spaces: "177", "1,77e2" (will become 177)
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

@app.post("/evaluate")
def evaluate():
    try:
        # --- be very tolerant about JSON/body formats ---
        data = request.get_json(silent=True)
        if not data:
            # fallbacks: raw body as JSON, or form-encoded
            raw = (request.data or b"").decode("utf-8", errors="ignore")
            if raw:
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
        if not data and request.form:
            data = request.form.to_dict(flat=True)

        # default structure to avoid KeyErrors
        data = data or {}

        # extract fields with gentle coercion
        photos = data.get("photos") or []
        if isinstance(photos, str):
            # accept single string or comma-separated
            photos = [p.strip() for p in photos.split(",") if p.strip()]

        gender = (data.get("gender") or "").strip()
        # normalize gender casing
        if gender.lower() in ("male", "m"):
            gender = "Male"
        elif gender.lower() in ("female", "f"):
            gender = "Female"

        height_cm = _to_float_or_none(data.get("height_cm"))
        age = _to_float_or_none(data.get("age"))
        measurements = (data.get("measurements") or "").strip()

        # --- validate nicely (return clear reasons instead of 400 html) ---
        if not isinstance(photos, list) or len(photos) < 1:
            return jsonify({"error": "validation", "detail": "photos must be a non-empty list of URLs"}), 422
        if any((not p) or (not isinstance(p, str)) for p in photos):
            return jsonify({"error": "validation", "detail": "photos list contains an empty or non-string URL"}), 422

        if gender not in ("Male", "Female"):
            return jsonify({"error": "validation", "detail": "gender must be 'Male' or 'Female'"}), 422

        if data.get("height_cm") is not None and height_cm is None:
            return jsonify({"error": "validation", "detail": "height_cm must be numeric (e.g., 177)"}), 422
        if data.get("age") is not None and age is None:
            return jsonify({"error": "validation", "detail": "age must be numeric (e.g., 19)"}), 422

        # --- your existing scoring logic ---
        best_sim, details = score_photos(photos[:5])
        parsed = parse_measurements(measurements)
        decision, reason = decide(gender, height_cm, age, best_sim, parsed)

        # flatten details to a single string for Bubble mapping
        details_text = "; ".join(
            f"{d.get('url','')} — {d.get('desc','')} (sim={d.get('best_similarity','')})"
            for d in details
        )

        return jsonify({
            "decision": decision,
            "confidence": round(best_sim, 3),
            "reason": reason,                  # always present
            "details": details,                # list of objects
            "details_text": details_text,      # flattened string
            "parsed_measurements_cm": parsed
        })

    except BadRequest:
        # Flask couldn't parse the body
        return jsonify({"error": "invalid_json", "detail": "Send JSON with Content-Type: application/json"}), 400
    except Exception as e:
        # Never return HTML; always JSON the error
        return jsonify({"error": "server_exception", "detail": str(e)}), 500
