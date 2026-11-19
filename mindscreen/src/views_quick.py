"""Quick analyser (no account, nothing stored): upload a daily file, get an instant look. Kept from v2 of the project."""
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Blueprint, abort, jsonify, render_template, request, send_from_directory

import footprint as fp
import predict_service as ps
from auth import GENDERS, LEVELS, csrf_exempt
from features import RAW_CAT

bp = Blueprint("quick", __name__)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = OrderedDict()          # in-memory only
DEMOS = {
    "balanced": ("Balanced routine", "Steady sleep, moderate screen time, active and social"),
    "heavy_user": ("Heavy night-time user", "Long screen time, short sleep, low activity"),
    "withdrawing": ("Gradual withdrawal", "Fine for 3 weeks, then the last 7 days change sharply"),
}
MODEL_KEYS = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Most_Used_Platform"]
TIPS = {"Daily social-media time": "Set app timers and keep one phone-free block each day.",
        "Sleep per night": "Fix a consistent bedtime and stop scrolling 30 minutes before sleep."}


def _profile(form):
    P = ps.predictor()
    prof = {"Age": form.get("Age", 21), "Gender": form.get("Gender"), "Academic_Level": form.get("Academic_Level")}
    return prof


def _analyse(raw_df, profile, source):
    P = ps.predictor()
    cats = P.schema["categories"]["Most_Used_Platform"]
    d = fp.parse(raw_df)
    s = fp.summarise(d, cats)
    run = lambda summ: P.predict({**profile, **{k: summ[k] for k in MODEL_KEYS}})
    overall = run(s)
    recent = run(fp.summarise(fp.recent_frame(d), cats)) if len(d) >= 14 else None
    a = {"id": uuid.uuid4().hex[:8].upper(), "when": datetime.now().strftime("%d %b %Y, %H:%M"), "source": source,
         "summary": s, "overall": overall, "recent": recent, "headline": recent or overall,
         "drift": fp.drift(d), "chart": fp.chart(d), "profile": profile}
    RESULTS[a["id"]] = a
    while len(RESULTS) > 50:
        RESULTS.popitem(last=False)
    return a


@bp.route("/quick")
def index():
    return render_template("quick/index.html", s=ps.predictor().schema, demos=DEMOS, genders=GENDERS, levels=LEVELS, error=None)


@bp.route("/quick/analyze", methods=["POST"])
def analyze():
    P = ps.predictor()
    prof = _profile(request.form)
    demo = request.form.get("demo")
    try:
        if demo in DEMOS:
            raw, source = pd.read_csv(ROOT / "data" / "demo" / f"{demo}.csv"), f"Simulated demo: {DEMOS[demo][0]}"
        else:
            f = request.files.get("file")
            if not f or not f.filename:
                raise fp.FootprintError("Please choose a file to upload, or try one of the demo students.")
            raw = pd.read_json(f) if f.filename.lower().endswith(".json") else pd.read_csv(f)
            source = f"Uploaded file: {Path(f.filename).name}"
        a = _analyse(raw, prof, source)
    except (fp.FootprintError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
        return render_template("quick/index.html", s=P.schema, demos=DEMOS, genders=GENDERS, levels=LEVELS, error=str(e)), 400
    return render_template("quick/result.html", a=a, tips=TIPS, m=P.metrics)


@bp.route("/quick/report/<rid>")
def report(rid):
    a = RESULTS.get(rid.upper())
    if not a:
        abort(404, "This report has expired (kept in memory only). Please run the analysis again.")
    return render_template("quick/report.html", a=a, s=ps.predictor().schema)


@bp.route("/quick/demo/<name>.csv")
def demo_file(name):
    if name not in DEMOS:
        abort(404)
    return send_from_directory(ROOT / "data" / "demo", f"{name}.csv", as_attachment=True)


@bp.route("/api/predict", methods=["POST"])
@csrf_exempt
def api_predict():
    """Stateless JSON endpoint (nothing stored): {"daily":[...], "profile":{...}} or summary fields."""
    if not request.is_json:
        return jsonify({"error": "Send JSON with Content-Type: application/json"}), 415
    P = ps.predictor()
    body = request.get_json(silent=True) or {}
    if "daily" in body:
        try:
            a = _analyse(pd.DataFrame(body["daily"]), body.get("profile", {}), "API")
        except fp.FootprintError as e:
            return jsonify({"error": str(e)}), 400
        h, dr = a["headline"], a["drift"]
        return jsonify({"risk_pct": h["risk_pct"], "tier": h["tier"], "overall_risk_pct": a["overall"]["risk_pct"],
                        "recent_risk_pct": a["recent"]["risk_pct"] if a["recent"] else None,
                        "change_level": dr["level"] if dr else None,
                        "changed_signals": [r["label"] for r in dr["rows"] if r["flag"]] if dr else []})
    r = P.predict(body)
    return jsonify({k: r[k] for k in ("risk_pct", "tier", "consensus", "agreement", "whatif", "model_used")})
