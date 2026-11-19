"""Unified prediction pipeline. ONE model family (LR / RF / GB / SVM trained on the public survey dataset),
fed by features from several categories. Categories the validated model was not trained on are shown to the
user but never silently fed to it."""
import json
from datetime import datetime, timezone
from pathlib import Path

from categories import CATEGORIES
from db import q, run
import explain
import features_unified as fu
from predictor import Predictor

ROOT = Path(__file__).resolve().parent.parent
_P = None

MODEL_FEATURE_SOURCES = {          # model input -> category that supplies it
    "Avg_Daily_Usage_Hours": "screen", "Sleep_Hours_Per_Night": "physical", "Most_Used_Platform": "screen"}
USED_CATEGORIES = ["screen", "physical"]
NOT_USED_CATEGORIES = ["behaviour", "social", "academic"]


def predictor() -> Predictor:
    global _P
    if _P is None:
        _P = Predictor()
    return _P


def model_version_name() -> str:
    p = predictor()
    stamp = datetime.fromtimestamp((ROOT / "models" / "schema.json").stat().st_mtime).strftime("%Y%m%d")
    return f"passive-{p.best.split()[0].lower()}-v1-{stamp}"


def sync_model_version(db_run=run):
    """Register the active model + its evaluation (from training on the public dataset) in model_versions."""
    p = predictor()
    name = model_version_name()
    metrics = {"models": p.metrics, "primary": p.best, "n_rows": p.schema["n_rows"], "test_size": p.schema["test_size"],
               "evaluation_data": "held-out 20% of the public student survey dataset (not platform data)"}
    sub = ROOT / "models" / "subgroups.json"
    if sub.exists():
        metrics["subgroups"] = json.loads(sub.read_text())
    db_run("UPDATE model_versions SET is_active=0")
    db_run("INSERT INTO model_versions(name,created_at,kind,metrics_json,notes,is_active) VALUES(?,?,?,?,?,1) "
           "ON CONFLICT(name) DO UPDATE SET metrics_json=excluded.metrics_json, is_active=1",
           (name, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), "sklearn-pipeline",
            json.dumps(metrics), "Trained on public survey data; not clinically validated."))
    return name


def sufficiency(n_days: int, notes: list) -> str:
    level = "Limited" if n_days < 14 else ("Moderate" if n_days < 28 else "Good")
    order = ["Limited", "Moderate", "Good"]
    if any(n.startswith("approximation") for n in notes) and level != "Limited":
        level = order[order.index(level) - 1]
    return level


def prepare_inputs(user, window_days=30) -> dict:
    """Returns {available, reasons, inputs, notes, n_days, window}. Never invents values."""
    pid = user["pseudo_id"]
    reasons, notes = [], []
    start, end = fu.window_bounds(pid, USED_CATEGORIES, window_days)
    screen = fu.series(pid, "screen", start, end) if start else None
    phys = fu.series(pid, "physical", start, end) if start else None
    if screen is None or screen.empty:
        reasons.append("Screen-time data has not been submitted.")
    if phys is None or phys.empty or phys["sleep_h"].notna().sum() == 0:
        reasons.append("Physical-wellness data with sleep duration has not been submitted.")
    if reasons:
        return {"available": False, "reasons": reasons, "inputs": None, "notes": notes, "n_days": 0, "window": (start, end)}

    usage_col = "social_h" if screen["social_h"].notna().sum() >= 7 else "total_screen_h"
    if usage_col == "total_screen_h":
        notes.append("approximation: social-media hours are missing, so total screen time was used. The model was trained on social-media hours, so this may overstate usage.")
    usage = screen[usage_col].dropna()
    sleep = phys["sleep_h"].dropna()
    n_days = int(min(len(usage), len(sleep)))
    if n_days < 7:
        reasons.append(f"At least 7 days of both screen-time and sleep data are needed in the window (found {len(usage)} and {len(sleep)}).")
        return {"available": False, "reasons": reasons, "inputs": None, "notes": notes, "n_days": n_days, "window": (start, end)}

    sch = predictor().schema
    prof_missing = [k for k, v in (("age", user["age"]), ("gender", user["gender"]), ("academic level", user["academic_level"])) if not v]
    if prof_missing:
        notes.append("profile defaults used for: " + ", ".join(prof_missing) + " (typical value of the training data). Add them in Settings for a better match.")
    app = screen["top_app"].dropna().mode().iloc[0] if screen["top_app"].notna().any() else "Other"
    inputs = {"Age": user["age"] or sch["defaults"]["Age"], "Gender": user["gender"] or sch["defaults_cat"]["Gender"],
              "Academic_Level": user["academic_level"] or sch["defaults_cat"]["Academic_Level"], "Most_Used_Platform": app,
              "Avg_Daily_Usage_Hours": float(usage.mean()), "Sleep_Hours_Per_Night": float(sleep.mean())}
    return {"available": True, "reasons": [], "inputs": inputs, "notes": notes, "n_days": n_days, "window": (start, end)}


def run_prediction(user, window_days=30, store=True) -> dict:
    prep = prepare_inputs(user, window_days)
    status = fu.category_status(user["pseudo_id"])
    comp = fu.overall_completeness(status)
    if not prep["available"]:
        return {"available": False, "reasons": prep["reasons"], "completeness": comp,
                "message": "A prediction needs validated Screen-time and Physical-wellness data. Add them on the Data Collection page."}
    P = predictor()
    res = P.predict(prep["inputs"])
    expl = explain.local_explanation(P.models[P.best], res["inputs"])
    version = model_version_name()
    suff = sufficiency(prep["n_days"], prep["notes"])
    out = {
        "available": True, "ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), "model_version": version, "model_name": P.best,
        "probability": res["risk_pct"], "tier": res["tier"], "colour": res["colour"], "votes": res["votes"],
        "agreement": res["agreement"], "consensus": res["consensus"], "inputs": res["inputs"], "whatif": res["whatif"],
        "n_days": prep["n_days"], "window": prep["window"], "sufficiency": suff, "notes": prep["notes"],
        "categories_used": USED_CATEGORIES,
        "categories_collected_not_used": [k for k in NOT_USED_CATEGORIES if status[k]["days"] > 0],
        "categories_missing": [k for k in CATEGORIES if status[k]["days"] == 0],
        "completeness": comp, "explanation": expl, "is_demo": bool(user["is_demo"]),
        "mode": "demo-data (real model on fictional records)" if user["is_demo"] else "real",
    }
    if store:
        cur = run("INSERT INTO predictions(pseudo_id,ts,model_version,mode,probability,tier,window_days,n_days,features_json,explanation_json,"
                  "categories_used,completeness,sufficiency,is_demo) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (user["pseudo_id"], out["ts"], version, out["mode"], out["probability"], out["tier"], window_days, out["n_days"],
                   json.dumps({k: out[k] for k in ("inputs", "votes", "agreement", "consensus", "whatif", "notes", "window", "model_name", "colour",
                                                   "categories_used", "categories_collected_not_used", "categories_missing")}),
                   json.dumps(expl), ",".join(USED_CATEGORIES), comp, suff, int(bool(user["is_demo"]))))
        out["id"] = cur.lastrowid
    return out


def latest_prediction(pseudo_id):
    r = q("SELECT * FROM predictions WHERE pseudo_id=? ORDER BY id DESC LIMIT 1", (pseudo_id,), one=True)
    return dict(r) if r else None


def load_prediction(row) -> dict:
    """Rebuild the full result dict from a stored predictions row."""
    d = json.loads(row["features_json"])
    d.update({"available": True, "id": row["id"], "ts": row["ts"], "model_version": row["model_version"], "probability": row["probability"],
              "tier": row["tier"], "n_days": row["n_days"], "window_days": row["window_days"], "sufficiency": row["sufficiency"],
              "completeness": row["completeness"], "mode": row["mode"], "is_demo": bool(row["is_demo"]),
              "explanation": json.loads(row["explanation_json"])})
    return d
