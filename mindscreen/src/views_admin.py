"""Researcher / admin panel. Aggregates only; groups smaller than K are suppressed; no individual records are exposed."""
import csv, io, json
from datetime import datetime, timezone

import pandas as pd
from flask import Blueprint, Response, abort, current_app, g, jsonify, render_template, request

import predict_service as ps
from auth import role_required
from categories import CATEGORIES, DATA_COLUMNS
from db import audit, q
from features import RAW_COLUMNS

bp = Blueprint("admin", __name__, url_prefix="/admin")
K = 5          # minimum cohort size for any aggregate that is displayed


def scope():
    return 1 if request.args.get("scope") == "demo" else 0


def overview_stats(sc):
    n_students = q("SELECT COUNT(*) n FROM users WHERE role='student' AND is_demo=?", (sc,), one=True)["n"]
    n_uploads = q("SELECT COUNT(*) n FROM uploads WHERE deleted_at IS NULL AND pseudo_id IN (SELECT pseudo_id FROM users WHERE is_demo=?)", (sc,), one=True)["n"]
    cats, total_records = [], 0
    for key, spec in CATEGORIES.items():
        t = spec["table"]
        n_rec = q(f"SELECT COUNT(*) n FROM {t} WHERE is_demo=?", (sc,), one=True)["n"]
        total_records += n_rec
        per = q(f"SELECT COUNT(*) n FROM {t} WHERE is_demo=? GROUP BY pseudo_id", (sc,))
        n_with = len(per)
        comp = round(100 * sum(min(r["n"], 14) / 14 for r in per) / n_students) if n_students else 0
        row = {"key": key, "label": spec["label"], "records": n_rec, "students_with_data": n_with,
               "completeness": comp, "suppressed": n_with < K, "missing": []}
        if n_with >= K:
            for c in DATA_COLUMNS[key]:
                if c["kind"] in ("num", "int"):
                    r = q(f"SELECT AVG({c['name']} IS NULL) m FROM {t} WHERE is_demo=?", (sc,), one=True)
                    row["missing"].append({"col": c["name"], "pct": round(100 * (r["m"] or 0))})
        cats.append(row)
    latest = q("""SELECT p.tier FROM predictions p JOIN (SELECT pseudo_id, MAX(id) mid FROM predictions WHERE is_demo=? GROUP BY pseudo_id) m ON p.id=m.mid""", (sc,))
    tiers = {t: sum(1 for r in latest if r["tier"] == t) for t in ("Low", "Moderate", "High")}
    pred = {"n": len(latest), "suppressed": len(latest) < K,
            "tiers": {t: (c if c >= K else f"<{K}") for t, c in tiers.items()}}
    return {"n_students": n_students, "n_uploads": n_uploads, "records": total_records, "cats": cats, "pred": pred}


@bp.route("")
@role_required("researcher")
def index():
    sc = scope()
    audit("admin_view", "overview", "demo" if sc else "real")
    return render_template("admin/overview.html", st=overview_stats(sc), sc=sc, K=K)


@bp.route("/api/aggregate-trends")
@role_required("researcher")
def aggregate_trends():
    sc = scope()
    metrics = [("screen", "total_screen_h", "Screen time (h)"), ("physical", "sleep_h", "Sleep (h)"),
               ("physical", "steps", "Steps"), ("social", "connectedness_self", "Connectedness (1-5)"), ("academic", "workload_level", "Workload (1-5)")]
    out = []
    for cat, col, label in metrics:
        t = CATEGORIES[cat]["table"]
        rows = q(f"SELECT date, AVG({col}) a, COUNT(DISTINCT pseudo_id) n FROM {t} WHERE is_demo=? AND {col} IS NOT NULL GROUP BY date HAVING n>=? ORDER BY date", (sc, K))
        out.append({"label": label, "points": [{"x": r["date"], "y": round(r["a"], 2)} for r in rows]})
    return jsonify({"min_cohort": K, "series": out})


@bp.route("/models")
@role_required("researcher")
def models():
    audit("admin_view", "models")
    versions = q("SELECT * FROM model_versions ORDER BY id DESC")
    active = next((v for v in versions if v["is_active"]), None)
    metrics = json.loads(active["metrics_json"]) if active else None
    return render_template("admin/models.html", versions=versions, active=active, metrics=metrics)


@bp.route("/audit")
@role_required("researcher")
def audit_log():
    audit("admin_view", "audit")
    rows = q("SELECT ts, actor, role, action, target, detail FROM audit_log ORDER BY id DESC LIMIT 200")
    return render_template("admin/audit.html", rows=rows)


@bp.route("/batch", methods=["GET", "POST"])
@role_required("researcher")
def batch():
    ctx = {"summary": None, "error": None, "cols": RAW_COLUMNS, "K": K}
    if request.method == "POST":
        try:
            res = ps.predictor().batch(pd.read_csv(request.files.get("file")))
            counts = res["risk_tier"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)
            if len(res) < K:
                raise ValueError(f"Cohorts smaller than {K} are not analysed (privacy safeguard).")
            ctx["summary"] = {"n": len(res), "counts": counts.to_dict(), "avg_usage": round(res["Avg_Daily_Usage_Hours"].mean(), 1),
                              "avg_sleep": round(res["Sleep_Hours_Per_Night"].mean(), 1), "high_pct": round(100 * counts["High"] / len(res), 1)}
            current_app.config.setdefault("LAST_BATCH", {})[g.user["id"]] = res[[c for c in res.columns if c != "Student_ID"]].to_csv(index=False)
            audit("batch_analysis", "cohort", f"n={len(res)}")
        except Exception as e:  # noqa
            ctx["error"] = str(e)
    return render_template("admin/batch.html", **ctx)


@bp.route("/batch/download")
@role_required("researcher")
def batch_download():
    data = current_app.config.get("LAST_BATCH", {}).get(g.user["id"])
    if not data:
        abort(404)
    return Response(data, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=cohort_results.csv"})


@bp.route("/report.csv")
@role_required("researcher")
def aggregate_report():
    sc = scope()
    st = overview_stats(sc)
    audit("admin_report", "aggregate", "demo" if sc else "real")
    out = io.StringIO(); w = csv.writer(out)
    w.writerow([f"MindScreen aggregate report ({'DEMO data' if sc else 'real data'}); groups smaller than {K} suppressed"])
    w.writerow(["Generated", datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")]); w.writerow(["Students", st["n_students"]]); w.writerow(["Uploads", st["n_uploads"]])
    w.writerow([]); w.writerow(["Category", "Students with data", "Records", "Completeness %"])
    for c in st["cats"]:
        w.writerow([c["label"], c["students_with_data"] if not c["suppressed"] else f"<{K}", c["records"] if not c["suppressed"] else "suppressed", c["completeness"] if not c["suppressed"] else "suppressed"])
    w.writerow([]); w.writerow(["Screening category (latest per student)", "Count"])
    for t, c in st["pred"]["tiers"].items():
        w.writerow([t, "suppressed" if st["pred"]["suppressed"] else c])
    w.writerow([]); w.writerow(["Disclaimer", "Screening/research aid, not a diagnosis."])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=mindscreen_aggregate.csv"})
