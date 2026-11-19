"""ML prediction, explainability, behavioural trends, reports, privacy & consent."""
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for)
from werkzeug.security import check_password_hash

import explain
import features_unified as fu
import footprint as fp
import predict_service as ps
import reports
from auth import consents_of, login_required, role_required, set_consent
from categories import CATEGORIES, DATA_COLUMNS
from db import audit, q, run

bp = Blueprint("analysis", __name__)


# ------------------------------------------------------------------ ML prediction
@bp.route("/predict")
@role_required("student")
def predict_page():
    pid = g.user["pseudo_id"]
    row = q("SELECT * FROM predictions WHERE pseudo_id=? ORDER BY id DESC LIMIT 1", (pid,), one=True)
    prep = ps.prepare_inputs(g.user, 30)
    history = q("SELECT ts, model_version, probability, tier, sufficiency, n_days FROM predictions WHERE pseudo_id=? ORDER BY id DESC LIMIT 8", (pid,))
    return render_template("app/predict.html", pred=ps.load_prediction(row) if row else None, prep=prep, history=history,
                           status=fu.category_status(pid))


@bp.route("/predict/run", methods=["POST"])
@role_required("student")
def predict_run():
    days = request.form.get("window", "30")
    days = int(days) if days in ("30", "60", "90") else 30
    res = ps.run_prediction(g.user, days)
    if not res["available"]:
        flash(res["message"], "error")
    else:
        audit("prediction_run", res["model_version"], f"p={res['probability']} suff={res['sufficiency']}")
        flash("Screening completed. This is a decision-support indicator, not a diagnosis.", "ok")
    return redirect(url_for("analysis.predict_page"))


@bp.route("/api/predictions/run", methods=["POST"])
@role_required("student")
def api_run():
    """JSON prediction endpoint for the signed-in student's own authorised record."""
    body = request.get_json(silent=True) or {}
    days = body.get("window_days", 30)
    days = days if days in (30, 60, 90) else 30
    res = ps.run_prediction(g.user, days)
    if not res["available"]:
        return jsonify({"available": False, "reasons": res["reasons"], "message": res["message"]}), 409
    audit("prediction_run", res["model_version"], "api")
    keep = ("available", "model_version", "ts", "probability", "tier", "sufficiency", "n_days", "categories_used",
            "categories_collected_not_used", "mode", "completeness")
    out = {k: res[k] for k in keep}
    out["explanation"] = {"method": res["explanation"]["method"], "items": res["explanation"]["items"]}
    return jsonify(out)


# ------------------------------------------------------------------ Explainable AI
@bp.route("/explain")
@role_required("student")
def explain_page():
    row = q("SELECT * FROM predictions WHERE pseudo_id=? ORDER BY id DESC LIMIT 1", (g.user["pseudo_id"],), one=True)
    pred = ps.load_prediction(row) if row else None
    shap_json = json.dumps([{"label": i["label"], "value": i["effect_pp"]} for i in pred["explanation"]["items"]]) if pred else "[]"
    global_json = json.dumps([{"label": i["label"], "value": i["value"]} for i in explain.global_importance()])
    return render_template("app/explain.html", pred=pred, shap_json=shap_json, global_json=global_json)


# ------------------------------------------------------------------ Behavioural trends
CHARTS = {
    "screen": ("Daily screen time", "hours", [("screen", "total_screen_h", "Total screen time", "y"), ("screen", "social_h", "Social media", "y"), ("screen", "night_h", "Night-time use", "y")]),
    "sleep": ("Sleep duration and consistency", "hours / score", [("physical", "sleep_h", "Sleep duration (h)", "y"), ("behaviour", "sleep_wake_consistency", "Sleep/wake consistency (0-10)", "y2")]),
    "study": ("Study routine", "score / hours", [("behaviour", "study_routine_consistency", "Study routine consistency (0-10)", "y"), ("academic", "study_h", "Study hours", "y2")]),
    "activity": ("Physical activity", "steps / minutes", [("physical", "steps", "Steps", "y"), ("physical", "exercise_min", "Exercise (min)", "y2")]),
    "social": ("Social connectedness check-ins (self-reported)", "1-5", [("social", "connectedness_self", "Connectedness", "y"), ("social", "loneliness_self", "Loneliness", "y"), ("social", "interaction_freq_self", "Interactions (count)", "y2")]),
    "academic": ("Academic workload", "level / %", [("academic", "workload_level", "Workload (1-5)", "y"), ("academic", "academic_pressure_self", "Pressure, self-reported (1-5)", "y"), ("academic", "assignment_completion_pct", "Assignments completed (%)", "y2")]),
}


def build_trends(pseudo_id, which, start, end):
    idx = pd.date_range(start, end, freq="D")
    if len(idx) > 400:
        idx = idx[-400:]
    out = []
    cache = {}
    for cid, (title, unit, defs) in CHARTS.items():
        if which not in ("all", cid):
            continue
        sets, any_data = [], False
        for cat, colname, label, axis in defs:
            if cat not in cache:
                cache[cat] = fu.series(pseudo_id, cat, idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d"))
            df = cache[cat]
            s = pd.Series(np.nan, index=idx)
            if not df.empty and colname in df.columns:
                d = df.set_index("date")[colname].astype(float)
                s.update(d)
            any_data |= bool(s.notna().any())
            sets.append({"label": label, "axis": axis, "data": [None if np.isnan(v) else round(float(v), 2) for v in s.values],
                         "missing": int(s.isna().sum())})
        out.append({"id": cid, "title": title, "unit": unit, "labels": [d.strftime("%Y-%m-%d") for d in idx], "datasets": sets, "has_data": any_data})
    return out


@bp.route("/api/trends")
@role_required("student")
def api_trends():
    pid = g.user["pseudo_id"]
    which = request.args.get("category", "all")
    if which not in ("all", "overview", *CHARTS):
        abort(400, "Unknown category")
    _, latest = fu.window_bounds(pid, list(CATEGORIES), 60)
    try:
        end = datetime.strptime(request.args.get("to") or latest or date.today().isoformat(), "%Y-%m-%d").date()
        start = datetime.strptime(request.args.get("from") or (end - timedelta(days=59)).isoformat(), "%Y-%m-%d").date()
    except ValueError:
        abort(400, "Dates must be YYYY-MM-DD")
    if start > end:
        abort(400, "'from' is after 'to'")
    charts = build_trends(pid, "screen" if which == "overview" else which, start, end)
    if which == "overview":     # compact overview: screen time + sleep on one chart
        sc = build_trends(pid, "screen", start, end)[0]; sl = build_trends(pid, "sleep", start, end)[0]
        charts = [{"id": "overview", "title": "Screen time and sleep", "unit": "hours", "labels": sc["labels"],
                   "datasets": [sc["datasets"][0], sl["datasets"][0]], "has_data": sc["has_data"] or sl["has_data"]}]
    return jsonify({"from": start.isoformat(), "to": end.isoformat(), "charts": charts})


@bp.route("/trends")
@role_required("student")
def trends_page():
    pid = g.user["pseudo_id"]
    _, latest = fu.window_bounds(pid, list(CATEGORIES), 60)
    end = datetime.strptime(latest, "%Y-%m-%d").date() if latest else date.today()
    drift = None
    frame = fu.daily_frame_for_monitor(pid)
    if len(frame) >= 14:
        drift = fp.drift(frame)
    return render_template("app/trends.html", start=(end - timedelta(days=59)).isoformat(), end=end.isoformat(), charts=CHARTS,
                           drift=drift, has_any=latest is not None)


# ------------------------------------------------------------------ Reports
def _report_ctx():
    row = q("SELECT * FROM predictions WHERE pseudo_id=? ORDER BY id DESC LIMIT 1", (g.user["pseudo_id"],), one=True)
    return reports.build_context(g.user, ps.load_prediction(row) if row else None)


@bp.route("/reports")
@role_required("student")
def reports_page():
    ctx = _report_ctx()
    return render_template("app/reports.html", ctx=ctx, rows=reports.trend_rows(ctx), disclaimer=reports.DISCLAIMER)


@bp.route("/reports/download.<fmt>")
@role_required("student")
def report_download(fmt):
    if fmt not in ("csv", "pdf"):
        abort(404)
    ctx = _report_ctx()
    audit("report_download", fmt)
    if fmt == "csv":
        return Response(reports.csv_report(ctx), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=mindscreen_report.csv"})
    return Response(reports.pdf_report(ctx), mimetype="application/pdf", headers={"Content-Disposition": "attachment; filename=mindscreen_report.pdf"})


# ------------------------------------------------------------------ Privacy & consent
@bp.route("/privacy")
@login_required
def privacy_page():
    return render_template("app/privacy.html", consents=consents_of(g.user["id"]))


@bp.route("/privacy/consent", methods=["POST"])
@role_required("student")
def privacy_consent():
    kind = request.form.get("kind")
    grant = request.form.get("grant") == "1"
    if kind not in ("wearable", "grades"):
        abort(400, "Unknown consent type")
    set_consent(g.user["id"], kind, grant)
    audit("consent_changed", kind, "granted" if grant else "withdrawn")
    if not grant:   # withdrawal removes the data that depended on it
        col = {"wearable": ("rec_physical", "resting_hr=NULL, hrv_ms=NULL"), "grades": ("rec_academic", "grade_pct=NULL")}[kind]
        run(f"UPDATE {col[0]} SET {col[1]} WHERE pseudo_id=?", (g.user["pseudo_id"],))
        flash(f"Consent withdrawn; stored {kind} data was removed.", "ok")
    else:
        flash("Consent recorded.", "ok")
    return redirect(url_for("analysis.privacy_page"))


def _wipe(pseudo_id):
    import os
    for k, spec in CATEGORIES.items():
        run(f"DELETE FROM {spec['table']} WHERE pseudo_id=?", (pseudo_id,))
    for u in q("SELECT stored_path FROM uploads WHERE pseudo_id=?", (pseudo_id,)):
        try:
            if u["stored_path"] and os.path.isfile(u["stored_path"]): os.remove(u["stored_path"])
        except OSError:
            pass
    run("DELETE FROM uploads WHERE pseudo_id=?", (pseudo_id,))
    run("DELETE FROM predictions WHERE pseudo_id=?", (pseudo_id,))


@bp.route("/privacy/delete-data", methods=["POST"])
@role_required("student")
def delete_data():
    if not check_password_hash(g.user["password_hash"], request.form.get("password", "")):
        flash("Password incorrect; nothing was deleted.", "error"); return redirect(url_for("analysis.privacy_page"))
    _wipe(g.user["pseudo_id"])
    audit("data_deleted", g.user["pseudo_id"], "all records, uploads, predictions")
    flash("All your data, uploads and predictions were deleted. Your account remains.", "ok")
    return redirect(url_for("analysis.privacy_page"))


@bp.route("/privacy/delete-account", methods=["POST"])
@login_required
def delete_account():
    from flask import session
    if not check_password_hash(g.user["password_hash"], request.form.get("password", "")):
        flash("Password incorrect; your account was not deleted.", "error"); return redirect(url_for("analysis.privacy_page"))
    pid, uid = g.user["pseudo_id"], g.user["id"]
    _wipe(pid)
    audit("account_deleted", pid)
    run("DELETE FROM users WHERE id=?", (uid,))
    session.clear()
    flash("Your account and all associated data were deleted.", "ok")   # new session, so the message survives
    return redirect(url_for("main.landing"))
