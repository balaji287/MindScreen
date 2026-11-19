"""Landing, overview dashboard, settings, about, model card."""
import json
from pathlib import Path

from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import features_unified as fu
import predict_service as ps
from auth import GENDERS, LEVELS, login_required, role_required
from categories import CATEGORIES
from db import audit, q, run

bp = Blueprint("main", __name__)
ROOT = Path(__file__).resolve().parent.parent


@bp.route("/")
def landing():
    return render_template("landing.html")


@bp.route("/app")
@login_required
def home():
    if g.user["role"] == "researcher":
        return redirect(url_for("admin.index"))
    pid = g.user["pseudo_id"]
    status = fu.category_status(pid)
    pred = ps.latest_prediction(pid)
    recent = q("SELECT ts, action, target FROM audit_log WHERE actor=? AND action IN ('upload_confirmed','upload_deleted','manual_entry','prediction_run','consent_changed','report_download','data_deleted') ORDER BY id DESC LIMIT 8", (pid,))
    return render_template("app/overview.html", status=status, completeness=fu.overall_completeness(status),
                           n_submitted=sum(1 for s in status.values() if s["days"] > 0), pred=pred,
                           warnings=fu.quality_warnings(pid, status)[:6], recent=recent)


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    msg = err = None
    if request.method == "POST":
        f = request.form
        if f.get("action") == "profile":
            age = int(f["age"]) if f.get("age", "").isdigit() and 16 <= int(f["age"]) <= 40 else None
            gender = f.get("gender") if f.get("gender") in GENDERS else None
            level = f.get("academic_level") if f.get("academic_level") in LEVELS else None
            run("UPDATE users SET age=?, gender=?, academic_level=? WHERE id=?", (age, gender, level, g.user["id"]))
            audit("profile_updated"); msg = "Profile saved."
        elif f.get("action") == "password":
            from auth import password_problem
            if not check_password_hash(g.user["password_hash"], f.get("current", "")):
                err = "Current password is incorrect."
            elif f.get("new") != f.get("confirm"):
                err = "New passwords do not match."
            elif password_problem(f.get("new", "")):
                err = password_problem(f.get("new", ""))
            else:
                run("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(f["new"]), g.user["id"]))
                audit("password_changed"); msg = "Password changed."
        g.user = q("SELECT * FROM users WHERE id=?", (g.user["id"],), one=True)
    return render_template("app/settings.html", msg=msg, err=err, genders=GENDERS, levels=LEVELS)


@bp.route("/about")
def about():
    p = ps.predictor()
    return render_template("about.html", s=p.schema)


@bp.route("/model-card")
def model_card():
    p = ps.predictor()
    ab = None
    f = ROOT / "models" / "ablation.json"
    if f.exists():
        ab = json.loads(f.read_text())
    return render_template("model_card.html", m=p.metrics, s=p.schema, ab=ab)


@bp.route("/healthz")
def healthz():
    return {"status": "ok"}
