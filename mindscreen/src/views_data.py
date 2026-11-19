"""Data collection: five separate category pages with upload -> preview -> confirm, manual entry, replace/delete."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import (Blueprint, Response, abort, current_app, flash, g, redirect, render_template, request, url_for)

import features_unified as fu
import ingest
from auth import consents_of, login_required, role_required
from categories import CATEGORIES, DATA_COLUMNS, template_csv
from db import audit, get_db, q, run

bp = Blueprint("data", __name__, url_prefix="/data")


def _cat(key):
    if key not in CATEGORIES:
        abort(404)
    return CATEGORIES[key]


def _own_upload(upload_id):
    u = q("SELECT * FROM uploads WHERE id=? AND pseudo_id=? AND deleted_at IS NULL", (upload_id, g.user["pseudo_id"]), one=True)
    if not u:
        abort(404)          # same answer for "missing" and "someone else's": no information leak
    return u


def _remove_file(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


@bp.route("")
@role_required("student")
def hub():
    status = fu.category_status(g.user["pseudo_id"])
    return render_template("app/data_hub.html", status=status, completeness=fu.overall_completeness(status))


@bp.route("/<key>")
@role_required("student")
def category(key):
    spec = _cat(key)
    pid = g.user["pseudo_id"]
    st = fu.category_status(pid)[key]
    uploads = q("SELECT * FROM uploads WHERE pseudo_id=? AND category=? AND deleted_at IS NULL ORDER BY id DESC LIMIT 10", (pid, key))
    return render_template("app/data_category.html", key=key, spec=spec, st=st, uploads=uploads, consents=consents_of(g.user["id"]),
                           errors=None)


@bp.route("/<key>/template.csv")
@role_required("student")
def template(key):
    _cat(key)
    return Response(template_csv(key), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=mindscreen_{key}_template.csv"})


@bp.route("/<key>/preview", methods=["POST"])
@role_required("student")
def preview(key):
    spec = _cat(key)
    pid = g.user["pseudo_id"]
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a CSV file.", "error"); return redirect(url_for("data.category", key=key))
    if not f.filename.lower().endswith(".csv"):
        flash("Only .csv files are accepted.", "error"); return redirect(url_for("data.category", key=key))
    raw = f.read(ingest.MAX_BYTES + 1)
    try:
        df = ingest.read_csv_bytes(raw)
    except ValueError as e:
        run("INSERT INTO uploads(pseudo_id,category,filename,rows_total,rows_valid,status,errors_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (pid, key, Path(f.filename).name[:80], 0, 0, "Needs correction", json.dumps([{"row": 0, "col": "", "msg": str(e)}]), datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")))
        audit("upload_rejected", key, str(e)); flash(str(e), "error"); return redirect(url_for("data.category", key=key))
    clean, rep = ingest.validate_frame(df, key, consents_of(g.user["id"]))
    path = ingest.store_original(current_app.config["UPLOAD_DIR"], pid, raw)
    status = "Uploaded" if (clean is not None and not rep["errors"]) else "Needs correction"
    cur = run("INSERT INTO uploads(pseudo_id,category,filename,stored_path,rows_total,rows_valid,status,errors_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
              (pid, key, Path(f.filename).name[:80], path, rep["rows_total"], rep["rows_valid"], status,
               json.dumps(rep["errors"][:100]), datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")))
    audit("upload_validated" if status == "Uploaded" else "upload_needs_correction", key, f"rows={rep['rows_total']} errors={len(rep['errors'])}")
    existing = q(f"SELECT COUNT(*) n FROM {spec['table']} WHERE pseudo_id=?", (pid,), one=True)["n"]
    return render_template("app/data_preview.html", key=key, spec=spec, upload_id=cur.lastrowid, rep=rep, status=status,
                           rows=(clean.head(10).to_dict("records") if clean is not None else []), existing=existing,
                           cols=[c for c in spec["columns"] if c["name"] != "period"])


@bp.route("/uploads/<int:upload_id>/confirm", methods=["POST"])
@role_required("student")
def confirm(upload_id):
    u = _own_upload(upload_id)
    key = u["category"]
    if u["status"] != "Uploaded":
        flash("This upload cannot be submitted (it needs correction).", "error"); return redirect(url_for("data.category", key=key))
    df = ingest.read_csv_bytes(Path(u["stored_path"]).read_bytes())
    clean, rep = ingest.validate_frame(df, key, consents_of(g.user["id"]))
    if clean is None:
        run("UPDATE uploads SET status='Needs correction', errors_json=? WHERE id=?", (json.dumps(rep["errors"][:100]), upload_id))
        flash("Validation failed on re-check; please correct the file.", "error"); return redirect(url_for("data.category", key=key))
    pid = g.user["pseudo_id"]
    if request.form.get("replace") == "1":
        table = CATEGORIES[key]["table"]
        run(f"DELETE FROM {table} WHERE pseudo_id=? AND source='upload'", (pid,))
        for old in q("SELECT id, stored_path FROM uploads WHERE pseudo_id=? AND category=? AND id<>? AND deleted_at IS NULL", (pid, key, upload_id)):
            _remove_file(old["stored_path"])
            run("UPDATE uploads SET deleted_at=?, stored_path=NULL WHERE id=?", (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), old["id"]))
    n = ingest.insert_records(key, pid, upload_id, clean, "upload", int(g.user["is_demo"]))
    run("UPDATE uploads SET status='Validated', rows_valid=? WHERE id=?", (n, upload_id))
    audit("upload_confirmed", key, f"records={n} replace={request.form.get('replace') == '1'}")
    flash(f"{n} records submitted to {CATEGORIES[key]['label']}.", "ok")
    return redirect(url_for("data.category", key=key))


@bp.route("/uploads/<int:upload_id>/cancel", methods=["POST"])
@role_required("student")
def cancel(upload_id):
    u = _own_upload(upload_id)
    _remove_file(u["stored_path"])
    run("UPDATE uploads SET deleted_at=?, stored_path=NULL WHERE id=?", (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), upload_id))
    audit("upload_cancelled", u["category"])
    return redirect(url_for("data.category", key=u["category"]))


@bp.route("/uploads/<int:upload_id>/delete", methods=["POST"])
@role_required("student")
def delete_upload(upload_id):
    u = _own_upload(upload_id)
    key = u["category"]
    run(f"DELETE FROM {CATEGORIES[key]['table']} WHERE pseudo_id=? AND upload_id=?", (g.user["pseudo_id"], upload_id))
    _remove_file(u["stored_path"])
    run("UPDATE uploads SET deleted_at=?, stored_path=NULL WHERE id=?", (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), upload_id))
    audit("upload_deleted", key, f"upload={upload_id}")
    flash("Submission deleted.", "ok")
    return redirect(url_for("data.category", key=key))


@bp.route("/<key>/clear", methods=["POST"])
@role_required("student")
def clear(key):
    spec = _cat(key)
    pid = g.user["pseudo_id"]
    run(f"DELETE FROM {spec['table']} WHERE pseudo_id=?", (pid,))
    for u in q("SELECT id, stored_path FROM uploads WHERE pseudo_id=? AND category=? AND deleted_at IS NULL", (pid, key)):
        _remove_file(u["stored_path"])
        run("UPDATE uploads SET deleted_at=?, stored_path=NULL WHERE id=?", (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), u["id"]))
    audit("upload_deleted", key, "all data in category")
    flash(f"All {spec['label']} data deleted.", "ok")
    return redirect(url_for("data.category", key=key))


@bp.route("/<key>/manual", methods=["POST"])
@role_required("student")
def manual(key):
    spec = _cat(key)
    row = {c["name"]: request.form.get(c["name"], "").strip() for c in spec["columns"]}
    df = pd.DataFrame([row]).replace("", None)
    clean, rep = ingest.validate_frame(df, key, consents_of(g.user["id"]))
    if clean is None or rep["errors"]:
        flash("Entry not saved: " + "; ".join(e["msg"] for e in rep["errors"][:3]), "error")
        return redirect(url_for("data.category", key=key))
    ingest.insert_records(key, g.user["pseudo_id"], None, clean, "manual", int(g.user["is_demo"]))
    audit("manual_entry", key, clean.iloc[0]["date"])
    flash("Entry saved (an entry for the same date was replaced).", "ok")
    return redirect(url_for("data.category", key=key))
