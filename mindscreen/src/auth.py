"""Authentication, roles, CSRF, login rate-limiting."""
import os, re, secrets, time
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse

from flask import (Blueprint, abort, current_app, flash, g, redirect, render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from db import audit, q, run

bp = Blueprint("auth", __name__)
FAILS: dict = {}
_DUMMY = generate_password_hash("not-a-real-password")
GENDERS = ["Female", "Male"]
LEVELS = ["High School", "Undergraduate", "Graduate"]


# ---------- helpers ----------
def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(24)
    return session["csrf"]


def safe_next(target):
    if not target:
        return None
    p = urlparse(target)
    return target if (not p.scheme and not p.netloc and target.startswith("/") and not target.startswith("//")) else None


def consents_of(user_id) -> set:
    rows = q("SELECT kind, granted FROM consents WHERE user_id=? ORDER BY id", (user_id,))
    state = {}
    for r in rows:
        state[r["kind"]] = bool(r["granted"])
    return {k for k, v in state.items() if v}


def set_consent(user_id, kind, granted: bool):
    run("INSERT INTO consents(user_id,kind,granted,ts) VALUES(?,?,?,?)", (user_id, kind, int(granted), now()))


def password_problem(pw: str):
    if len(pw) < 10:
        return "Password must be at least 10 characters."
    if not (re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw)):
        return "Password must contain letters and at least one digit."
    return None


def create_user(email, password, role="student", age=None, gender=None, level=None, is_demo=0):
    pseudo = ("D-" if is_demo else "S-") + secrets.token_hex(4)
    while q("SELECT 1 FROM users WHERE pseudo_id=?", (pseudo,), one=True):
        pseudo = ("D-" if is_demo else "S-") + secrets.token_hex(4)
    cur = run("INSERT INTO users(email,password_hash,role,pseudo_id,age,gender,academic_level,is_demo,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
              (email.lower().strip(), generate_password_hash(password), role, pseudo, age, gender, level, is_demo, now()))
    return cur.lastrowid, pseudo


# ---------- decorators ----------
def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if g.user is None:
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return f(*a, **k)
    return w


def role_required(role):
    def deco(f):
        @wraps(f)
        @login_required
        def w(*a, **k):
            if g.user["role"] != role:
                audit("access_denied", request.path, f"needs role {role}")
                abort(403)
            return f(*a, **k)
        return w
    return deco


def csrf_exempt(f):
    f._csrf_exempt = True
    return f


def init_app(app):
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _load():
        g.user = None
        uid = session.get("uid")
        if uid:
            g.user = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
            if g.user is None:
                session.clear()
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            view = app.view_functions.get(request.endpoint)
            if view is not None and getattr(view, "_csrf_exempt", False):
                return
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
            if not sent or not secrets.compare_digest(str(sent), session.get("csrf", "")):
                abort(400, "Security check failed (CSRF token missing or invalid). Reload the page and try again.")

    @app.context_processor
    def _ctx():
        return {"user": g.get("user")}


# ---------- routes ----------
@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        email = f.get("email", "").strip().lower()
        pw, pw2 = f.get("password", ""), f.get("confirm", "")
        invite = f.get("invite", "").strip()
        err = None
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 120:
            err = "Please enter a valid email address."
        elif pw != pw2:
            err = "Passwords do not match."
        elif password_problem(pw):
            err = password_problem(pw)
        elif not f.get("consent_core"):
            err = "Consent is required before any data can be collected or processed."
        elif q("SELECT 1 FROM users WHERE email=?", (email,), one=True):
            err = "An account with this email already exists."
        role = "student"
        if invite:
            code = current_app.config.get("RESEARCHER_INVITE_CODE", "")
            if code and secrets.compare_digest(invite, code):
                role = "researcher"
            else:
                err = err or "Invalid invite code."
        if err:
            return render_template("auth/register.html", error=err, genders=GENDERS, levels=LEVELS, form=f), 400
        age = int(f["age"]) if f.get("age", "").isdigit() and 16 <= int(f["age"]) <= 40 else None
        gender = f.get("gender") if f.get("gender") in GENDERS else None
        level = f.get("academic_level") if f.get("academic_level") in LEVELS else None
        uid, pseudo = create_user(email, pw, role, age, gender, level)
        set_consent(uid, "core", True)
        audit("register", pseudo, role, actor=pseudo, role=role)
        session.clear(); session["uid"] = uid; session.permanent = True
        return redirect(url_for("main.home"))
    return render_template("auth/register.html", error=None, genders=GENDERS, levels=LEVELS, form={})


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        key = (request.remote_addr, email)
        recent = [t for t in FAILS.get(key, []) if time.time() - t < 600]
        FAILS[key] = recent
        if len(recent) >= 5:
            audit("login_blocked", "", "rate limit", actor="anonymous")
            return render_template("auth/login.html", error="Too many failed attempts. Please wait 10 minutes.", next=request.args.get("next")), 429
        u = q("SELECT * FROM users WHERE email=?", (email,), one=True)
        ok = check_password_hash(u["password_hash"] if u else _DUMMY, request.form.get("password", "")) and u is not None
        if not ok:
            FAILS[key].append(time.time())
            audit("login_failed", "", "", actor="anonymous")
            return render_template("auth/login.html", error="Invalid email or password.", next=request.args.get("next")), 401
        FAILS.pop(key, None)
        session.clear(); session["uid"] = u["id"]; session.permanent = True
        audit("login", u["pseudo_id"], "", actor=u["pseudo_id"], role=u["role"])
        return redirect(safe_next(request.args.get("next") or request.form.get("next")) or url_for("main.home"))
    return render_template("auth/login.html", error=None, next=request.args.get("next"))


@bp.route("/logout", methods=["POST"])
def logout():
    if g.user:
        audit("logout", g.user["pseudo_id"])
    session.clear()
    return redirect(url_for("main.landing"))
