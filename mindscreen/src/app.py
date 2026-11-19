"""MindScreen: multimodal student wellbeing screening platform.
Run:  python src/app.py   ->  http://127.0.0.1:5000
"""
import os, secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, render_template, request

import auth, db
from categories import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path):
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def create_app(test_config=None):
    load_env(ROOT / ".env")
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    inst = Path(os.environ.get("MINDSCREEN_INSTANCE", ROOT / "instance"))
    app.config.update(
        INSTANCE=inst, DB_PATH=inst / "mindscreen.db", UPLOAD_DIR=inst / "uploads",
        MAX_CONTENT_LENGTH=3 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("MINDSCREEN_HTTPS", "0") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        RESEARCHER_INVITE_CODE=os.environ.get("RESEARCHER_INVITE_CODE", ""),
    )
    if test_config:
        app.config.update(test_config)
    inst = Path(app.config["INSTANCE"]); inst.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    key = os.environ.get("MINDSCREEN_SECRET_KEY")
    if not key:                                    # dev fallback: random key persisted outside the code
        kf = inst / "secret.key"
        if not kf.exists():
            kf.write_text(secrets.token_hex(32)); os.chmod(kf, 0o600)
        key = kf.read_text().strip()
    app.config.setdefault("SECRET_KEY", key)
    app.config["SECRET_KEY"] = app.config.get("SECRET_KEY") or key

    con = db.connect(app.config["DB_PATH"]); db.init_schema(con); con.close()
    app.teardown_appcontext(db.close_db)
    auth.init_app(app)

    from views_main import bp as main_bp
    from views_data import bp as data_bp
    from views_analysis import bp as analysis_bp
    from views_admin import bp as admin_bp
    from views_quick import bp as quick_bp
    for b in (auth.bp, main_bp, data_bp, analysis_bp, admin_bp, quick_bp):
        app.register_blueprint(b)
    import json as _json
    app.jinja_env.globals.update(CATEGORIES=CATEGORIES)
    app.jinja_env.filters["fromjson"] = _json.loads

    with app.app_context():
        import predict_service
        predict_service.sync_model_version()

    @app.after_request
    def headers(r):
        r.headers["X-Content-Type-Options"] = "nosniff"
        r.headers["X-Frame-Options"] = "DENY"
        r.headers["Referrer-Policy"] = "no-referrer"
        r.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'"
        if not request.path.startswith("/static/"):
            r.headers["Cache-Control"] = "no-store"
        return r

    for code, title in [(400, "Request not accepted"), (403, "Not allowed"), (404, "Page not found"), (413, "File too large"), (429, "Too many attempts")]:
        app.register_error_handler(code, lambda e, code=code, title=title: (render_template("error.html", code=code, title=title, msg=getattr(e, "description", "")), code))
    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
