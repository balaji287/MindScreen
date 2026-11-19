"""SQLite helpers + schema. Record tables are generated from categories.py."""
import sqlite3
from pathlib import Path
from flask import current_app, g
from categories import CATEGORIES

SQL_TYPE = {"num": "REAL", "int": "INTEGER", "cat": "TEXT", "text": "TEXT", "date": "TEXT"}

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('student','researcher')),
  pseudo_id TEXT UNIQUE NOT NULL, age INTEGER, gender TEXT, academic_level TEXT,
  is_demo INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consents(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK(kind IN ('core','wearable','grades')), granted INTEGER NOT NULL, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uploads(
  id INTEGER PRIMARY KEY, pseudo_id TEXT NOT NULL, category TEXT NOT NULL, filename TEXT, stored_path TEXT,
  rows_total INTEGER, rows_valid INTEGER, status TEXT NOT NULL, errors_json TEXT, token TEXT,
  created_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, pseudo_id TEXT NOT NULL, ts TEXT NOT NULL, model_version TEXT NOT NULL,
  mode TEXT NOT NULL, probability REAL, tier TEXT, window_days INTEGER, n_days INTEGER,
  features_json TEXT, explanation_json TEXT, categories_used TEXT, completeness REAL, sufficiency TEXT, is_demo INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS model_versions(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, kind TEXT,
  metrics_json TEXT, notes TEXT, is_active INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, actor TEXT NOT NULL, role TEXT, action TEXT NOT NULL,
  target TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_uploads_p ON uploads(pseudo_id, category);
CREATE INDEX IF NOT EXISTS idx_pred_p ON predictions(pseudo_id, ts);
"""


def record_table_sql(key: str) -> str:
    spec = CATEGORIES[key]
    cols = ",\n  ".join(f"{c['name']} {SQL_TYPE[c['kind']]}" for c in spec["columns"] if c["name"] != "date")
    return (f"CREATE TABLE IF NOT EXISTS {spec['table']}(\n  id INTEGER PRIMARY KEY, pseudo_id TEXT NOT NULL, "
            f"upload_id INTEGER, date TEXT NOT NULL,\n  {cols},\n  source TEXT NOT NULL DEFAULT 'upload', "
            f"is_demo INTEGER NOT NULL DEFAULT 0, UNIQUE(pseudo_id, date));\n"
            f"CREATE INDEX IF NOT EXISTS idx_{spec['table']}_p ON {spec['table']}(pseudo_id, date);")


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_schema(con: sqlite3.Connection):
    con.executescript(CORE_SCHEMA)
    for k in CATEGORIES:
        con.executescript(record_table_sql(k))
    con.commit()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
    return g.db


def close_db(_=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def q(sql, args=(), one=False):
    cur = get_db().execute(sql, args)
    rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def run(sql, args=()):
    db = get_db()
    cur = db.execute(sql, args)
    db.commit()
    return cur


def audit(action, target="", detail="", actor=None, role=None):
    from datetime import datetime, timezone
    from flask import g as _g
    u = getattr(_g, "user", None)
    actor = actor or (u["pseudo_id"] if u else "anonymous")
    role = role or (u["role"] if u else None)
    run("INSERT INTO audit_log(ts,actor,role,action,target,detail) VALUES(?,?,?,?,?,?)",
        (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), actor, role, action, str(target)[:80], str(detail)[:300]))
