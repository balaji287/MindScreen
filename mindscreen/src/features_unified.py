"""Unified data layer: read validated records across the five categories, measure completeness / data quality,
and build one feature set per student. Missing data stays missing: nothing is silently filled in."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from categories import CATEGORIES, DATA_COLUMNS
from db import q

FULL_DAYS = 14        # days of data that count as a "complete" category


def series(pseudo_id, key, start=None, end=None) -> pd.DataFrame:
    spec = CATEGORIES[key]
    sql, args = f"SELECT * FROM {spec['table']} WHERE pseudo_id=?", [pseudo_id]
    if start: sql += " AND date>=?"; args.append(str(start))
    if end: sql += " AND date<=?"; args.append(str(end))
    rows = q(sql + " ORDER BY date", args)
    if not rows:
        return pd.DataFrame(columns=["date"] + [c["name"] for c in DATA_COLUMNS[key]])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop(columns=["id", "pseudo_id", "upload_id", "source", "is_demo"], errors="ignore")


def category_status(pseudo_id) -> dict:
    out = {}
    for key, spec in CATEGORIES.items():
        r = q(f"SELECT COUNT(*) n, MIN(date) a, MAX(date) b FROM {spec['table']} WHERE pseudo_id=?", (pseudo_id,), one=True)
        up = q("SELECT * FROM uploads WHERE pseudo_id=? AND category=? AND deleted_at IS NULL ORDER BY id DESC LIMIT 1", (pseudo_id, key), one=True)
        n = r["n"]
        if n > 0:
            status = "Validated"
        elif up:
            status = {"Preview": "Uploaded"}.get(up["status"], up["status"])
        else:
            status = "Not submitted"
        out[key] = {"status": status, "days": n, "first": r["a"], "last": r["b"],
                    "completeness": round(100 * min(n, FULL_DAYS) / FULL_DAYS),
                    "latest_upload_status": up["status"] if up else None}
    return out


def overall_completeness(status: dict) -> int:
    return round(sum(v["completeness"] for v in status.values()) / len(status))


def quality_warnings(pseudo_id, status=None) -> list:
    status = status or category_status(pseudo_id)
    warns = []
    for key, st in status.items():
        label = CATEGORIES[key]["label"]
        if st["status"] == "Not submitted":
            warns.append({"level": "info", "text": f"{label}: not submitted. Predictions use only the categories that are available."})
            continue
        if st["latest_upload_status"] == "Needs correction":
            warns.append({"level": "warn", "text": f"{label}: your latest upload needs correction."})
        if st["days"] and st["days"] < 7:
            warns.append({"level": "warn", "text": f"{label}: only {st['days']} day(s) of data; at least 7 are needed for stable summaries."})
        if st["last"] and (date.today() - datetime.strptime(st["last"], "%Y-%m-%d").date()).days > 14:
            warns.append({"level": "info", "text": f"{label}: latest record is from {st['last']} (more than 14 days ago)."})
        df = series(pseudo_id, key)
        if len(df) >= 2:
            d = df["date"].sort_values()
            gaps = d.diff().dt.days.dropna()
            if (gaps > 3).any():
                warns.append({"level": "info", "text": f"{label}: {int((gaps > 3).sum())} gap(s) longer than 3 days; gaps are shown as missing, not filled."})
            cols = [c["name"] for c in DATA_COLUMNS[key] if c["name"] in df.columns]
            miss = df[cols].isna().mean()
            for n, s in miss.items():
                if 0.5 < s < 1.0:
                    warns.append({"level": "info", "text": f"{label}: '{n}' is missing on {s:.0%} of days."})
    return warns


def window_bounds(pseudo_id, keys, window_days):
    """Window = last `window_days` days ending at the latest record among `keys`."""
    ends = []
    for k in keys:
        r = q(f"SELECT MAX(date) b FROM {CATEGORIES[k]['table']} WHERE pseudo_id=?", (pseudo_id,), one=True)
        if r and r["b"]:
            ends.append(r["b"])
    if not ends:
        return None, None
    end = max(ends)
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")
    return start, end


def _m(df, col):
    return float(df[col].mean()) if col in df.columns and df[col].notna().any() else None


def unified_features(pseudo_id, window_days=30) -> dict:
    """One consistent feature set across categories (None = not available). Used for prediction + reporting."""
    start, end = window_bounds(pseudo_id, list(CATEGORIES), window_days)
    f = {"window": {"start": start, "end": end, "days": window_days}, "available_categories": []}
    if not start:
        return f
    d = {k: series(pseudo_id, k, start, end) for k in CATEGORIES}
    f["available_categories"] = [k for k, df in d.items() if len(df)]
    s, b, p, so, a = d["screen"], d["behaviour"], d["physical"], d["social"], d["academic"]
    f["screen"] = {"total_h": _m(s, "total_screen_h"), "social_h": _m(s, "social_h"), "entertainment_h": _m(s, "entertainment_h"),
                   "productivity_h": _m(s, "productivity_h"), "pickups": _m(s, "pickups"), "night_h": _m(s, "night_h"), "days": len(s),
                   "top_app": (s["top_app"].dropna().mode().iloc[0] if "top_app" in s and s["top_app"].notna().any() else None)}
    f["behaviour"] = {k: _m(b, k) for k in ["sleep_wake_consistency", "study_routine_consistency", "concentration_self",
                                              "motivation_self", "stress_self", "mood_checkin_self"]} | {"days": len(b)}
    f["physical"] = {"sleep_h": _m(p, "sleep_h"), "sleep_sd": float(p["sleep_h"].std(ddof=0)) if "sleep_h" in p and p["sleep_h"].notna().sum() > 1 else None,
                     "sleep_quality": _m(p, "sleep_quality_self"), "steps": _m(p, "steps"), "exercise_min": _m(p, "exercise_min"), "days": len(p)}
    f["social"] = {k: _m(so, k) for k in ["interaction_freq_self", "social_time_min", "connectedness_self", "loneliness_self"]} | {"days": len(so)}
    f["academic"] = {k: _m(a, k) for k in ["attendance_pct", "assignment_completion_pct", "study_h", "workload_level", "academic_pressure_self"]} | {"days": len(a)}
    return f


def daily_frame_for_monitor(pseudo_id) -> pd.DataFrame:
    """Adapter so the existing change-monitor (footprint.drift) can run on stored records."""
    s, p, so = series(pseudo_id, "screen"), series(pseudo_id, "physical"), series(pseudo_id, "social")
    if s.empty and p.empty:
        return pd.DataFrame()
    base = pd.DataFrame({"date": sorted(set(s["date"]).union(p["date"]).union(so["date"]))})
    out = base.merge(s[["date", "total_screen_h", "night_h"]] if not s.empty else pd.DataFrame(columns=["date"]), on="date", how="left")
    out = out.merge(p[["date", "sleep_h", "steps", "exercise_min"]] if not p.empty else pd.DataFrame(columns=["date"]), on="date", how="left")
    out = out.merge(so[["date", "interaction_freq_self"]] if not so.empty else pd.DataFrame(columns=["date"]), on="date", how="left")
    ren = {"total_screen_h": "screen_h", "exercise_min": "active_min", "interaction_freq_self": "social"}
    out = out.rename(columns=ren)
    out["late_min"] = out["night_h"] * 60 if "night_h" in out else np.nan
    for c in ["screen_h", "sleep_h", "steps", "active_min", "social", "late_min"]:
        if c not in out: out[c] = np.nan
    out["social_media_h"] = np.nan
    out["top_app"] = ""
    return out.dropna(subset=["screen_h", "sleep_h"], how="all").reset_index(drop=True)
