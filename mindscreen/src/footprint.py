"""Digital-footprint engine: parse an uploaded daily-activity file, summarise it for the ML model,
detect behaviour change against the person's OWN baseline, and draw trend charts.

Accepted columns (names are flexible, see ALIASES). Required: date, screen time, sleep.
Optional: social-media hours, steps, active minutes, social interactions (calls+messages+meetups),
late-night screen minutes, top app.
"""
import base64, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALIASES = {
    "date": ["date", "day"],
    "screen_h": ["screen_hours", "screen_time_hours", "screen_time_h", "screen_time"],
    "screen_min": ["screen_minutes", "screen_time_minutes", "screen_time_min"],
    "social_media_h": ["social_media_hours", "social_hours"],
    "sleep_h": ["sleep_hours", "sleep_time_hours", "sleep_h", "sleep"],
    "sleep_min": ["sleep_minutes", "sleep_min", "minutes_asleep"],
    "steps": ["steps", "step_count"],
    "active_min": ["active_minutes", "active_min", "exercise_minutes"],
    "social": ["social_interactions", "social_contacts", "calls_messages", "interactions"],
    "late_min": ["late_night_screen_minutes", "night_screen_minutes", "late_night_minutes"],
    "top_app": ["top_app", "most_used_app", "main_app"],
}
PLATFORMS = ["instagram", "tiktok", "facebook", "whatsapp", "twitter", "linkedin", "wechat",
             "snapchat", "line", "kakaotalk", "vkontakte", "youtube"]
DISPLAY = {"Instagram": "Instagram", "Tiktok": "TikTok", "Whatsapp": "WhatsApp", "Linkedin": "LinkedIn",
           "Wechat": "WeChat", "Line": "LINE", "Kakaotalk": "KakaoTalk", "Vkontakte": "VKontakte",
           "Youtube": "YouTube"}

# label, worse direction (+1 = higher is worse, -1 = lower is worse)
SIGNALS = {
    "screen_h": ("Screen time (h/day)", +1), "sleep_h": ("Sleep (h/night)", -1),
    "sleep_sd": ("Sleep irregularity (SD, h)", +1), "steps": ("Steps per day", -1),
    "active_min": ("Active minutes", -1), "social": ("Social interactions per day", -1),
    "late_min": ("Late-night screen (min)", +1),
}


class FootprintError(ValueError):
    pass


def _pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    out = pd.DataFrame()
    c = _pick(df, ALIASES["date"])
    if c is None:
        raise FootprintError("No date column found (expected 'date').")
    out["date"] = pd.to_datetime(df[c], errors="coerce")
    # screen time (hours or minutes)
    c = _pick(df, ALIASES["screen_h"]); m = _pick(df, ALIASES["screen_min"])
    if c: out["screen_h"] = pd.to_numeric(df[c], errors="coerce")
    elif m: out["screen_h"] = pd.to_numeric(df[m], errors="coerce") / 60
    else: raise FootprintError("No screen-time column found (expected 'screen_hours' or 'screen_minutes').")
    c = _pick(df, ALIASES["sleep_h"]); m = _pick(df, ALIASES["sleep_min"])
    if c: out["sleep_h"] = pd.to_numeric(df[c], errors="coerce")
    elif m: out["sleep_h"] = pd.to_numeric(df[m], errors="coerce") / 60
    else: raise FootprintError("No sleep column found (expected 'sleep_hours' or 'sleep_minutes').")
    for key in ["social_media_h", "steps", "active_min", "social", "late_min"]:
        c = _pick(df, ALIASES[key])
        out[key] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
    c = _pick(df, ALIASES["top_app"])
    out["top_app"] = df[c].astype(str) if c else ""
    out = out.dropna(subset=["date", "screen_h", "sleep_h"]).sort_values("date").drop_duplicates("date")
    out = out[(out.screen_h.between(0, 20)) & (out.sleep_h.between(0, 16))].reset_index(drop=True)
    if len(out) < 7:
        raise FootprintError(f"Need at least 7 valid days of data; found {len(out)}.")
    return out


def platform_from(top_apps: pd.Series, categories: list) -> str:
    apps = top_apps[top_apps.str.len() > 0].str.lower()
    if apps.empty:
        return "Other"
    mode = apps.mode()[0]
    mode = "twitter" if mode in ("x", "x (twitter)") else mode
    for p in PLATFORMS:
        if p in mode:
            name = DISPLAY.get(p.capitalize(), p.capitalize())
            return name if name in categories else "Other"
    return "Other"


def summarise(d: pd.DataFrame, categories: list) -> dict:
    usage = d["social_media_h"] if d["social_media_h"].notna().any() else d["screen_h"]
    return {
        "days": int(len(d)), "start": d.date.iloc[0].strftime("%d %b"), "end": d.date.iloc[-1].strftime("%d %b %Y"),
        "Avg_Daily_Usage_Hours": float(usage.mean()), "Sleep_Hours_Per_Night": float(d.sleep_h.mean()),
        "Most_Used_Platform": platform_from(d.top_app, categories),
        "usage_source": "social-media hours" if d["social_media_h"].notna().any() else "total screen time",
        "context": {
            "steps": None if d.steps.isna().all() else float(d.steps.mean()),
            "active_min": None if d.active_min.isna().all() else float(d.active_min.mean()),
            "social": None if d.social.isna().all() else float(d.social.mean()),
            "late_min": None if d.late_min.isna().all() else float(d.late_min.mean()),
            "sleep_sd": float(d.sleep_h.std(ddof=0)),
        },
    }


def recent_frame(d: pd.DataFrame, n=7) -> pd.DataFrame:
    return d.tail(n)


def drift(d: pd.DataFrame, recent_days=7):
    """Compare the last `recent_days` with the person's own earlier baseline.
    Heuristic (no labels needed): flag a signal when it moved >= 2.5 standard errors AND >= 10% in the WORSE direction."""
    if len(d) < recent_days + 7:
        return None
    base, rec = d.iloc[:-recent_days], d.tail(recent_days)
    rows = []
    for key, (label, worse_dir) in SIGNALS.items():
        if key == "sleep_sd":
            b_series = base.sleep_h.rolling(recent_days).std(ddof=0).dropna()
            bm, bs = float(b_series.mean()), float(b_series.std(ddof=0)) if len(b_series) > 1 else 0.0
            rm = float(rec.sleep_h.std(ddof=0)); n = 1
            se = max(bs, 0.05 * abs(bm), 0.05)
        else:
            if d[key].isna().all():
                continue
            b, r = base[key].dropna(), rec[key].dropna()
            if len(b) < 5 or len(r) < 3:
                continue
            bm, rm, n = float(b.mean()), float(r.mean()), len(r)
            se = max(float(b.std(ddof=0)) / np.sqrt(n), 0.03 * abs(bm), 1e-6)
        z = (rm - bm) / se
        worse = z * worse_dir > 0
        dec = 0 if abs(bm) >= 100 else 1
        rows.append({"key": key, "label": label, "baseline": round(bm, dec), "recent": round(rm, dec),
                     "pct": round(100 * (rm - bm) / bm, 0) if bm else 0.0, "z": round(float(z), 1),
                     "flag": bool(worse and abs(z) >= 2.5 and abs(100 * (rm - bm) / bm if bm else 0) >= 10), "worse": bool(worse)})
    n_flag = sum(r["flag"] for r in rows)
    level = "Stable" if n_flag == 0 else ("Watch" if n_flag <= 2 else "Notable change")
    colour = {"Stable": "green", "Watch": "amber", "Notable change": "red"}[level]
    return {"rows": rows, "n_flag": n_flag, "level": level, "colour": colour, "recent_days": recent_days}


def chart(d: pd.DataFrame, recent_days=7) -> str:
    panels = [("screen_h", "Screen time (h)", "#3b5bdb"), ("sleep_h", "Sleep (h)", "#5f3dc4"),
              ("steps", "Steps", "#2f9e6b"), ("social", "Social interactions", "#e8a317")]
    panels = [p for p in panels if d[p[0]].notna().any()]
    cols = 2; rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows), squeeze=False)
    cut = d.date.iloc[-recent_days] if len(d) > recent_days else d.date.iloc[0]
    for ax, (k, title, colr) in zip(axes.ravel(), panels):
        ax.plot(d.date, d[k], color=colr, lw=1.8)
        if len(d) >= recent_days + 7:
            bm = d[k].iloc[:-recent_days].mean()
            ax.axhline(bm, color="grey", ls="--", lw=1)
            ax.axvspan(cut, d.date.iloc[-1], color=colr, alpha=.12)
        ax.set_title(title, fontsize=10, loc="left"); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.text(0.99, 0.005, "dashed = your baseline · shaded = last 7 days", ha="right", fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
