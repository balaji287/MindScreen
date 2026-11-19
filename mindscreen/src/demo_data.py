"""Generates SIMULATED demo uploads (30 days) so the app can be demonstrated without a real phone export.
These files are for demo only; they are NOT used for training.   Run: python src/demo_data.py"""
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
dates = pd.date_range("2026-10-16", periods=30)


def persona(seed, base, recent=None, app="Instagram", cut=23):
    rng = np.random.default_rng(seed)
    rows = []
    for i, dt in enumerate(dates):
        p = recent if (recent and i >= cut) else base
        wk = 1.15 if dt.dayofweek >= 5 else 1.0
        rows.append({
            "date": dt.date().isoformat(),
            "screen_hours": round(max(0.5, rng.normal(p["screen"] * wk, p["screen_sd"])), 1),
            "sleep_hours": round(float(np.clip(rng.normal(p["sleep"], p["sleep_sd"]), 3, 10)), 1),
            "steps": int(max(500, rng.normal(p["steps"], p["steps"] * .18))),
            "active_minutes": int(max(0, rng.normal(p["active"], 8))),
            "social_interactions": int(max(0, rng.normal(p["social"], 2.5))),
            "late_night_screen_minutes": int(max(0, rng.normal(p["late"], 10))),
            "top_app": app,
        })
    return pd.DataFrame(rows)


good = dict(screen=3.0, screen_sd=.5, sleep=7.6, sleep_sd=.4, steps=8800, active=55, social=14, late=12)
heavy = dict(screen=7.0, screen_sd=.7, sleep=5.2, sleep_sd=.8, steps=3600, active=14, social=5, late=90)
decline = dict(screen=5.8, screen_sd=.7, sleep=5.9, sleep_sd=1.1, steps=3300, active=10, social=3, late=70)

persona(1, good, app="Instagram").to_csv(OUT / "balanced.csv", index=False)
persona(2, heavy, app="TikTok").to_csv(OUT / "heavy_user.csv", index=False)
persona(3, good, recent=decline, app="Instagram").to_csv(OUT / "withdrawing.csv", index=False)
print("wrote", [p.name for p in OUT.glob("*.csv")])
