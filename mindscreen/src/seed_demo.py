"""Create FICTIONAL demo accounts + records (flagged is_demo=1, kept apart from real data).
Run:  python src/seed_demo.py        Passwords: set DEMO_PASSWORD in the environment, otherwise random ones are printed once."""
import os, secrets, sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import create_app
import auth, ingest
from categories import CATEGORIES
from db import get_db, run

N_DAYS = 42
PERSONAS = {   # name -> (screen, social, sleep, steps, stress, connectedness, attendance, trend, categories to include)
    "steady":   dict(scr=3.8, soc=1.8, slp=7.6, stp=8800, strs=2, con=4, att=92, trend=0, cats="all"),
    "heavy":    dict(scr=8.0, soc=6.8, slp=5.1, stp=3600, strs=4, con=2, att=71, trend=0, cats="all"),
    "decline":  dict(scr=4.2, soc=2.2, slp=7.4, stp=8200, strs=2, con=4, att=90, trend=1, cats="all"),
    "partial":  dict(scr=5.0, soc=3.0, slp=6.8, stp=6500, strs=3, con=3, att=85, trend=0, cats=("screen", "physical")),
    "screen_only": dict(scr=6.0, soc=4.0, slp=0, stp=0, strs=3, con=3, att=80, trend=0, cats=("screen",)),
}


def frames(seed, p, end):
    rng = np.random.default_rng(seed)
    days = [end - timedelta(days=i) for i in range(N_DAYS - 1, -1, -1)]
    drop = lambda: rng.random() < 0.08            # a few missing days: gaps are kept, not filled
    f = {}
    for key in CATEGORIES: f[key] = []
    for i, d in enumerate(days):
        w = 1.0 + (0.9 * max(0, i - (N_DAYS - 8)) / 7 if p["trend"] else 0)     # last week worsens for the 'decline' persona
        ds = d.isoformat()
        if not drop():
            scr = float(np.clip(rng.normal(p["scr"] * (1 + 0.5 * (w - 1)), 0.6), 0.8, 14))
            soc = float(np.clip(rng.normal(p["soc"] * (1 + 0.6 * (w - 1)), 0.4), 0.2, scr))
            f["screen"].append((ds, round(scr, 1), round(soc, 1), round(max(0, (scr - soc) * 0.5), 1), round(max(0, (scr - soc) * 0.3), 1),
                                int(rng.normal(70 * w, 12)), round(float(np.clip(rng.normal(0.5 * w, 0.3), 0, 4)), 1), "Instagram"))
        if p["slp"] and not drop():
            slp = float(np.clip(rng.normal(p["slp"] - 1.6 * (w - 1), 0.6), 3.5, 10))
            f["physical"].append((ds, round(slp, 1), int(np.clip(round(rng.normal(3.6 - 1.2 * (w - 1), 0.8)), 1, 5)),
                                  int(max(300, rng.normal(p["stp"] / w, 1200))), round(max(0, rng.normal(40 / w, 15)), 0), int(rng.integers(2, 6)), None, None))
        if not drop():
            f["behaviour"].append((ds, round(float(np.clip(rng.normal(7 / w, 1), 0, 10)), 1), round(float(np.clip(rng.normal(6.5 / w, 1.2), 0, 10)), 1),
                                   rng.choice(["light", "moderate", "active"] if w < 1.3 else ["sedentary", "light"]), int(rng.integers(0, 2 if w < 1.3 else 4)),
                                   int(np.clip(round(rng.normal(3.6 / w, 0.7)), 1, 5)), int(np.clip(round(rng.normal(3.5 / w, 0.7)), 1, 5)),
                                   int(np.clip(round(rng.normal(p["strs"] * w, 0.7)), 1, 5)), int(np.clip(round(rng.normal(3.6 / w, 0.8)), 1, 5))))
        if not drop():
            f["social"].append((ds, int(max(0, rng.normal(9 / w, 2.5))), int(rng.random() < 0.3 / w), round(max(0, rng.normal(90 / w, 30)), 0),
                                int(np.clip(round(rng.normal(p["con"] / w, 0.7)), 1, 5)), int(np.clip(round(rng.normal(2.2 * w, 0.7)), 1, 5))))
        if d.weekday() < 5 and not drop():
            f["academic"].append((ds, None, round(float(np.clip(rng.normal(p["att"] / (1 + 0.2 * (w - 1)), 5), 0, 100)), 0),
                                  round(float(np.clip(rng.normal(80 / w, 10), 0, 100)), 0), round(max(0, rng.normal(3.2 / w, 1)), 1),
                                  int(np.clip(round(rng.normal(3, 0.7)), 1, 5)), int(np.clip(round(rng.normal(2.8 * w, 0.8)), 1, 5)), None))
    return f


def main():
    app = create_app()
    pw_student = os.environ.get("DEMO_PASSWORD") or ("Demo-" + secrets.token_urlsafe(9) + "7")
    pw_res = os.environ.get("DEMO_PASSWORD") or ("Demo-" + secrets.token_urlsafe(9) + "7")
    with app.app_context():
        db = get_db()
        if db.execute("SELECT 1 FROM users WHERE is_demo=1").fetchone():
            print("Demo data already exists. Delete instance/mindscreen.db to re-seed."); return
        end = date.today()
        accounts = [("demo.student@example.org", "decline", 22, "Female", "Undergraduate", pw_student)]
        for i, (name, a) in enumerate([("steady", 21), ("steady", 20), ("steady", 23), ("heavy", 21), ("heavy", 22), ("decline", 19), ("partial", 24), ("screen_only", 20)]):
            accounts.append((f"demo.{name}{i}@example.org", name, a, ["Female", "Male"][i % 2], ["Undergraduate", "Graduate"][i % 2], pw_student))
        for i, (email, persona, age, gender, level, pw) in enumerate(accounts):
            uid, pseudo = auth.create_user(email, pw, "student", age, gender, level, is_demo=1)
            auth.set_consent(uid, "core", True)
            p = PERSONAS[persona]
            fr = frames(100 + i, p, end)
            for key, spec in CATEGORIES.items():
                if p["cats"] != "all" and key not in p["cats"]:
                    continue
                names = [c["name"] for c in spec["columns"]]
                rows = [r for r in fr[key]]
                if not rows: continue
                df = pd.DataFrame(rows, columns=names)
                clean, rep = ingest.validate_frame(df, key, set())
                if clean is None:
                    print("seed validation problem", key, rep["errors"][:2]); continue
                ingest.insert_records(key, pseudo, None, clean, "demo", 1)
        # extra demo cohort members so aggregates reach the minimum group size
        for j in range(6):
            uid, pseudo = auth.create_user(f"demo.extra{j}@example.org", pw_student, "student", 20 + j % 4, ["Female", "Male"][j % 2], "Undergraduate", is_demo=1)
            auth.set_consent(uid, "core", True)
            fr = frames(300 + j, PERSONAS[["steady", "heavy", "decline"][j % 3]], end)
            for key, spec in CATEGORIES.items():
                df = pd.DataFrame(fr[key], columns=[c["name"] for c in spec["columns"]])
                clean, rep = ingest.validate_frame(df, key, set())
                if clean is not None: ingest.insert_records(key, pseudo, None, clean, "demo", 1)
        auth.create_user("demo.researcher@example.org", pw_res, "researcher", is_demo=1)
        # real predictions (real model, fictional inputs) for every demo student
        import predict_service as ps
        from db import q
        for u in q("SELECT * FROM users WHERE is_demo=1 AND role='student'"):
            ps.run_prediction(u, 30)
    print("\nDemo accounts created (FICTIONAL data, flagged as demo):")
    print(f"  student     demo.student@example.org      password: {pw_student}")
    print(f"  researcher  demo.researcher@example.org   password: {pw_res}")
    print("Other demo students use the same student password. Data is stored in instance/mindscreen.db.\n")


if __name__ == "__main__":
    main()
