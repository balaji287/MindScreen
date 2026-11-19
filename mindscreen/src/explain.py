"""Explainability: local SHAP values for the current prediction + global importance from training.
Explanations describe the MODEL's calculation, not causes."""
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features import RAW_COLUMNS, add_features

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"Age": "Age", "Avg_Daily_Usage_Hours": "Social-media time (avg h/day)", "Sleep_Hours_Per_Night": "Sleep (avg h/night)",
          "Gender": "Gender", "Academic_Level": "Academic level", "Most_Used_Platform": "Most-used platform",
          "sleep_deficit": "Sleep deficit", "usage_sleep_ratio": "Usage-to-sleep ratio", "screen_share_of_waking": "Screen share of waking hours"}
_cache = {}


def _background():
    if "bg" not in _cache:
        df = pd.read_csv(ROOT / "data" / "students_behaviour.csv")[RAW_COLUMNS]
        _cache["bg"] = df.sample(60, random_state=1).reset_index(drop=True)
    return _cache["bg"]


def _fmt(col, v):
    return f"{v:.1f}" if isinstance(v, (int, float, np.floating)) and col != "Age" else str(int(v) if col == "Age" else v)


def local_explanation(model, inputs: dict) -> dict:
    bg = _background()
    num = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]

    def f(a):
        X = pd.DataFrame(a, columns=RAW_COLUMNS)
        for c in num:
            X[c] = X[c].astype(float)
        return model.predict_proba(add_features(X))[:, 1]

    x = np.array([[inputs[c] for c in RAW_COLUMNS]], dtype=object)
    method, base, vals = None, None, None
    try:
        import shap
        warnings.filterwarnings("ignore")
        ex = shap.KernelExplainer(f, bg.values.astype(object))
        np.random.seed(0)
        vals = np.asarray(ex.shap_values(x, nsamples=300, silent=True)).reshape(-1)
        base = float(np.ravel(ex.expected_value)[0]); method = "SHAP (KernelExplainer, 60-student background sample)"
    except Exception:
        # fallback: swap one feature at a time with its typical value (approximate, not additive)
        out = float(f(x)[0]); base = float(f(bg.values.astype(object)).mean()); vals = []
        for i, c in enumerate(RAW_COLUMNS):
            y = x.copy(); y[0, i] = bg[c].median() if c in num else bg[c].mode()[0]
            vals.append(out - float(f(y)[0]))
        vals = np.array(vals); method = "Approximate one-at-a-time attribution (SHAP not installed)"
    out_p = float(f(x)[0])
    items = [{"feature": c, "label": LABELS.get(c, c), "value": _fmt(c, inputs[c]), "effect_pp": round(100 * float(v), 1),
              "direction": "raises" if v > 0.0005 else ("lowers" if v < -0.0005 else "no effect")}
             for c, v in zip(RAW_COLUMNS, vals)]
    items.sort(key=lambda d: abs(d["effect_pp"]), reverse=True)
    return {"method": method, "base_pct": round(100 * base, 1), "output_pct": round(100 * out_p, 1), "items": items,
            "text": plain_text(items, base, out_p)}


def plain_text(items, base, out):
    up = [i for i in items if i["direction"] == "raises"][:2]
    down = [i for i in items if i["direction"] == "lowers"][:2]
    s = [f"For a typical student in the training survey the model's estimate is about {100 * base:.0f}%; for this record it is {100 * out:.0f}%."]
    if up:
        s.append("Pushing the estimate up: " + "; ".join(f"{i['label'].lower()} ({i['value']}, about +{abs(i['effect_pp']):.0f} points)" for i in up) + ".")
    if down:
        s.append("Pulling it down: " + "; ".join(f"{i['label'].lower()} ({i['value']}, about −{abs(i['effect_pp']):.0f} points)" for i in down) + ".")
    s.append("These are the model's statistical contributions learned from a survey dataset. They are not proof that any habit caused a condition.")
    return " ".join(s)


def global_importance(top=8) -> list:
    p = ROOT / "models" / "feature_importance.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return [{"label": LABELS.get(k, k), "value": round(v, 4)} for k, v in list(d.items())[:top]]
