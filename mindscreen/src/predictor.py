"""Inference engine: loads trained models, predicts risk, explains it with what-if analysis."""
import json, subprocess, sys
from pathlib import Path

import joblib
import pandas as pd

from features import CONTROLLABLE, RAW_COLUMNS, add_features

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

TIERS = [(0.35, "Low", "green"), (0.65, "Moderate", "amber"), (1.01, "High", "red")]


def tier_for(p: float):
    for cut, label, colour in TIERS:
        if p < cut:
            return label, colour


class Predictor:
    def __init__(self):
        try:
            self._load()
        except Exception:                      # missing files or sklearn version mismatch
            subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)
            self._load()

    def _load(self):
        self.schema = json.loads((MODELS / "schema.json").read_text())
        self.metrics = json.loads((MODELS / "metrics.json").read_text())
        self.importance = json.loads((MODELS / "feature_importance.json").read_text())
        self.models = {n: joblib.load(MODELS / f) for n, f in self.schema["model_files"].items()}
        self.best = self.schema["best_model"]

    # ---------- core ----------
    def _frame(self, rows) -> pd.DataFrame:
        return add_features(pd.DataFrame(rows)[RAW_COLUMNS])

    def clean(self, raw: dict) -> dict:
        """Validate / coerce a dict of user inputs; clamp numbers to the training range."""
        out = {}
        for c, (lo, hi) in self.schema["ranges"].items():
            try:
                v = float(raw.get(c, self.schema["defaults"][c]))
            except (TypeError, ValueError):
                v = self.schema["defaults"][c]
            out[c] = min(max(v, lo if c != "Age" else 16), hi if c != "Age" else 40)
        for c, opts in self.schema["categories"].items():
            v = raw.get(c)
            if v in opts:
                out[c] = v
            else:   # unknown platform -> "Other" (encoder ignores unseen values); others -> most common
                out[c] = "Other" if c == "Most_Used_Platform" else self.schema["defaults_cat"].get(c, opts[0])
        return out

    def risk(self, rows, model=None) -> list:
        return self.models[model or self.best].predict_proba(self._frame(rows))[:, 1].tolist()

    def predict(self, raw: dict) -> dict:
        x = self.clean(raw)
        p = self.risk([x])[0]
        label, colour = tier_for(p)
        votes = {}
        for n, m in self.models.items():
            pn = float(m.predict_proba(self._frame([x]))[:, 1][0])
            votes[n] = {"risk": round(pn * 100, 1), "at_risk": pn >= 0.5}
        agree = sum(v["at_risk"] for v in votes.values())
        return {
            "inputs": x, "risk_pct": round(p * 100, 1), "tier": label, "colour": colour,
            "votes": votes, "agreement": f"{max(agree, len(votes) - agree)}/{len(votes)}",
            "consensus": "At risk" if agree > len(votes) / 2 else ("Healthy" if agree < len(votes) / 2 else "Split"),
            "whatif": self.whatif(x, p), "model_used": self.best,
        }

    # ---------- explainability: counterfactual "what-if" ----------
    def whatif(self, x: dict, base_p: float):
        ref, out = self.schema["reference"], []
        for col, nice in CONTROLLABLE.items():
            worse = x[col] > ref[col] if col != "Sleep_Hours_Per_Night" else x[col] < ref[col]
            if not worse:
                continue
            y = dict(x); y[col] = ref[col]
            p2 = self.risk([y])[0]
            out.append({"feature": nice, "now": round(x[col], 1), "target": round(ref[col], 1),
                        "new_risk": round(p2 * 100, 1), "drop": round((base_p - p2) * 100, 1)})
        out.sort(key=lambda d: d["drop"], reverse=True)
        return out

    # ---------- institution / batch ----------
    def batch(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        rows = [self.clean(r) for r in df[RAW_COLUMNS].to_dict("records")]
        p = self.risk(rows)
        res = df.copy()
        res["risk_pct"] = [round(v * 100, 1) for v in p]
        res["risk_tier"] = [tier_for(v)[0] for v in p]
        return res
