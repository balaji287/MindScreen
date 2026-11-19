"""Train + evaluate 4 ML models on student behaviour data.
Run:  python src/train.py
Outputs -> models/*.joblib, models/metrics.json, models/schema.json, reports/*.png
"""
import json, warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from features import RAW_CAT, RAW_COLUMNS, RAW_NUM, add_features, build_preprocessor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42
AT_RISK_THRESHOLD = 5   # self-reported wellbeing score (1-10) <= 5  ->  "at risk"


def load():
    df = pd.read_csv(ROOT / "data" / "students_behaviour.csv")
    y = (df["Mental_Health_Score"] <= AT_RISK_THRESHOLD).astype(int)
    X = add_features(df[RAW_COLUMNS])
    return df, X, y


def model_zoo():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"m__C": [0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
            {"m__n_estimators": [200, 400], "m__max_depth": [None, 8], "m__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}),
        "SVM (RBF)": (
            CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"), method="sigmoid", cv=3),
            {"m__estimator__C": [1, 10], "m__estimator__gamma": ["scale", 0.05]}),
    }


def main():
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "static" / "reports").mkdir(parents=True, exist_ok=True)
    df, X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Rows: {len(df)} | train {len(Xtr)} / test {len(Xte)} | at-risk share {y.mean():.1%}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted, probas = {}, {}, {}
    for name, (est, grid) in model_zoo().items():
        pipe = Pipeline([("p", build_preprocessor()), ("m", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
        best = gs.best_estimator_
        cvr = cross_validate(best, Xtr, ytr, cv=cv,
                             scoring=["accuracy", "precision", "recall", "f1"])
        pred, proba = best.predict(Xte), best.predict_proba(Xte)[:, 1]
        results[name] = {
            "test_accuracy": accuracy_score(yte, pred),
            "test_precision": precision_score(yte, pred),
            "test_recall": recall_score(yte, pred),
            "test_f1": f1_score(yte, pred),
            "test_roc_auc": roc_auc_score(yte, proba),
            "confusion": confusion_matrix(yte, pred).tolist(),
            "cv_accuracy": cvr["test_accuracy"].mean(),
            "cv_precision": cvr["test_precision"].mean(),
            "cv_f1": cvr["test_f1"].mean(),
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in gs.best_params_.items()},
        }
        fitted[name], probas[name] = best, (pred, proba)
        r = results[name]
        print(f"{name:20s} acc {r['test_accuracy']:.3f} prec {r['test_precision']:.3f} "
              f"rec {r['test_recall']:.3f} f1 {r['test_f1']:.3f} auc {r['test_roc_auc']:.3f} "
              f"| 5-fold CV acc {r['cv_accuracy']:.3f}")
        joblib.dump(best, ROOT / "models" / (name.split()[0].lower() + ".joblib"))

    best_name = max(results, key=lambda k: results[k]["test_f1"])
    print("Best model (by held-out F1):", best_name)

    # ---- Subgroup check on the held-out test set (suppressed when a group is too small) ----
    bp = probas[best_name][0]
    sub = {"model": best_name, "min_group_size": 20, "groups": {}}
    for col in ["Gender", "Academic_Level"]:
        for g in sorted(Xte[col].unique()):
            mask = (Xte[col] == g).values
            n = int(mask.sum())
            if n < 20:
                sub["groups"][f"{col}: {g}"] = {"n": n, "suppressed": True}
                continue
            yt, yp = yte.values[mask], bp[mask]
            sub["groups"][f"{col}: {g}"] = {
                "n": n, "positives": int(yt.sum()), "suppressed": False,
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0))}
    (ROOT / "models" / "subgroups.json").write_text(json.dumps(sub, indent=2))

    # ---- "Healthy reference" profile for what-if advice (median of not-at-risk students)
    healthy = X[y == 0]
    reference = {c: float(healthy[c].median()) for c in
                 ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]}
    schema = {
        "best_model": best_name,
        "model_files": {n: n.split()[0].lower() + ".joblib" for n in results},
        "reference": reference,
        "categories": {c: sorted(df[c].unique().tolist()) for c in RAW_CAT},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in RAW_NUM},
        "defaults": {c: float(df[c].median()) for c in RAW_NUM},
        "defaults_cat": {c: df[c].mode()[0] for c in RAW_CAT},
        "n_rows": int(len(df)), "at_risk_share": float(y.mean()),
        "threshold": AT_RISK_THRESHOLD,
        "test_size": int(len(Xte)),
    }
    (ROOT / "models" / "schema.json").write_text(json.dumps(schema, indent=2))
    (ROOT / "models" / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---- Plots ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.2; xs = np.arange(len(results))
    for i, m in enumerate(metrics):
        vals = [results[n][m] * 100 for n in results]
        bars = ax.bar(xs + i * w, vals, w, label=m.replace("test_", "").title())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .5, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(90, ls="--", c="grey", lw=1); ax.set_ylim(60, 104)
    ax.set_xticks(xs + 1.5 * w); ax.set_xticklabels(list(results)); ax.set_ylabel("%")
    ax.set_title("Held-out test performance (20% unseen students)"); ax.legend(ncol=4, loc="lower center")
    fig.tight_layout(); fig.savefig(ROOT / "static" / "reports" / "model_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (n, (pred, _)) in zip(axes, probas.items()):
        ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Healthy", "At risk"],
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(n, fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "static" / "reports" / "confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for n, (_, pr) in probas.items():
        fpr, tpr, _ = roc_curve(yte, pr)
        ax.plot(fpr, tpr, label=f"{n} ({results[n]['test_roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(ROOT / "static" / "reports" / "roc_curves.png", dpi=150); plt.close(fig)

    pi = permutation_importance(fitted[best_name], Xte, yte, scoring="f1", n_repeats=15, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); imp.plot.barh(ax=ax, color="#4f7cac")
    ax.set_xlabel("Drop in F1 when feature is shuffled"); ax.set_title(f"What drives the prediction ({best_name})")
    fig.tight_layout(); fig.savefig(ROOT / "static" / "reports" / "feature_importance.png", dpi=150); plt.close(fig)
    (ROOT / "models" / "feature_importance.json").write_text(
        json.dumps(imp.sort_values(ascending=False).round(4).to_dict(), indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    df.boxplot(column="Avg_Daily_Usage_Hours", by="Mental_Health_Score", ax=axes[0], grid=False)
    axes[0].set_title("Screen time vs wellbeing score"); axes[0].set_xlabel("Wellbeing score"); axes[0].set_ylabel("Hours / day")
    df.boxplot(column="Sleep_Hours_Per_Night", by="Mental_Health_Score", ax=axes[1], grid=False)
    axes[1].set_title("Sleep vs wellbeing score"); axes[1].set_xlabel("Wellbeing score"); axes[1].set_ylabel("Hours / night")
    g = df.assign(risk=y.map({0: "Healthy", 1: "At risk"})).groupby(["Most_Used_Platform", "risk"]).size().unstack(fill_value=0)
    g = g.loc[g.sum(axis=1).sort_values(ascending=False).index[:6]]
    (g.div(g.sum(axis=1), axis=0) * 100)["At risk"].plot.bar(ax=axes[2], color="#c0504d")
    axes[2].set_title("% at-risk by main platform (top 6)"); axes[2].set_xlabel(""); axes[2].set_ylabel("%")
    plt.suptitle(""); fig.tight_layout(); fig.savefig(ROOT / "static" / "reports" / "eda.png", dpi=150); plt.close(fig)
    print("Saved models + reports.")


if __name__ == "__main__":
    main()
