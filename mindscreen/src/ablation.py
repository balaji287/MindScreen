"""Reference experiment: what if we ALSO used the survey's self-reported items?
Shows the price of removing questionnaire inputs.   Run: python src/ablation.py -> models/ablation.json"""
import json, warnings
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import ROOT, SEED, AT_RISK_THRESHOLD, model_zoo
from features import CAT, NUM, RAW_COLUMNS, SELF_REPORTED_CAT, SELF_REPORTED_NUM, add_features

warnings.filterwarnings("ignore")
df = pd.read_csv(ROOT / "data" / "students_behaviour.csv")
y = (df["Mental_Health_Score"] <= AT_RISK_THRESHOLD).astype(int)
extra = SELF_REPORTED_NUM + SELF_REPORTED_CAT
X = add_features(df[RAW_COLUMNS + extra])
num, cat = NUM + SELF_REPORTED_NUM, CAT + SELF_REPORTED_CAT
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
out = {}
for name, (est, grid) in model_zoo().items():
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    gs = GridSearchCV(Pipeline([("p", pre), ("m", est)]), grid, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
    p = gs.best_estimator_.predict(Xte)
    out[name] = {"accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p),
                 "recall": recall_score(yte, p), "f1": f1_score(yte, p)}
    print(f"{name:20s}", {k: round(v, 3) for k, v in out[name].items()})
(ROOT / "models" / "ablation.json").write_text(json.dumps(out, indent=2))
