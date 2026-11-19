"""CSV / manual-entry validation and storage. Original files are kept separately from cleaned records."""
import io, re, secrets
from datetime import date as _date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from categories import CATEGORIES
from db import get_db, run

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 2000


def _period_to_date(p: str):
    p = str(p).strip()
    m = re.fullmatch(r"(\d{4})-W(\d{1,2})", p)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}-W{int(m.group(2)):02d}-1", "%G-W%V-%u").date().isoformat()
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{4})-(\d{2})", p)
    return f"{m.group(1)}-{m.group(2)}-01" if m else None


def validate_frame(df: pd.DataFrame, key: str, consents: set):
    """Return (clean_df, report). report = {errors:[{row,col,msg}], warnings:[str], rows_total, rows_valid}."""
    spec = CATEGORIES[key]
    cols = {c["name"]: c for c in spec["columns"]}
    errors, warnings = [], []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rep = {"errors": errors, "warnings": warnings, "rows_total": int(len(df)), "rows_valid": 0}

    if len(df) == 0:
        errors.append({"row": 0, "col": "", "msg": "The file has no data rows."}); return None, rep
    if len(df) > MAX_ROWS:
        errors.append({"row": 0, "col": "", "msg": f"Too many rows ({len(df)}); the limit is {MAX_ROWS}."}); return None, rep

    unknown = [c for c in df.columns if c not in cols]
    if unknown:
        warnings.append("Ignored unknown column(s): " + ", ".join(unknown[:8]))
    if "date" not in df.columns and not ("period" in df.columns and "period" in cols):
        errors.append({"row": 0, "col": "date", "msg": "Missing required column: date" +
                       (" (or period)" if "period" in cols else "")}); return None, rep
    required = [c["name"] for c in spec["columns"] if c["required"] and c["name"] != "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append({"row": 0, "col": ",".join(missing), "msg": "Missing required column(s): " + ", ".join(missing)}); return None, rep
    for c in spec["columns"]:
        if c["consent"] and c["name"] in df.columns and df[c["name"]].notna().any() and c["consent"] not in consents:
            errors.append({"row": 0, "col": c["name"],
                           "msg": f"'{c['name']}' needs your separate '{c['consent']}' consent (Privacy & Consent page). "
                                  "Remove the column or grant consent."})
    if errors:
        return None, rep

    out = pd.DataFrame(index=df.index)
    # dates
    raw_dates = df["date"] if "date" in df.columns else pd.Series([None] * len(df), index=df.index)
    dates = pd.to_datetime(raw_dates, errors="coerce", format="%Y-%m-%d")
    if "period" in df.columns and "period" in cols:
        fallback = df["period"].map(lambda p: _period_to_date(p) if pd.notna(p) else None)
        dates = dates.fillna(pd.to_datetime(fallback, errors="coerce"))
    row_bad = pd.Series(False, index=df.index)
    for i in df.index[dates.isna()]:
        errors.append({"row": int(i) + 1, "col": "date", "msg": "Invalid or missing date (use YYYY-MM-DD)."}); row_bad[i] = True
    today = pd.Timestamp(_date.today())
    for i in df.index[(dates > today + pd.Timedelta(days=1)) & dates.notna()]:
        errors.append({"row": int(i) + 1, "col": "date", "msg": "Date is in the future."}); row_bad[i] = True
    out["date"] = dates.dt.strftime("%Y-%m-%d")
    dup = out["date"].duplicated(keep=False) & dates.notna()
    for i in df.index[dup]:
        errors.append({"row": int(i) + 1, "col": "date", "msg": f"Duplicate date {out['date'][i]} in the file."}); row_bad[i] = True

    # values
    for name, c in cols.items():
        if name == "date" or name not in df.columns:
            if name != "date":
                out[name] = np.nan if c["kind"] in ("num", "int") else None
            continue
        s = df[name]
        if c["kind"] in ("num", "int"):
            num = pd.to_numeric(s, errors="coerce")
            for i in df.index[s.notna() & num.isna() & (s.astype(str).str.strip() != "")]:
                errors.append({"row": int(i) + 1, "col": name, "msg": f"'{s[i]}' is not a number."}); row_bad[i] = True
            if c["lo"] is not None:
                for i in df.index[num.notna() & ((num < c["lo"]) | (num > c["hi"]))]:
                    errors.append({"row": int(i) + 1, "col": name, "msg": f"{num[i]} is outside the allowed range {c['lo']}-{c['hi']}."}); row_bad[i] = True
            if c["kind"] == "int":
                for i in df.index[num.notna() & (num % 1 != 0)]:
                    errors.append({"row": int(i) + 1, "col": name, "msg": f"{num[i]} must be a whole number."}); row_bad[i] = True
            out[name] = num
        elif c["kind"] == "cat":
            v = s.astype(str).str.strip().str.lower().where(s.notna() & (s.astype(str).str.strip() != ""), None)
            for i in df.index[v.notna() & ~v.isin(c["choices"])]:
                errors.append({"row": int(i) + 1, "col": name, "msg": f"'{v[i]}' is not one of {', '.join(c['choices'])}."}); row_bad[i] = True
            out[name] = v
        else:  # text
            v = s.astype(str).str.strip().where(s.notna() & (s.astype(str).str.strip() != ""), None)
            v = v.map(lambda x: x.lstrip("=+-@") if isinstance(x, str) else x)   # neutralise spreadsheet-formula injection
            for i in df.index[v.map(lambda x: isinstance(x, str) and len(x) > 30)]:
                errors.append({"row": int(i) + 1, "col": name, "msg": "Text longer than 30 characters."}); row_bad[i] = True
            out[name] = v

    if key == "screen":
        cat_sum = out[["social_h", "entertainment_h", "productivity_h"]].sum(axis=1, min_count=1)
        for i in df.index[out.total_screen_h.notna() & cat_sum.notna() & (cat_sum > out.total_screen_h + 1.0)]:
            errors.append({"row": int(i) + 1, "col": "total_screen_h", "msg": "Category hours add up to more than the total."}); row_bad[i] = True
    data_cols = [n for n in cols if n not in ("date", "period")]
    empty = out[data_cols].isna().all(axis=1) if len(data_cols) else pd.Series(False, index=out.index)
    for i in df.index[empty & ~row_bad]:
        errors.append({"row": int(i) + 1, "col": "", "msg": "Row has a date but no values."}); row_bad[i] = True

    miss_share = out[data_cols].isna().mean()
    for n, s in miss_share.items():
        if 0.3 < s < 1 and n in df.columns:
            warnings.append(f"'{n}' is empty in {s:.0%} of rows (gaps are kept as missing, never filled in).")
    rep["rows_valid"] = int((~row_bad).sum())
    clean = out[~row_bad].drop(columns=["period"], errors="ignore")
    return (clean if not errors else None), rep


def read_csv_bytes(raw: bytes):
    if len(raw) > MAX_BYTES:
        raise ValueError(f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).")
    if b"\x00" in raw[:4096]:
        raise ValueError("This does not look like a text CSV file.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("File must be UTF-8 encoded CSV.")
    try:
        return pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
    except Exception:
        raise ValueError("Could not read the file as CSV.")


def store_original(upload_dir: Path, pseudo_id: str, raw: bytes) -> str:
    d = upload_dir / pseudo_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / (secrets.token_hex(12) + ".csv")
    p.write_bytes(raw)
    return str(p)


def insert_records(key: str, pseudo_id: str, upload_id, clean: pd.DataFrame, source="upload", is_demo=0, replace_existing=True):
    spec = CATEGORIES[key]
    names = [c["name"] for c in spec["columns"] if c["name"] != "date"]
    verb = "INSERT OR REPLACE" if replace_existing else "INSERT OR IGNORE"
    sql = (f"{verb} INTO {spec['table']}(pseudo_id,upload_id,date,{','.join(names)},source,is_demo) "
           f"VALUES({','.join('?' * (5 + len(names)))})")
    db = get_db()
    n = 0
    kinds = {c["name"]: c["kind"] for c in spec["columns"]}
    for _, r in clean.iterrows():
        vals = []
        for c in names:
            v = r[c] if c in r.index else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(None)
            elif kinds[c] == "int":
                vals.append(int(v))
            elif kinds[c] == "num":
                vals.append(float(v))
            else:
                vals.append(str(v))
        db.execute(sql, [pseudo_id, upload_id, r["date"], *vals, source, is_demo]); n += 1
    db.commit()
    return n
