"""CSV and PDF reports for one authorised student, and k-anonymised aggregate CSV for researchers."""
import csv, io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from categories import CATEGORIES
import features_unified as fu

DISCLAIMER = ("MindScreen is an educational/research screening and decision-support aid. It is NOT a medical diagnosis. "
              "Model outputs describe statistical patterns learned from a public survey dataset and have not been clinically validated.")


def fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def build_context(user, prediction):
    pid = user["pseudo_id"]
    status = fu.category_status(pid)
    feats = fu.unified_features(pid, 30)
    return {"pid": pid, "status": status, "completeness": fu.overall_completeness(status), "features": feats,
            "prediction": prediction, "warnings": fu.quality_warnings(pid, status), "generated": datetime.now().strftime("%d %b %Y, %H:%M")}


def trend_rows(ctx):
    f = ctx["features"]
    rows = []
    if f.get("screen"):
        rows += [("Screen time (avg h/day)", fmt(f["screen"]["total_h"])), ("Social-media time (avg h/day)", fmt(f["screen"]["social_h"])),
                 ("Night-time screen use (avg h)", fmt(f["screen"]["night_h"])), ("Phone pick-ups (avg/day)", fmt(f["screen"]["pickups"], 0))]
        rows += [("Sleep (avg h/night)", fmt(f["physical"]["sleep_h"])), ("Sleep variability (SD, h)", fmt(f["physical"]["sleep_sd"], 2)),
                 ("Steps (avg/day)", fmt(f["physical"]["steps"], 0)), ("Exercise (avg min)", fmt(f["physical"]["exercise_min"], 0)),
                 ("Study routine consistency (0-10)", fmt(f["behaviour"]["study_routine_consistency"])),
                 ("Social connectedness, self-reported (1-5)", fmt(f["social"]["connectedness_self"])),
                 ("Academic workload (1-5)", fmt(f["academic"]["workload_level"]))]
    return rows


def csv_report(ctx) -> str:
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["MindScreen report (screening aid, not a diagnosis)"]); w.writerow(["Pseudonymous ID", ctx["pid"]]); w.writerow(["Generated", ctx["generated"]])
    w.writerow([]); w.writerow(["Data completeness (%)", ctx["completeness"]])
    w.writerow([]); w.writerow(["Category", "Status", "Days of data", "First date", "Last date"])
    for k, s in ctx["status"].items():
        w.writerow([CATEGORIES[k]["label"], s["status"], s["days"], s["first"] or "", s["last"] or ""])
    w.writerow([]); w.writerow(["Trend summary (last 30 days)", "Value"])
    for a, b in trend_rows(ctx): w.writerow([a, b])
    p = ctx["prediction"]; w.writerow([]); w.writerow(["Model output"])
    if p and p.get("available", True) and p.get("probability") is not None:
        w.writerow(["Model version", p["model_version"]]); w.writerow(["Prediction time (UTC)", p["ts"]])
        w.writerow(["Estimated likelihood of low self-reported wellbeing (%)", p["probability"]]); w.writerow(["Screening category", p["tier"]])
        w.writerow(["Data sufficiency", p["sufficiency"]])
        w.writerow([]); w.writerow(["Explanation: feature", "value", "effect on estimate (points)"])
        for i in p["explanation"]["items"]: w.writerow([i["label"], i["value"], i["effect_pp"]])
    else:
        w.writerow(["No prediction available for this record."])
    w.writerow([]); w.writerow(["Disclaimer", DISCLAIMER])
    return out.getvalue()


def pdf_report(ctx) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="MindScreen report")
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Heading2"], textColor=colors.HexColor("#0b1f3a"), spaceBefore=10)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
    note = ParagraphStyle("N", parent=B, backColor=colors.HexColor("#fff8e6"), borderPadding=6, fontSize=8.5)
    def tbl(data, widths=None):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f2f2")),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3dd")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t
    el = [Paragraph("MindScreen: Student Well-being Screening Report", ss["Title"]),
          Paragraph(f"Pseudonymous ID: <b>{ctx['pid']}</b> &nbsp;·&nbsp; Generated {ctx['generated']}", B), Spacer(1, 6),
          Paragraph(DISCLAIMER, note), Paragraph("Data completeness", H),
          Paragraph(f"Overall completeness: <b>{ctx['completeness']}%</b> (average over five categories of min(days submitted, 14)/14).", B),
          tbl([["Category", "Status", "Days", "First", "Last"]] + [[CATEGORIES[k]["label"], s["status"], s["days"], s["first"] or "-", s["last"] or "-"] for k, s in ctx["status"].items()])]
    el += [Paragraph("Selected trends (last 30 days)", H), tbl([["Measure", "Average"]] + [list(r) for r in trend_rows(ctx)], [100 * mm, 40 * mm])]
    p = ctx["prediction"]
    el.append(Paragraph("Model output", H))
    if p and p.get("probability") is not None:
        el += [Paragraph(f"Screening category: <b>{p['tier']}</b> indicator &nbsp;·&nbsp; model-estimated likelihood of low self-reported wellbeing: <b>{p['probability']}%</b>.", B),
               Paragraph(f"Model version: {p['model_version']} &nbsp;·&nbsp; predicted {p['ts']} UTC &nbsp;·&nbsp; data sufficiency: {p['sufficiency']} ({p['n_days']} days).", B),
               Paragraph("Categories used by the model: Screen time, Physical wellness. Collected but not used by the validated model: "
                         + (", ".join(CATEGORIES[k]["label"] for k in p["categories_collected_not_used"]) or "none") + ".", B),
               Paragraph("Explainability", H), Paragraph(p["explanation"]["text"], B),
               tbl([["Feature", "Value", "Effect on estimate"]] + [[i["label"], i["value"], f"{i['effect_pp']:+.1f} pts"] for i in p["explanation"]["items"]], [80 * mm, 35 * mm, 35 * mm])]
    else:
        el.append(Paragraph("No prediction is available for this record (insufficient validated data).", B))
    el.append(Paragraph("Limitations", H))
    for t in ["The model was trained on a public survey dataset (self-reported wellbeing label), not on clinical outcomes.",
              "Only screen-time and sleep (plus profile fields) enter the validated model; other categories are shown for context.",
              "Explanations describe model behaviour, not causes. Low attendance, grades or activity do not by themselves indicate a condition.",
              "Students can use the platform without being labelled or penalised. A professional assessment is needed for any decision."]:
        el.append(Paragraph("• " + t, B))
    if ctx["warnings"]:
        el += [Paragraph("Data-quality notes", H)] + [Paragraph("• " + w["text"], B) for w in ctx["warnings"][:8]]
    doc.build(el)
    return buf.getvalue()
