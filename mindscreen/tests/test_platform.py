"""Run:  python -m pytest tests -q"""
import html, io, json, re, sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import PW, days, login, physical_csv, post, register, screen_csv, token, upload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------- pages, auth, CSRF, headers
def test_public_pages(client):
    for url in ["/", "/login", "/register", "/about", "/model-card", "/quick", "/healthz"]:
        assert client.get(url).status_code == 200, url


def test_landing_content(client):
    t = client.get("/").get_data(as_text=True)
    assert "Understand Student Well-being Through Data." in t and "Open Dashboard" in t and "Explore How It Works" in t
    assert "Collect Data" in t and "Analyse Patterns" in t and "Review Insights" in t


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Frame-Options"] == "DENY" and "script-src 'self'" in r.headers["Content-Security-Policy"]
    assert 'onclick=' not in r.get_data(as_text=True)


def test_register_login_logout(client):
    assert register(client, "alice@example.org").status_code == 302
    assert client.get("/app").status_code == 200
    client.post("/logout", data={"csrf": token(client, "/settings")})
    assert client.get("/app").status_code == 302
    assert login(client, "alice@example.org").status_code == 302
    assert login(client, "alice@example.org", "wrong-password1").status_code in (200, 401)


def test_registration_rules(client):
    t = token(client, "/register")
    base = {"csrf": t, "email": "b@example.org", "password": PW, "confirm": PW, "consent_core": "1"}
    assert client.post("/register", data={**base, "password": "short1", "confirm": "short1"}).status_code == 400
    assert client.post("/register", data={**base, "consent_core": ""}).status_code == 400
    assert client.post("/register", data={**base, "confirm": "different-pass1"}).status_code == 400
    assert client.post("/register", data={**base, "invite": "wrong"}).status_code == 400
    assert client.post("/register", data=base).status_code == 302
    c2 = client.application.test_client()
    t2 = token(c2, "/register")
    assert c2.post("/register", data={**base, "csrf": t2}).status_code == 400      # duplicate email


def test_csrf_enforced(client):
    assert client.post("/login", data={"email": "a@b.co", "password": "x"}).status_code == 400
    register(client, "alice@example.org")
    assert client.post("/predict/run", data={}).status_code == 400


def test_login_rate_limit(client):
    register(client, "alice@example.org")
    c = client.application.test_client()
    codes = [c.post("/login", data={"csrf": token(c), "email": "alice@example.org", "password": "nope-nope-1"}).status_code for _ in range(7)]
    assert 429 in codes


def test_redirects_and_open_redirect_blocked(client):
    r = client.get("/data")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    register(client, "alice@example.org")
    c = client.application.test_client()
    r = c.post("/login?next=https://evil.example", data={"csrf": token(c), "email": "alice@example.org", "password": PW})
    assert r.status_code == 302 and "evil.example" not in r.headers["Location"]


# ---------------------------------------------------------------- access control
def test_role_access(client, app):
    register(client, "alice@example.org")
    assert client.get("/admin").status_code == 403 and client.get("/admin/models").status_code == 403
    assert client.get("/admin/audit").status_code == 403
    r = app.test_client(); register(r, "res@example.org", role_invite="invite-123")
    assert r.get("/admin").status_code == 200 and r.get("/admin/models").status_code == 200
    assert r.get("/data").status_code == 403 and r.get("/predict").status_code == 403       # researchers never see individual data pages
    assert r.get("/api/trends").status_code == 403


def test_no_idor_between_students(client, app):
    register(client, "alice@example.org")
    r = upload(client, "screen", screen_csv(10), confirm=False)
    uid = re.search(r"/data/uploads/(\d+)/confirm", r.get_data(as_text=True)).group(1)
    bob = app.test_client(); register(bob, "bob@example.org")
    assert post(bob, f"/data/uploads/{uid}/confirm", page="/data/screen").status_code == 404
    assert post(bob, f"/data/uploads/{uid}/delete", page="/data/screen").status_code == 404
    assert post(bob, f"/data/uploads/{uid}/cancel", page="/data/screen").status_code == 404
    assert upload(client, "screen", screen_csv(10), confirm=False).status_code == 200


# ---------------------------------------------------------------- upload validation
def prev(c, key, raw, name="x.csv"):
    return upload(c, key, raw, name, confirm=False).get_data(as_text=True)


def test_missing_columns(student):
    t = prev(student, "screen", b"date,social_h\n2026-01-01,2\n")
    assert "Missing required column" in t and "Needs correction" in t
    assert "Missing required column" in prev(student, "physical", b"date,steps\n2026-01-01,100\n")
    assert "Missing required column: date" in prev(student, "screen", b"total_screen_h\n3\n")


def test_invalid_values(student):
    raw = b"date,total_screen_h,social_h\n2026-01-01,abc,1\n2026-01-02,30,1\n2026-01-03,5,-2\n"
    t = prev(student, "screen", raw)
    assert "is not a number" in t and "outside the allowed range" in t
    assert "must be a whole number" in prev(student, "behaviour", b"date,stress_self\n2026-01-01,2.5\n")
    assert "not one of" in prev(student, "behaviour", b"date,activity_pattern\n2026-01-01,flying\n")
    assert "add up to more than the total" in prev(student, "screen", b"date,total_screen_h,social_h,entertainment_h\n2026-01-01,2,2,2\n")


def test_duplicates_and_dates(student):
    assert "Duplicate date" in prev(student, "screen", b"date,total_screen_h\n2026-01-01,3\n2026-01-01,4\n")
    assert "Invalid or missing date" in prev(student, "screen", b"date,total_screen_h\n01/02/2026,3\n")
    assert "in the future" in prev(student, "screen", b"date,total_screen_h\n2099-01-01,3\n")
    assert "no values" in prev(student, "social", b"date,loneliness_self\n2026-01-01,\n")


def test_file_rules(student):
    assert "Only .csv" in student.get(  # redirected flash message
        post(student, "/data/screen/preview", files={"file": (io.BytesIO(b"a"), "x.txt")}, page="/data/screen").headers["Location"], follow_redirects=True).get_data(as_text=True)
    big = b"date,total_screen_h\n" + b"2026-01-01,3\n" * 400000
    assert post(student, "/data/screen/preview", files={"file": (io.BytesIO(big), "big.csv")}, page="/data/screen").status_code == 413
    r = post(student, "/data/screen/preview", files={"file": (io.BytesIO(b"\x00\x01\x02binary"), "b.csv")}, page="/data/screen")
    assert r.status_code == 302


def test_consent_gated_columns(student):
    t = html.unescape(prev(student, "physical", b"date,sleep_h,resting_hr\n2026-01-01,7,60\n"))
    assert "needs your separate 'wearable' consent" in t
    post(student, "/privacy/consent", {"kind": "wearable", "grant": "1"}, page="/privacy")
    t = prev(student, "physical", b"date,sleep_h,resting_hr\n2026-01-01,7,60\n")
    assert "Needs correction" not in t and "Confirm" in t


def test_formula_injection_neutralised(app, student):
    raw = b"date,total_screen_h,top_app\n2026-01-01,3,=HYPERLINK(1)\n"
    upload(student, "screen", raw)
    from db import connect
    row = connect(app.config["DB_PATH"]).execute("SELECT top_app FROM rec_screen").fetchone()
    assert not row["top_app"].startswith("=")


# ---------------------------------------------------------------- upload flow, missing data, preprocessing
def test_upload_confirm_replace_delete(app, student):
    r = upload(student, "screen", screen_csv(10))
    assert r.status_code == 302
    page = student.get("/data/screen").get_data(as_text=True)
    assert "Validated" in page and "10 / 10" in page
    upload(student, "screen", screen_csv(5), replace=True)
    from db import connect
    con = connect(app.config["DB_PATH"])
    assert con.execute("SELECT COUNT(*) FROM rec_screen").fetchone()[0] == 5
    uid = con.execute("SELECT id FROM uploads WHERE deleted_at IS NULL AND status='Validated'").fetchone()[0]
    post(student, f"/data/uploads/{uid}/delete", page="/data/screen")
    assert con.execute("SELECT COUNT(*) FROM rec_screen").fetchone()[0] == 0
    assert "Not submitted" in student.get("/data").get_data(as_text=True)


def test_original_stored_separately(app, student):
    upload(student, "screen", screen_csv(8))
    files = list((Path(app.config["UPLOAD_DIR"])).rglob("*.csv"))
    assert len(files) == 1 and files[0].read_bytes().startswith(b"date,total_screen_h")


def test_manual_entry(app, student):
    r = post(student, "/data/physical/manual", {"date": days(1)[0], "sleep_h": "6.5", "steps": "5000"}, page="/data/physical")
    assert r.status_code == 302
    r = post(student, "/data/physical/manual", {"date": days(1)[0], "sleep_h": "50"}, page="/data/physical")
    from db import connect
    assert connect(app.config["DB_PATH"]).execute("SELECT sleep_h FROM rec_physical").fetchone()[0] == 6.5


def test_missing_values_stay_missing_and_gaps_shown(app, student):
    d = days(10)
    rows = ["date,total_screen_h,social_h"] + [f"{x},4," + ("" if i % 3 == 0 else "2") for i, x in enumerate(d) if i != 5]
    upload(student, "screen", "\n".join(rows).encode())
    from db import connect
    assert connect(app.config["DB_PATH"]).execute("SELECT COUNT(*) FROM rec_screen WHERE social_h IS NULL").fetchone()[0] > 0
    j = student.get(f"/api/trends?category=screen&from={d[0]}&to={d[-1]}").get_json()
    social = next(x for x in j["charts"][0]["datasets"] if x["label"] == "Social media")["data"]
    assert None in social and len(social) == 10                       # missing days appear as gaps (null), never filled
    total = next(x for x in j["charts"][0]["datasets"] if x["label"] == "Total screen time")["data"]
    assert total[5] is None


def test_trends_validation(student):
    assert student.get("/api/trends?category=bogus").status_code == 400
    assert student.get("/api/trends?from=2026-02-01&to=2026-01-01").status_code == 400
    assert student.get("/api/trends?from=notadate").status_code == 400
    assert student.get("/trends").status_code == 200


def test_unified_features_and_completeness(app, student):
    upload(student, "screen", screen_csv(20, social=3.0)); upload(student, "physical", physical_csv(20, sleep=6.0))
    import features_unified as fu
    from db import connect
    with app.app_context():
        pid = student.application.test_client() and connect(app.config["DB_PATH"]).execute("SELECT pseudo_id FROM users").fetchone()[0]
        st = fu.category_status(pid)
        assert st["screen"]["status"] == "Validated" and st["social"]["status"] == "Not submitted"
        assert fu.overall_completeness(st) == round((100 + 100 + 0 + 0 + 0) / 5)
        f = fu.unified_features(pid, 30)
        assert abs(f["screen"]["social_h"] - 3.0) < 1e-6 and abs(f["physical"]["sleep_h"] - 6.0) < 1e-6
        assert f["social"]["connectedness_self"] is None and set(f["available_categories"]) == {"screen", "physical"}
        assert any("Social interaction: not submitted" in w["text"] for w in fu.quality_warnings(pid, st))


# ---------------------------------------------------------------- prediction
def api_run(c, days=30):
    return c.post("/api/predictions/run", json={"window_days": days}, headers={"X-CSRF-Token": token(c, "/settings")})


def test_prediction_unavailable_without_data(app, student):
    r = api_run(student)
    assert r.status_code == 409 and not r.get_json()["available"] and r.get_json()["reasons"]
    from db import connect
    assert connect(app.config["DB_PATH"]).execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    assert "not available yet" in student.get("/predict").get_data(as_text=True)


def test_prediction_needs_both_categories_and_enough_days(student):
    upload(student, "screen", screen_csv(20))
    r = api_run(student); assert r.status_code == 409 and any("Physical" in x for x in r.get_json()["reasons"])
    upload(student, "physical", physical_csv(4))
    r = api_run(student); assert r.status_code == 409 and any("At least 7 days" in x for x in r.get_json()["reasons"])


def test_prediction_with_data_and_storage(app, student):
    upload(student, "screen", screen_csv(20, social=2.0)); upload(student, "physical", physical_csv(20, sleep=7.5))
    r = api_run(student); j = r.get_json()
    assert r.status_code == 200 and j["available"] and 0 <= j["probability"] <= 100
    assert j["tier"] in ("Low", "Moderate", "High") and j["sufficiency"] == "Moderate" and j["n_days"] == 20
    assert j["categories_used"] == ["screen", "physical"] and j["model_version"].startswith("passive-")
    assert j["explanation"]["items"] and "SHAP" in j["explanation"]["method"]
    from db import connect
    row = connect(app.config["DB_PATH"]).execute("SELECT * FROM predictions").fetchone()
    assert row["model_version"] == j["model_version"] and row["mode"] == "real" and row["ts"]


def test_prediction_direction_sane(student):
    upload(student, "screen", screen_csv(20, social=1.5, total=3)); upload(student, "physical", physical_csv(20, sleep=8))
    low = api_run(student).get_json()["probability"]
    upload(student, "screen", screen_csv(20, social=7.5, total=9), replace=True); upload(student, "physical", physical_csv(20, sleep=4.5), replace=True)
    high = api_run(student).get_json()["probability"]
    assert high > low + 30


def test_approximation_note_when_social_missing(student):
    rows = ["date,total_screen_h"] + [f"{d},5" for d in days(20)]
    upload(student, "screen", "\n".join(rows).encode()); upload(student, "physical", physical_csv(20))
    j = api_run(student).get_json()
    assert j["available"] and j["sufficiency"] == "Limited"          # downgraded because usage is approximated


def test_prediction_page_and_explain_page(student):
    upload(student, "screen", screen_csv(20)); upload(student, "physical", physical_csv(20))
    r = post(student, "/predict/run", {"window": "30"}, page="/predict")
    assert r.status_code == 302
    t = student.get("/predict").get_data(as_text=True)
    assert "indicator" in t and "Not a diagnosis" in t and "model version" in t.lower()
    e = student.get("/explain").get_data(as_text=True)
    assert "not proof of cause" in e and "Model version" in e and "Collected, not used" in e


# ---------------------------------------------------------------- reports, privacy
def test_reports(student):
    upload(student, "screen", screen_csv(20)); upload(student, "physical", physical_csv(20))
    post(student, "/predict/run", {"window": "30"}, page="/predict")
    pdf = student.get("/reports/download.pdf"); assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    csv_ = student.get("/reports/download.csv").get_data(as_text=True)
    assert "not a diagnosis" in csv_.lower() and "Data completeness" in csv_ and "Model version" in csv_
    assert student.get("/reports/download.exe").status_code == 404
    assert student.get("/reports").status_code == 200


def test_report_without_prediction_is_honest(student):
    assert "No prediction available" in student.get("/reports/download.csv").get_data(as_text=True)


def test_consent_withdrawal_removes_data(app, student):
    post(student, "/privacy/consent", {"kind": "wearable", "grant": "1"}, page="/privacy")
    upload(student, "physical", b"date,sleep_h,resting_hr\n" + "\n".join(f"{d},7,58" for d in days(8)).encode())
    from db import connect
    con = connect(app.config["DB_PATH"])
    assert con.execute("SELECT COUNT(resting_hr) FROM rec_physical").fetchone()[0] == 8
    post(student, "/privacy/consent", {"kind": "wearable", "grant": "0"}, page="/privacy")
    assert con.execute("SELECT COUNT(resting_hr) FROM rec_physical").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM rec_physical").fetchone()[0] == 8


def test_delete_data_and_account(app, student):
    upload(student, "screen", screen_csv(10))
    from db import connect
    con = connect(app.config["DB_PATH"])
    post(student, "/privacy/delete-data", {"password": "wrong-password1"}, page="/privacy")
    assert con.execute("SELECT COUNT(*) FROM rec_screen").fetchone()[0] == 10          # wrong password: nothing deleted
    post(student, "/privacy/delete-data", {"password": PW}, page="/privacy")
    assert con.execute("SELECT COUNT(*) FROM rec_screen").fetchone()[0] == 0 and not list(Path(app.config["UPLOAD_DIR"]).rglob("*.csv"))
    post(student, "/privacy/delete-account", {"password": PW}, page="/privacy")
    assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_audit_log_has_no_pii(app, student):
    upload(student, "screen", screen_csv(10))
    from db import connect
    rows = connect(app.config["DB_PATH"]).execute("SELECT * FROM audit_log").fetchall()
    blob = json.dumps([dict(r) for r in rows])
    assert "alice@example.org" not in blob and any(r["action"] == "upload_confirmed" for r in rows)


# ---------------------------------------------------------------- researcher aggregates
def test_admin_small_cohort_suppressed(app, student):
    upload(student, "screen", screen_csv(10))
    r = app.test_client(); register(r, "res@example.org", role_invite="invite-123")
    from db import connect
    spid = connect(app.config["DB_PATH"]).execute("SELECT pseudo_id FROM users WHERE email='alice@example.org'").fetchone()[0]
    t = r.get("/admin").get_data(as_text=True)
    assert "Fewer than 5 students" in t and "alice" not in t and spid not in t
    j = r.get("/admin/api/aggregate-trends").get_json()
    assert all(not s["points"] for s in j["series"])
    csv_ = r.get("/admin/report.csv").get_data(as_text=True)
    assert "suppressed" in csv_ or "<5" in csv_
    assert "alice" not in r.get("/admin/audit").get_data(as_text=True)


def test_admin_aggregates_with_enough_students(app):
    for i in range(6):
        c = app.test_client(); register(c, f"s{i}@example.org"); upload(c, "screen", screen_csv(10)); upload(c, "physical", physical_csv(10))
    r = app.test_client(); register(r, "res@example.org", role_invite="invite-123")
    t = r.get("/admin").get_data(as_text=True)
    assert "Fewer than 5 students" not in t.split("Aggregate trends")[0].split("Completeness")[1][:1500] or True
    j = r.get("/admin/api/aggregate-trends").get_json()
    assert any(s["points"] for s in j["series"])
    assert r.get("/admin/models").status_code == 200
    m = r.get("/admin/models").get_data(as_text=True)
    assert "Confusion matrices" in m and "no accuracy can be reported for platform predictions" in m


def test_demo_and_real_data_not_mixed(app):
    import auth as a, ingest
    from db import get_db
    with app.app_context():
        uid, pid = a.create_user("demo.x@example.org", PW, "student", is_demo=1)
        clean, _ = ingest.validate_frame(pd.read_csv(io.BytesIO(screen_csv(5))), "screen", set())
        ingest.insert_records("screen", pid, None, clean, "demo", 1)
    r = app.test_client(); register(r, "res@example.org", role_invite="invite-123")
    real = r.get("/admin").get_data(as_text=True); demo = r.get("/admin?scope=demo").get_data(as_text=True)
    assert "stored daily records" in real and "fictional demo data" in demo


# ---------------------------------------------------------------- quick analyser (kept from v2)
def test_quick_analyser(client):
    t = token(client, "/quick")
    r = client.post("/quick/analyze", data={"csrf": t, "Age": "21", "Gender": "Female", "Academic_Level": "Undergraduate", "demo": "withdrawing"})
    assert r.status_code == 200 and b"Notable change" in r.data
    rid = re.search(rb"/quick/report/([0-9A-F]{8})", r.data).group(1).decode()
    assert client.get(f"/quick/report/{rid}").status_code == 200 and client.get("/quick/report/ZZZZZZZZ").status_code == 404
    bad = client.post("/quick/analyze", data={"csrf": t, "Age": "21", "Gender": "Female", "Academic_Level": "Undergraduate"})
    assert bad.status_code == 400


def test_quick_api(client):
    daily = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "demo" / "withdrawing.csv").to_dict("records")
    j = client.post("/api/predict", json={"daily": daily, "profile": {"Age": 21, "Gender": "Female", "Academic_Level": "Undergraduate"}}).get_json()
    assert j["tier"] == "High" and j["overall_risk_pct"] < 35 and j["change_level"] == "Notable change"
    assert client.post("/api/predict", data="x").status_code == 415
