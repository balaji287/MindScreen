import io, re, sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app import create_app          # noqa: E402
import auth                         # noqa: E402

PW = "Str0ng-pass-123"


@pytest.fixture()
def app(tmp_path):
    auth.FAILS.clear()
    a = create_app({"TESTING": True, "INSTANCE": tmp_path, "DB_PATH": tmp_path / "t.db", "UPLOAD_DIR": tmp_path / "up",
                    "SECRET_KEY": "test-secret", "RESEARCHER_INVITE_CODE": "invite-123"})
    (tmp_path / "up").mkdir(exist_ok=True)
    return a


@pytest.fixture()
def client(app):
    return app.test_client()


def token(c, url="/login"):
    html = c.get(url).get_data(as_text=True)
    m = re.search(r'name="csrf" value="([^"]+)"', html)
    assert m, f"no csrf token on {url}"
    return m.group(1)


def register(c, email, role_invite=None, age="21", gender="Female", level="Undergraduate"):
    t = token(c, "/register")
    data = {"csrf": t, "email": email, "password": PW, "confirm": PW, "age": age, "gender": gender,
            "academic_level": level, "consent_core": "1"}
    if role_invite: data["invite"] = role_invite
    return c.post("/register", data=data)


def login(c, email, pw=PW):
    return c.post("/login", data={"csrf": token(c), "email": email, "password": pw})


def post(c, url, data=None, files=None, page=None):
    d = dict(data or {}); d["csrf"] = token(c, page or "/settings")
    if files: d.update(files)
    return c.post(url, data=d, content_type="multipart/form-data" if files else None)


def days(n, end=None):
    end = end or date.today()
    return [(end - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def screen_csv(n=20, social=2.0, total=4.0):
    rows = ["date,total_screen_h,social_h,entertainment_h,productivity_h,pickups,night_h,top_app"]
    rows += [f"{d},{total},{social},0.5,0.5,60,0.5,Instagram" for d in days(n)]
    return "\n".join(rows).encode()


def physical_csv(n=20, sleep=7.5):
    rows = ["date,sleep_h,sleep_quality_self,steps,exercise_min,activity_days_week"]
    rows += [f"{d},{sleep},4,8000,30,4" for d in days(n)]
    return "\n".join(rows).encode()


def upload(c, key, raw, name="x.csv", confirm=True, replace=False):
    r = post(c, f"/data/{key}/preview", files={"file": (io.BytesIO(raw), name)}, page=f"/data/{key}")
    if not confirm:
        return r
    m = re.search(r"/data/uploads/(\d+)/confirm", r.get_data(as_text=True))
    assert m, "preview did not offer confirmation:\n" + r.get_data(as_text=True)[:600]
    data = {"replace": "1"} if replace else {}
    return post(c, f"/data/uploads/{m.group(1)}/confirm", data=data, page=f"/data/{key}")


@pytest.fixture()
def student(client):
    register(client, "alice@example.org")
    return client
