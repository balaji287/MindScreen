"""Single source of truth for the five data categories: columns, ranges, validation rules, templates.
Everything else (DB tables, CSV validation, manual-entry forms, templates, docs) is generated from here."""
from collections import OrderedDict


def col(name, label, kind="num", lo=None, hi=None, required=False, self_reported=False,
        consent=None, choices=None, help=""):
    return dict(name=name, label=label, kind=kind, lo=lo, hi=hi, required=required,
                self_reported=self_reported, consent=consent, choices=choices, help=help)


DATE = col("date", "Date (YYYY-MM-DD)", "date", required=True)

CATEGORIES = OrderedDict([
    ("screen", dict(
        label="Screen time", short="A", table="rec_screen", icon="📱",
        desc="Aggregate phone-usage durations per day: total, social media, entertainment, productivity, "
             "night-time use and pick-ups. Only totals are used: never message content, passwords or browsing history.",
        columns=[
            DATE,
            col("total_screen_h", "Total screen time (h)", lo=0, hi=24, required=True),
            col("social_h", "Social-media time (h)", lo=0, hi=24),
            col("entertainment_h", "Entertainment / video (h)", lo=0, hi=24),
            col("productivity_h", "Educational / productivity apps (h)", lo=0, hi=24),
            col("pickups", "Phone pick-ups / unlocks", "int", 0, 1000),
            col("night_h", "Night-time screen use (h)", lo=0, hi=12, help="e.g. between 11 pm and 5 am"),
            col("top_app", "Most-used app (optional)", "text", help="free text, max 30 characters"),
        ],
        template_rows=[("2026-10-01", 5.2, 3.1, 1.2, 0.9, 74, 0.8, "Instagram"),
                       ("2026-10-02", 4.6, 2.7, 1.1, 0.8, 61, 0.4, "Instagram"),
                       ("2026-10-03", 6.1, 3.9, 1.5, 0.7, 88, 1.3, "YouTube")])),
    ("behaviour", dict(
        label="Behavioural", short="B", table="rec_behaviour", icon="🧭",
        desc="Routine consistency and daily check-ins. Columns ending in _self are self-reported; "
             "the consistency scores are measured/derived (e.g. from sleep-time and study-time variability).",
        columns=[
            DATE,
            col("sleep_wake_consistency", "Sleep/wake consistency (0-10, measured)", lo=0, hi=10),
            col("study_routine_consistency", "Study routine consistency (0-10, measured)", lo=0, hi=10),
            col("activity_pattern", "Daily activity pattern", "cat", choices=["sedentary", "light", "moderate", "active"]),
            col("routine_change_self", "Change in routine today (0 none - 3 major)", "int", 0, 3, self_reported=True),
            col("concentration_self", "Concentration (1-5)", "int", 1, 5, self_reported=True),
            col("motivation_self", "Motivation (1-5)", "int", 1, 5, self_reported=True),
            col("stress_self", "Stress (1-5)", "int", 1, 5, self_reported=True),
            col("mood_checkin_self", "Daily mood check-in (1-5)", "int", 1, 5, self_reported=True),
        ],
        template_rows=[("2026-10-01", 7.5, 6.0, "moderate", 0, 4, 3, 2, 4),
                       ("2026-10-02", 6.8, 7.0, "light", 1, 3, 3, 3, 3),
                       ("2026-10-03", 5.9, 5.5, "sedentary", 2, 2, 2, 4, 2)])),
    ("physical", dict(
        label="Physical wellness", short="C", table="rec_physical", icon="🏃",
        desc="Sleep, steps and exercise. Heart-rate and HRV are optional wearable measurements and need separate "
             "consent. They never establish a mental-health condition on their own.",
        columns=[
            DATE,
            col("sleep_h", "Sleep duration (h)", lo=0, hi=16, required=True),
            col("sleep_quality_self", "Sleep quality (1-5)", "int", 1, 5, self_reported=True),
            col("steps", "Daily steps", "int", 0, 60000),
            col("exercise_min", "Exercise (min)", lo=0, hi=600),
            col("activity_days_week", "Active days this week (0-7)", "int", 0, 7),
            col("resting_hr", "Resting heart rate (bpm)", "int", 30, 200, consent="wearable"),
            col("hrv_ms", "Heart-rate variability (ms)", lo=5, hi=300, consent="wearable"),
        ],
        template_rows=[("2026-10-01", 7.2, 4, 7800, 35, 4, "", ""),
                       ("2026-10-02", 6.4, 3, 5200, 0, 4, "", ""),
                       ("2026-10-03", 8.0, 4, 9100, 50, 5, "", "")])),
    ("social", dict(
        label="Social interaction", short="D", table="rec_social", icon="🤝",
        desc="Voluntary, aggregated indicators only: how often, how long and how connected. "
             "No conversations, contact lists or message content are ever read.",
        columns=[
            DATE,
            col("interaction_freq_self", "Social interactions today (count)", "int", 0, 100, self_reported=True),
            col("group_activity", "Took part in a group/campus activity (0/1)", "int", 0, 1, self_reported=True),
            col("social_time_min", "Time in social activities (min)", lo=0, hi=1440, self_reported=True),
            col("connectedness_self", "Social connectedness (1-5)", "int", 1, 5, self_reported=True),
            col("loneliness_self", "Loneliness (1-5)", "int", 1, 5, self_reported=True),
        ],
        template_rows=[("2026-10-01", 8, 1, 120, 4, 2), ("2026-10-02", 5, 0, 45, 3, 3), ("2026-10-03", 3, 0, 20, 2, 4)])),
    ("academic", dict(
        label="Academic", short="E", table="rec_academic", icon="🎓",
        desc="Attendance, assignments, study hours and workload. Low grades or attendance do not by themselves "
             "mean mental-health risk. Grades are optional and need separate consent.",
        columns=[
            DATE,
            col("period", "Academic period label (optional)", "text", help="e.g. 2026-W40; used only if date is empty"),
            col("attendance_pct", "Attendance (%)", lo=0, hi=100),
            col("assignment_completion_pct", "Assignments completed (%)", lo=0, hi=100),
            col("study_h", "Study hours", lo=0, hi=24),
            col("workload_level", "Workload indicator (1-5)", "int", 1, 5),
            col("academic_pressure_self", "Academic pressure (1-5)", "int", 1, 5, self_reported=True),
            col("grade_pct", "Grade / performance (%)", lo=0, hi=100, consent="grades"),
        ],
        template_rows=[("2026-10-01", "", 92, 80, 3.5, 3, 3, ""), ("2026-10-02", "", 88, 75, 2.0, 3, 4, ""),
                       ("2026-10-03", "", 90, 85, 4.0, 4, 3, "")])),
])

STATUSES = ["Not submitted", "Uploaded", "Needs correction", "Validated"]
DATA_COLUMNS = {k: [c for c in v["columns"] if c["name"] != "date"] for k, v in CATEGORIES.items()}


def template_csv(key: str) -> str:
    spec = CATEGORIES[key]
    names = [c["name"] for c in spec["columns"]]
    lines = ["# FICTIONAL example rows: replace with your own data (leave a cell empty if unknown)", ",".join(names)]
    for r in spec["template_rows"]:
        lines.append(",".join("" if v == "" else str(v) for v in r))
    return "\n".join(lines) + "\n"
