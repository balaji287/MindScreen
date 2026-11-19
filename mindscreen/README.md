# MindScreen: Multimodal Student Mental-Health Screening Platform

A screening and decision-support web application (not a diagnosis tool). Students submit **five categories of data** (screen time, behavioural, physical wellness, social interaction, academic) through separate upload pages; the platform validates them, builds one unified feature set, runs one prediction pipeline, explains the result with SHAP, shows trends with honest gaps, and produces reports. Researchers see privacy-protected aggregates only.

This is the earlier *MindScreen* model project, integrated with the full web application. The trained model, training code, change monitor and quick analyser from the previous version are kept and working.

## Run it (exact commands)
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then set MINDSCREEN_SECRET_KEY (see below)
python src/seed_demo.py             # optional: FICTIONAL demo accounts; prints their passwords once
python src/app.py                   # open http://127.0.0.1:5000
```
Generate a secret key: `python -c "import secrets; print(secrets.token_hex(32))"`. If you skip it, a random key is created in `instance/secret.key` (fine for development).
Pre-trained models are included. To retrain: `python src/train.py` (about a minute). If your scikit-learn version differs, the app retrains automatically on first start.

**Tests:** `pip install -r requirements-dev.txt && python -m pytest tests -q` (38 tests).

**Accounts:** register as a Student on `/register`. A Researcher account needs the invite code you set in `.env` (`RESEARCHER_INVITE_CODE`); with no code set, researcher sign-up is disabled. `python src/seed_demo.py` creates `demo.student@example.org` and `demo.researcher@example.org` (+ 13 fictional students) with random passwords (or `DEMO_PASSWORD` from `.env`).

## What to look at (2-minute demo)
1. Log in as the demo student → **Overview** (completeness, trend chart, data-quality notes).
2. **Data Collection** → *Screen time* → download the template, upload it, see validation + preview.
3. **ML Prediction** → run a screening → category, calibrated estimate, 4-model agreement, data sufficiency.
4. **Explainable AI** → SHAP contributions, categories used / not used, model version + timestamp.
5. **Behavioural Trends** → date range + category; gaps shown as gaps; change-from-baseline monitor.
6. **Reports** → PDF/CSV. **Privacy & Consent** → consents, delete data/account.
7. Log in as the demo researcher → **Overview** (switch to *Demo* scope), **Model evaluation**, **Audit log**.

## Feature status: what is functional, what is a placeholder
| Area | Status |
|---|---|
| Landing page, registration, login/logout, Student + Researcher roles, server-side role checks | **Functional** |
| Five separate data pages: CSV drag-and-drop upload, sample templates, manual entry, preview, validation (missing columns, invalid values, duplicates, dates), statuses, replace/delete | **Functional** |
| Unified storage, pseudonymous IDs, completeness %, data-quality warnings, originals stored separately from cleaned records | **Functional** |
| Prediction pipeline (LR/RF/GB/SVM trained on a public survey dataset), model version + timestamp stored, "unavailable" state when data is missing | **Functional** |
| SHAP explanations (local) + permutation importance (global) | **Functional** |
| Trends (interactive charts, date range, category filter, gaps preserved) and change-from-baseline monitor | **Functional** |
| PDF/CSV reports; researcher aggregate CSV | **Functional** |
| Consent (core, wearable, grades), data deletion, account deletion, audit log, k-anonymity (groups < 5 hidden), CSRF, rate limiting, security headers | **Functional** |
| Model evaluation panel (metrics, confusion matrices, group comparison) | **Functional, with a caveat**: figures come from the public training dataset; there are no clinician-labelled outcomes for platform users, so no platform accuracy is reported |
| **Behavioural, Social and Academic data in the ML model** | **Not used by the model.** They are collected, validated, trended and reported, but the model was never trained on them. A true multi-category model needs labelled data (see docs/FUTURE_SCOPE.md). The Prediction and Explain pages say so |
| Demo accounts | **Demo data**: fictional records, flagged `is_demo`, separated from real data in every aggregate. Predictions on them are real model outputs on fictional inputs |
| Wearable heart-rate/HRV columns | Accepted only with separate consent; stored; **not used by the model** |
| Quick analyser (`/quick`, `/api/predict`) | Functional (kept from the previous version): no account, nothing stored |
| Email verification, password reset, HTTPS/production hosting, clinical validation | **Not implemented** |

## Stack
Flask + Jinja templates + SQLite, scikit-learn, SHAP, ReportLab, Chart.js (bundled in `static/vendor`, no CDN). I kept the existing Flask stack rather than rebuilding in React/Vite, as your brief allows ("retain it unless there is a strong reason"): it preserves the working model, tests and quick analyser, and needs no Node build step.

## Project structure
```
src/
  app.py                create_app(), security headers, error pages
  auth.py               register/login, roles, CSRF, rate limit, consents
  categories.py         single source of truth for the 5 categories (columns, ranges, templates)
  db.py                 SQLite schema (tables generated from categories.py), audit helper
  ingest.py             CSV/manual validation + storage
  features_unified.py   completeness, quality warnings, unified features, baseline adapter
  predict_service.py    unified prediction pipeline, model-version registry
  predictor.py          loads the 4 trained models, tiers, what-if
  explain.py            SHAP + global importance
  reports.py            PDF / CSV reports
  views_*.py            main, data, analysis, admin, quick
  train.py ablation.py features.py footprint.py demo_data.py seed_demo.py
templates/ static/      UI (static/vendor = Chart.js)
models/                 trained models + metrics.json, schema.json, subgroups.json
data/                   students_behaviour.csv (public training data), demo/ (simulated uploads)
tests/                  pytest suite
docs/                   PROJECT_REPORT.md, FUTURE_SCOPE.md
```
Database tables: `users` (role, pseudonymous ID), `consents`, `uploads`, `rec_screen`, `rec_behaviour`, `rec_physical`, `rec_social`, `rec_academic`, `predictions` (model version, mode, explanation), `model_versions` (+ evaluation JSON), `audit_log`.

## CSV formats
Every category page has a downloadable template. All files: UTF-8 CSV, ≤ 2 MB, ≤ 2000 rows, `date` as YYYY-MM-DD, empty cells allowed (stay missing). Columns ending `_self` are self-reported. Full column list with ranges is shown on each page.

## Environment variables
See `.env.example`: `MINDSCREEN_SECRET_KEY`, `RESEARCHER_INVITE_CODE`, `DEMO_PASSWORD`, `MINDSCREEN_HTTPS`, `MINDSCREEN_INSTANCE`.

## Honest limits
Research prototype. The model is trained on a small public survey with a self-reported label; it uses only screen-time and sleep (plus profile). Not clinically validated, not a medical device. Details: `docs/PROJECT_REPORT.md`, section 8.
