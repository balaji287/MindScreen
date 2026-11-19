# Future Scope and Commercialisation Plan

## 1. Who would pay
| Customer | Pain | What MindScreen gives | Revenue model |
|---|---|---|---|
| Colleges / universities | Counsellors see students only after a crisis | Cohort dashboard + early-warning list (Campus page), change alerts | per-student per-year licence |
| Hospitals / mental-health clinics | Need appropriate referrals; long intake forms | Consent-based referral summary + API into the hospital system | per-referral fee or annual integration licence |
| Corporate wellness / insurers | Burnout and attrition | Same engine on workplace data | subscription |

## 2. From CSV upload to continuous tracking (the roadmap that makes it sellable)
| Phase | Input | What changes |
|---|---|---|
| 1 (this platform) | five-category validated uploads, accounts and roles, unified pipeline; screen time + sleep scored; other categories tracked | trends, SHAP, reports, researcher aggregates, audit log |
| 2 | **Companion app**: Android Digital Wellbeing / iOS Screen Time, Google Fit / Apple Health / Fitbit sync with consent | removes manual upload; daily data accumulates |
| 3 | **Social-interaction signals**: counts of calls, messages, meet-ups (never content); campus attendance/library use with college consent | scored once labelled data exists |
| 4 | **Wearable + neurological**: heart-rate variability, sleep staging, consumer EEG headband features | multimodal model |
| 5 | **Sequence model** (LSTM/Transformer) on daily series to predict deterioration | trained on clinic-labelled, consented data |

## 3. Hospital tie-up workflow
1. Student or college runs screening (upload or app sync, no questionnaire).
2. A High indicator or a notable change triggers a gentle prompt: "Would you like to share a report with a counsellor?"
3. With **explicit consent**, the PDF/CSV report (Reports page) or the authenticated JSON endpoint (`/api/predictions/run`) goes to the partner clinic.
4. The clinic does the real assessment and returns outcome labels. These labels, which no public dataset provides, form the training data for the next model version (data flywheel).
5. Anonymised cohort statistics go to the college counselling cell.

## 4. Requirements before any real deployment
* Ethics-board approval and informed consent; data-protection compliance (India DPDP Act 2023; HIPAA/GDPR if applicable).
* Data minimisation: only counts and durations, no content or location; anonymised IDs; encryption.
* Bias audit across gender, region and academic level.
* Clinical validation with a psychiatrist against PHQ-9 / clinical interview, including validation of the change monitor.
* Clear "screening, not diagnosis" messaging and a human in the loop for every High-tier case.

## 5. Pitch line
> "MindScreen turns the phone data every student already has into a personal early-warning system. It notices when a student's own routine changes, and gives hospitals a consent-based referral pipeline instead of waiting for a crisis."
