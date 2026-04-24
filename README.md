# Relevant Priors API — Deployment Guide

## How It Works

```
POST /predict
│
├── GBM model scores each (current, prior) pair -> probability 0-1
│   Features: body-part overlap, modality overlap, Jaccard similarity,
│             prior age (years), number of overlapping groups
│
├── prob > 0.75  -> predict True  (no LLM needed)
├── prob < 0.25  -> predict False (no LLM needed)
└── 0.25-0.75   -> GPT-4o-mini classifies in one batched call per case
                   (async, 25 concurrent, cached by MD5)
```

**Accuracy:**
- Model alone: **93.42%** on public split (27,614 pairs)
- Quick API check (10 cases, 173 priors): **98.27%**

---

## Deploy on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) -> New Project -> Deploy from GitHub
3. In **Variables** tab, add:
   ```
   OPENAI_API_KEY=sk-...your key...
   ```
4. Railway auto-detects the Dockerfile, builds, and trains the model
5. In **Settings** -> **Networking** -> Generate Domain (port 8080)

---

## Test Your Endpoint
```bash
curl -X POST https://YOUR_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "relevant-priors-v1",
    "cases": [{
      "case_id": "test1",
      "patient_id": "p1",
      "patient_name": "Test Patient",
      "current_study": {
        "study_id": "curr1",
        "study_description": "CT CHEST WITH CNTRST",
        "study_date": "2026-01-01"
      },
      "prior_studies": [
        {"study_id": "p1", "study_description": "CT CHEST WITHOUT CNTRST", "study_date": "2025-01-01"},
        {"study_id": "p2", "study_description": "MRI BRAIN WO", "study_date": "2024-01-01"}
      ]
    }]
  }'
```

Expected response:
```json
{
  "predictions": [
    {"case_id": "test1", "study_id": "p1", "predicted_is_relevant": true},
    {"case_id": "test1", "study_id": "p2", "predicted_is_relevant": false}
  ]
}
```

---

## Local Testing

```bash
# Install deps
pip install -r requirements.txt

# Re-train model (optional — Dockerfile does this automatically)
python3 train_model.py --data relevant_priors_public.json

# Run server locally
uvicorn app:app --reload --port 8080

# Evaluate offline accuracy
python3 evaluate_local.py --data relevant_priors_public.json
```
