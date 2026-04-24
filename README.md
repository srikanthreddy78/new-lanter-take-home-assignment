# Relevant Priors API — Deployment Guide

## Quick Start (Railway.app — free, 5 min)

### 1. Push to GitHub
```bash
git init
git add app.py train_model.py requirements.txt Dockerfile railway.toml experiments.md model.pkl
git commit -m "relevant priors API"
gh repo create relevant-priors-api --public --push
```

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Select your repo
3. In **Variables**, add:
   ```
   ANTHROPIC_API_KEY=sk-ant-...your key...
   PORT=8000
   ```
4. Railway auto-detects the Dockerfile and builds it.
5. Copy the generated URL (e.g. `https://relevant-priors-api-production.up.railway.app`)

### 3. Test your endpoint
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

### 4. Submit
- **Endpoint URL**: `https://YOUR_URL/predict`
- **Code zip**: `relevant-priors-submission.zip` (from this repo)
- **Write-up**: `experiments.md`

---

## How It Works

```
POST /predict
│
├── GBM model scores each (current, prior) pair → probability 0–1
│   Features: body-part overlap, modality overlap, Jaccard similarity,
│             prior age (years), number of overlapping groups
│
├── prob > 0.75  → predict True  (no LLM needed)
├── prob < 0.25  → predict False (no LLM needed)
└── 0.25–0.75   → Claude Haiku classifies in one batched call per case
                  (async, 25 concurrent, cached by MD5)
```

**Accuracy:**
- Model alone: **93.42%** on public split (27,614 pairs)
- Model + Claude: estimated **~95%**

---

## Local Testing

```bash
# Install deps
pip install -r requirements.txt

# Re-train model (optional — model.pkl already included)
python3 train_model.py --data relevant_priors_public.json

# Run server locally
uvicorn app:app --reload --port 8000

# Evaluate rule accuracy offline
python3 evaluate_local.py --data relevant_priors_public.json
```
