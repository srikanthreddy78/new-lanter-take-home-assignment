# Relevant Priors — Experiments & Write-up

## Problem Summary

Given a current patient examination and a list of prior examinations for the same patient, predict which priors are relevant for a radiologist to view alongside the current exam.

**Dataset:** 996 cases, 27,614 labeled prior–current pairs (public split).  
**Baseline class distribution:** 23.78% relevant, 76.22% not relevant.

---

## Experiment 1 — Naive Baselines

| Strategy | Accuracy |
|---|---|
| Predict all False (never show priors) | 76.22% |
| Predict all True (show everything) | 23.78% |

These establish the floor. The dataset is imbalanced — ~3:1 toward not-relevant.

---

## Experiment 2 — Description Exact Match

Observation: when `current.study_description == prior.study_description` (case-insensitive), the pair is relevant **99.9%** of the time (871/872 cases).

This single rule correctly classifies 872 pairs.

---

## Experiment 3 — Modality + Body-Part Keyword Extraction

I built a synonym-aware keyword extractor grouping descriptions by:
- **10 modality groups** (CT, MRI, X-Ray, Mammography, Ultrasound/Echo, Nuclear/PET, DXA, Fluoro, Angiography)
- **28 body-part groups** (brain/head, chest/lung, breast/mammography, heart/cardiac, lumbar, cervical, knee, hip, etc.)

Results by category:

| Category | % Relevant | Optimal Prediction | n |
|---|---|---|---|
| Same modality + same body part | 92.1% | True | 2,838 |
| Same body part only | 79.5% | True | 3,248 |
| Same modality only | 16.8% | False | 3,237 |
| Neither | 4.5% | False | 18,291 |

Applying optimal per-bucket prediction: **~91.8% accuracy**.

Key insight: **body part overlap is a stronger signal than modality overlap.** A CT chest and MRI chest are more likely to be relevant together than a CT chest and CT knee.

---

## Experiment 4 — Token-Level Jaccard Similarity

Jaccard similarity on whitespace-tokenized uppercased descriptions gave strong signal:

| Jaccard Range | % Relevant |
|---|---|
| j = 1.0 (exact) | 99.9% |
| j = 0.7–0.9 | 78–83% |
| j = 0.4–0.7 | 47–75% |
| j = 0.1–0.3 | 43–63% |
| j = 0.0 | 10.7% |

Combining Jaccard thresholds with modality/body-part logic:
- `same_part AND j >= 0.2 → True` captures near-synonym descriptions correctly (e.g. "MAM screen BI with tomo" vs "MAMMOGRAPHY SCREENING BILATERAL")
- `j >= 0.7 → True` catches high-overlap pairs regardless of keyword extraction hits

This pushed rule accuracy to **91.88%** with no external API needed.

---

## Experiment 5 — Ambiguous Case Analysis

After applying all rules, ~3,657 pairs fell into "ambiguous" buckets:

| Bucket | n | % Relevant | Default |
|---|---|---|---|
| `ambiguous_same_part` (same body part, j < 0.2) | 2,608 | 76.4% | True |
| `ambiguous_same_mod` (same mod, j ≥ 0.4) | 343 | 21.6% | False |
| `ambiguous_jaccard` (j in 0.1–0.7, no overlap) | 706 | 12.5% | False |

The same-part ambiguous bucket defaults to True (76.4% correct). The other two default to False.

---

## Experiment 6 — Claude API for Ambiguous Cases

For the `ambiguous_same_part` bucket specifically, the LLM is asked:

> "Given current study [description], which of these prior studies are relevant for a radiologist to see?"

All ambiguous priors for a case are batched into a **single API call** per case (as the challenge hints recommend), with 20 concurrent calls via `asyncio.Semaphore`.

Expected gain: ~1.4–1.8% additional accuracy on ambiguous pairs.

**Prompt design:** The prompt explicitly tells the model to focus on whether the prior images the same body part or condition, regardless of modality. This aligns with clinical practice — a radiologist reading an MRI brain benefits from seeing a prior CT head.

**Model:** `claude-haiku-4-5-20251001` — fast and cost-effective for this classification task.

**Caching:** Results are cached in-memory by `md5(current_desc + "|||" + prior_desc)` to avoid redundant calls on retries.

---

## Final Architecture

```
POST /predict
│
├── For each (current, prior) pair:
│   ├── Exact match? → True
│   ├── High Jaccard (≥0.7)? → True
│   ├── Same mod + same part? → True
│   ├── Same part + j≥0.2? → True
│   ├── Same part + j<0.2? → AMBIGUOUS (LLM)
│   ├── Same mod + j≥0.4? → AMBIGUOUS (LLM)
│   ├── Same mod only? → False
│   ├── j<0.1? → False
│   └── moderate j, no overlap? → False (LLM if time allows)
│
└── LLM layer (async, 20 concurrent, cached)
    └── Batch all ambiguous priors per case → single Claude API call
```

**Expected accuracy:** ~91.9% (rule-only) → ~93–94% (with LLM)

---

## What Worked

1. **Body-part synonym grouping** — the biggest single improvement. Variations like "MAM screen BI with tomo" and "MAMMOGRAPHY SCREENING BILATERAL" both map to the same breast body-part group.
2. **Jaccard on top of keyword rules** — catches synonym-rich pairs the keyword extractor missed.
3. **Defaulting ambiguous same-part to True** — empirically correct 76%+ of the time.
4. **Batching LLM calls per case** — respects the 360s timeout constraint and the challenge's own hint.

## What Failed / Limitations

1. **Pure modality matching** — a CT chest vs CT knee is almost never relevant; modality alone is a weak signal.
2. **Single-token keyword extraction** — multi-word phrases like "CAT SCAN" or "PLAIN FILM" were initially missed; fixed with substring matching.
3. **No imaging metadata** — the challenge only provides study descriptions and dates, not accession numbers, body region tags, or DICOM series data. Richer data would make classification easier.

## Next Improvements

1. **Fine-tune a small text classifier** (e.g. a 3-layer MLP on top of a pre-trained sentence encoder like `all-MiniLM-L6-v2`) trained on the labeled public pairs. Study descriptions are short enough for fast inference.
2. **Radiology ontology lookup** — map descriptions to RadLex or SNOMED CT codes, then check code-level similarity. This would be more robust than keyword heuristics.
3. **Add temporal signal** — more recent priors may be more clinically relevant; a date-distance feature could help.
4. **Self-hosted embedding model** — cosine similarity between sentence embeddings of study descriptions, avoiding the LLM latency cost entirely.
5. **Train on public split, evaluate on held-out** — with 27,614 labeled pairs, a simple logistic regression over (same_mod, same_part, jaccard, date_diff) features could rival the current rule+LLM approach at far lower latency.
