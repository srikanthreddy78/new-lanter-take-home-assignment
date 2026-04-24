"""
Relevant Priors API — New Grad Residency Challenge
Predicts whether prior patient exams are relevant to the current exam.

Architecture:
  1. GBM model (trained on 27,614 public examples, 92.5% CV accuracy)
     Returns calibrated probability for each (current, prior) pair.
  2. High-confidence predictions (prob < 0.25 or > 0.75) → return directly.
  3. Uncertain zone (0.25–0.75) → batch per case to Claude API for refinement.
  4. Everything cached in-memory so retries are instant.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Relevant Priors API")

# ─────────────────────────────────────────────────────────────
# Synonym groups  (kept in sync with train_model.py)
# ─────────────────────────────────────────────────────────────

BODY_PART_GROUPS: list[set[str]] = [
    {"BRAIN", "HEAD", "CRANIAL", "INTRACRANIAL", "CEREBR", "SKULL"},
    {"CHEST", "LUNG", "THORAX", "PULMON", "MEDIASTIN"},
    {"ABDOMEN", "ABD", "ABDOMINAL"},
    {"PELV", "PELVIC"},
    {"ABD", "ABDOMEN", "PELV", "PELVIC", "ABDOM"},
    {"SPINE", "SPINAL", "VERTEBR"},
    {"LUMBAR", "LUMB", "LSP", "L-SPINE"},
    {"CERVICAL", "C-SPINE", "CSPINE"},
    {"THORACIC SPINE", "TSPINE", "T-SPINE"},
    {"BREAST", "MAM", "MAMMO", "MAMMOGRAPH"},
    {"HEART", "CARDIAC", "CARDIO", "CORONARY", "MYO", "MYOCARD", "VENTRIC"},
    {"NECK", "THYROID", "SOFT TISSUE NECK"},
    {"KNEE"}, {"HIP"}, {"SHOULDER"}, {"ANKLE"},
    {"FOOT", "FEET", "CALCAN"},
    {"WRIST"}, {"HAND", "FINGER", "DIGIT"},
    {"ELBOW"}, {"FOREARM"}, {"TIBIA", "FIBULA"},
    {"LIVER", "HEPAT"}, {"KIDNEY", "RENAL"},
    {"PANCREAS", "PANCREATIC"},
    {"ORBIT", "EYE", "OCULAR"},
    {"BONE", "OSSEOUS", "DXA", "DEXA"},
    {"PROSTATE"}, {"UTERUS", "OVARY", "OVARIAN"},
    {"AORTA", "AORTIC"},
]

MODALITY_GROUPS: list[set[str]] = [
    {"CT", "CAT SCAN"},
    {"MRI", "MR "},
    {"XR", "X-RAY", "XRAY", "RADIOGRAPH"},
    {"MAM", "MAMMO", "MAMMOGRAPH"},
    {"US", "ULTRASOUND", "SONOGRAM"},
    {"ECHO"},
    {"NM", "NUCLEAR", "PET", "SPECT", "SCINTIG"},
    {"DXA", "DEXA"},
    {"FLUORO"},
    {"ANGIO"},
    {"VAS", "DOPPLER", "TRANSCRANIAL"},
]


def _normalize(desc: str) -> tuple[frozenset[int], frozenset[int]]:
    text = " " + desc.upper() + " "
    mods: set[int] = set()
    parts: set[int] = set()
    for i, g in enumerate(MODALITY_GROUPS):
        for kw in g:
            if kw in text:
                mods.add(i)
                break
    for i, g in enumerate(BODY_PART_GROUPS):
        for kw in g:
            if kw in text:
                parts.add(i)
                break
    return frozenset(mods), frozenset(parts)


def _jaccard(a: str, b: str) -> float:
    ta = set(a.upper().split())
    tb = set(b.upper().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _date_diff_years(d1: str, d2: str) -> float:
    try:
        a = datetime.strptime(d1, "%Y-%m-%d")
        b = datetime.strptime(d2, "%Y-%m-%d")
        return abs((a - b).days) / 365.25
    except Exception:
        return 5.0


def extract_features(
    current_desc: str, prior_desc: str,
    current_date: str, prior_date: str,
) -> list[float]:
    cm, cp = _normalize(current_desc)
    pm, pp = _normalize(prior_desc)
    same_mod = int(bool(cm & pm))
    same_part = int(bool(cp & pp))
    j = _jaccard(current_desc, prior_desc)
    exact = int(current_desc.strip().upper() == prior_desc.strip().upper())
    age = min(_date_diff_years(current_date, prior_date), 15.0)
    n_mod_overlap = len(cm & pm)
    n_part_overlap = len(cp & pp)
    both_have_mod = int(bool(cm) and bool(pm))
    both_have_part = int(bool(cp) and bool(pp))
    curr_con = int("W CON" in current_desc.upper() or "WITH CNTR" in current_desc.upper()
                   or " W C" in current_desc.upper())
    prior_con = int("W CON" in prior_desc.upper() or "WITH CNTR" in prior_desc.upper()
                    or " W C" in prior_desc.upper())
    same_contrast = int(curr_con == prior_con)
    return [
        same_mod, same_part, j, exact, age,
        n_mod_overlap, n_part_overlap,
        both_have_mod, both_have_part, same_contrast,
    ]


# ─────────────────────────────────────────────────────────────
# Load ML model
# ─────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model.pkl"
_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                artifact = pickle.load(f)
            _model = artifact["model"]
            logger.info("Model loaded (CV acc: %.4f)", artifact.get("cv_accuracy", 0))
        else:
            logger.warning("model.pkl not found — falling back to heuristics")
    return _model


# ─────────────────────────────────────────────────────────────
# LLM refinement for uncertain zone
# ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_CACHE: dict[str, bool] = {}
MAX_CONCURRENT_LLM = 25

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.75


def _cache_key(current_desc: str, prior_desc: str) -> str:
    return hashlib.md5(f"{current_desc}|||{prior_desc}".encode()).hexdigest()


async def _llm_batch(
    client: httpx.AsyncClient,
    current_desc: str,
    prior_descs: list[str],
) -> list[bool]:
    """One OpenAI API call for all uncertain priors in a case."""
    if not prior_descs:
        return []

    results: list[bool | None] = [None] * len(prior_descs)
    uncached: list[int] = []
    for i, p in enumerate(prior_descs):
        ck = _cache_key(current_desc, p)
        if ck in LLM_CACHE:
            results[i] = LLM_CACHE[ck]
        else:
            uncached.append(i)

    if not uncached:
        return results  # type: ignore

    lines = "\n".join(
        f"{idx + 1}. {prior_descs[i]}" for idx, i in enumerate(uncached)
    )
    prompt = (
        f"A radiologist is about to read: {current_desc}\n\n"
        "For each numbered prior study, answer true if it should be shown "
        "because it images the same body region or condition (regardless of modality), "
        "or false if it covers a different body region.\n"
        "Reply ONLY with a JSON array of true/false values in order. No explanation.\n\n"
        f"{lines}"
    )

    try:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        preds = json.loads(raw)
        if not isinstance(preds, list):
            raise ValueError("Expected list")
        preds = [bool(p) for p in preds]
        while len(preds) < len(uncached):
            preds.append(True)
    except Exception as e:
        logger.warning("LLM call failed: %s — defaulting uncertain→True", e)
        preds = [True] * len(uncached)

    for local_idx, orig_idx in enumerate(uncached):
        pred = preds[local_idx]
        results[orig_idx] = pred
        LLM_CACHE[_cache_key(current_desc, prior_descs[orig_idx])] = pred

    return results  # type: ignore


# ─────────────────────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    get_model()


@app.get("/predict")
def predict_get():
    return {"status": "ok", "message": "Send POST request to this endpoint"}


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    t0 = time.time()
    body = await request.json()
    cases = body.get("cases", [])
    total_priors = sum(len(c.get("prior_studies", [])) for c in cases)
    logger.info("Request: %d cases, %d total priors", len(cases), total_priors)

    clf = get_model()

    # Phase 1 — ML scoring
    ml_probs: dict[tuple[str, str], float] = {}
    uncertain_by_case: dict[str, dict[str, Any]] = {}

    for case in cases:
        case_id = case["case_id"]
        curr_desc = case["current_study"]["study_description"]
        curr_date = case["current_study"]["study_date"]
        priors = case.get("prior_studies", [])

        if clf is not None and priors:
            feats = [
                extract_features(curr_desc, p["study_description"],
                                 curr_date, p["study_date"])
                for p in priors
            ]
            probs = clf.predict_proba(feats)[:, 1]
            for prior, prob in zip(priors, probs):
                ml_probs[(case_id, prior["study_id"])] = float(prob)
        else:
            for prior in priors:
                cm, cp = _normalize(curr_desc)
                pm, pp = _normalize(prior["study_description"])
                same_part = bool(cp & pp)
                same_mod = bool(cm & pm)
                if same_part and same_mod:
                    prob = 0.92
                elif same_part:
                    prob = 0.70
                elif same_mod:
                    prob = 0.20
                else:
                    prob = 0.07
                ml_probs[(case_id, prior["study_id"])] = prob

        unc = [
            p for p in priors
            if LOW_THRESHOLD <= ml_probs.get((case_id, p["study_id"]), 0.5) <= HIGH_THRESHOLD
        ]
        if unc:
            uncertain_by_case[case_id] = {"current": curr_desc, "priors": unc}

    # Phase 2 — LLM for uncertain zone
    llm_preds: dict[tuple[str, str], bool] = {}
    if uncertain_by_case and OPENAI_API_KEY:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        async def refine_case(case_id: str, info: dict) -> None:
            async with semaphore:
                async with httpx.AsyncClient() as client:
                    prior_descs = [p["study_description"] for p in info["priors"]]
                    preds = await _llm_batch(client, info["current"], prior_descs)
                    for prior, pred in zip(info["priors"], preds):
                        llm_preds[(case_id, prior["study_id"])] = pred

        await asyncio.gather(
            *[refine_case(cid, info) for cid, info in uncertain_by_case.items()]
        )

    # Phase 3 — Assemble response
    predictions = []
    for case in cases:
        case_id = case["case_id"]
        for prior in case.get("prior_studies", []):
            sid = prior["study_id"]
            key = (case_id, sid)
            prob = ml_probs.get(key, 0.5)
            if key in llm_preds:
                pred = llm_preds[key]
            elif prob >= HIGH_THRESHOLD:
                pred = True
            elif prob <= LOW_THRESHOLD:
                pred = False
            else:
                pred = prob >= 0.5
            predictions.append({
                "case_id": case_id,
                "study_id": sid,
                "predicted_is_relevant": pred,
            })

    elapsed = time.time() - t0
    n_uncertain = sum(len(v["priors"]) for v in uncertain_by_case.values())
    logger.info(
        "Done: %d predictions | %d uncertain→LLM | %.2fs | llm_cache=%d",
        len(predictions), n_uncertain, elapsed, len(LLM_CACHE),
    )
    return JSONResponse({"predictions": predictions})


@app.get("/health")
def health():
    clf = get_model()
    return {
        "status": "ok",
        "model_loaded": clf is not None,
        "cached_pairs": len(LLM_CACHE),
        "llm_enabled": bool(OPENAI_API_KEY),
    }
