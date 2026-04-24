"""
train_model.py — Train and save the relevance classifier.

Run locally:
    python3 train_model.py --data relevant_priors_public.json

Produces model.pkl which app.py loads at startup.
"""

import argparse
import json
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────
# Synonym groups (must stay in sync with app.py)
# ─────────────────────────────────────────────────────────────

BODY_PART_GROUPS: list[set[str]] = [
    {"BRAIN", "HEAD", "CRANIAL", "INTRACRANIAL", "CEREBR", "SKULL"},
    {"CHEST", "LUNG", "THORAX", "PULMON", "MEDIASTIN"},       # thoracic CHEST, not spine
    {"ABDOMEN", "ABD", "ABDOMINAL"},
    {"PELV", "PELVIC"},
    {"ABD", "ABDOMEN", "PELV", "PELVIC", "ABDOM"},            # combined abdo-pelvic
    {"SPINE", "SPINAL", "VERTEBR"},
    {"LUMBAR", "LUMB", "LSP", "L-SPINE"},
    {"CERVICAL", "C-SPINE", "CSPINE"},
    {"THORACIC SPINE", "TSPINE", "T-SPINE"},                  # thoracic spine != chest
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


def _normalize(desc: str):
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


def extract_features(current_desc: str, prior_desc: str,
                     current_date: str, prior_date: str) -> list[float]:
    cm, cp = _normalize(current_desc)
    pm, pp = _normalize(prior_desc)

    same_mod = int(bool(cm & pm))
    same_part = int(bool(cp & pp))
    j = _jaccard(current_desc, prior_desc)
    exact = int(current_desc.strip().upper() == prior_desc.strip().upper())
    age = min(_date_diff_years(current_date, prior_date), 15.0)
    n_mod_overlap = len(cm & pm)
    n_part_overlap = len(cp & pp)

    # Additional features that improve accuracy
    both_have_mod = int(bool(cm) and bool(pm))
    both_have_part = int(bool(cp) and bool(pp))
    # Contrast match: "with contrast" vs "without" is a soft signal
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


FEATURE_NAMES = [
    "same_mod", "same_part", "jaccard", "exact", "age_years",
    "n_mod_overlap", "n_part_overlap",
    "both_have_mod", "both_have_part", "same_contrast",
]


def build_dataset(data_path: str):
    with open(data_path) as f:
        data = json.load(f)
    cases = data["cases"]
    truth_map = {
        (t["case_id"], t["study_id"]): t["is_relevant_to_current"]
        for t in data["truth"]
    }
    X, y = [], []
    for case in cases:
        curr_desc = case["current_study"]["study_description"]
        curr_date = case["current_study"]["study_date"]
        for prior in case["prior_studies"]:
            label = truth_map.get((case["case_id"], prior["study_id"]))
            if label is None:
                continue
            feats = extract_features(curr_desc, prior["study_description"],
                                     curr_date, prior["study_date"])
            X.append(feats)
            y.append(int(label))
    return np.array(X), np.array(y)


def train(data_path: str, output_path: str = "model.pkl"):
    print(f"Loading data from {data_path}...")
    X, y = build_dataset(data_path)
    print(f"Dataset: {len(X)} samples, {y.mean():.3f} positive rate")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    print("Running 5-fold cross-validation...")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    print("Training final model on full dataset...")
    clf.fit(X, y)

    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.4f}")

    artifact = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "cv_accuracy": float(scores.mean()),
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved to {output_path}  (CV accuracy: {scores.mean():.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="relevant_priors_public.json")
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.output)
