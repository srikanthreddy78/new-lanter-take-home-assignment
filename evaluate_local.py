"""
Local evaluator — runs the full prediction pipeline against the public split
WITHOUT making network calls (uses the rule-based logic only so you can test fast).

Usage:
  python evaluate_local.py --data relevant_priors_public.json
"""

import argparse
import json
import sys
from collections import defaultdict


# ── Copy the rule logic from app.py so we can run offline ──────────────────

BODY_PART_GROUPS = [
    {"BRAIN", "HEAD", "CRANIAL", "INTRACRANIAL", "CEREBR", "SKULL"},
    {"CHEST", "LUNG", "THORAX", "THORACIC", "PULMON", "MEDIASTIN"},
    {"ABDOMEN", "ABD", "ABDOMINAL"},
    {"PELV", "PELVIC"},
    {"ABD", "ABDOMEN", "PELV", "PELVIC", "ABDOM"},
    {"SPINE", "SPINAL", "VERTEBR"},
    {"LUMBAR", "LUMB", "LSP", "L-SPINE"},
    {"CERVICAL", "C-SPINE", "CSPINE", "NECK"},
    {"THORACIC", "TSPINE", "T-SPINE"},
    {"BREAST", "MAM", "MAMMO", "MAMMOGRAPH"},
    {"HEART", "CARDIAC", "CARDIO", "CORONARY", "MYO", "MYOCARD", "VENTRIC"},
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

MODALITY_GROUPS = [
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
]


def _normalize(desc):
    text = " " + desc.upper() + " "
    mods, parts = set(), set()
    for i, group in enumerate(MODALITY_GROUPS):
        for kw in group:
            if kw in text:
                mods.add(i)
                break
    for i, group in enumerate(BODY_PART_GROUPS):
        for kw in group:
            if kw in text:
                parts.add(i)
                break
    return frozenset(mods), frozenset(parts)


def _jaccard(a, b):
    ta, tb = set(a.upper().split()), set(b.upper().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rule_based_classify(current_desc, prior_desc):
    if current_desc.strip().upper() == prior_desc.strip().upper():
        return True, "exact_match"
    curr_mod, curr_part = _normalize(current_desc)
    prior_mod, prior_part = _normalize(prior_desc)
    same_mod = bool(curr_mod & prior_mod)
    same_part = bool(curr_part & prior_part)
    j = _jaccard(current_desc, prior_desc)
    if j >= 0.7:
        return True, f"high_jaccard={j:.2f}"
    if same_mod and same_part:
        return True, "same_mod+same_part"
    if same_part and not same_mod:
        if j >= 0.2:
            return True, f"same_part+jaccard={j:.2f}"
        return None, "ambiguous_same_part"
    if same_mod and not same_part:
        if j >= 0.4:
            return None, "ambiguous_same_mod"
        return False, "same_mod_diff_part"
    if j < 0.1:
        return False, "no_overlap"
    return None, "ambiguous_jaccard"


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(data_path):
    with open(data_path) as f:
        data = json.load(f)

    cases = data["cases"]
    truth_map = {
        (t["case_id"], t["study_id"]): t["is_relevant_to_current"]
        for t in data["truth"]
    }

    correct = 0
    total = 0
    reason_stats = defaultdict(lambda: [0, 0])  # reason → [correct, total]
    ambiguous = 0

    for case in cases:
        case_id = case["case_id"]
        curr = case["current_study"]["study_description"]
        for prior in case["prior_studies"]:
            prior_id = prior["study_id"]
            prior_desc = prior["study_description"]
            label = truth_map.get((case_id, prior_id))
            if label is None:
                continue

            pred, reason = rule_based_classify(curr, prior_desc)
            if pred is None:
                # For local eval: ambiguous → True (body-part bucket default)
                pred = True
                reason = reason + "_defaultTrue"
                ambiguous += 1

            total += 1
            is_correct = (pred == label)
            if is_correct:
                correct += 1
            reason_stats[reason][0] += is_correct
            reason_stats[reason][1] += 1

    print(f"\n{'='*55}")
    print(f"TOTAL ACCURACY: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.2f}%)")
    print(f"Ambiguous (defaulted True): {ambiguous}")
    print(f"\nBreakdown by rule:")
    for reason, (c, t) in sorted(reason_stats.items(), key=lambda x: -x[1][1]):
        print(f"  {reason:<35} {c:5d}/{t:5d} = {c/t:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="relevant_priors_public.json")
    args = parser.parse_args()
    evaluate(args.data)
