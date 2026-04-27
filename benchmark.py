"""
Scores structured separation-protocol extractions (JSON) along four dimensions:
  - Extraction accuracy
  - Protocol completeness
  - Hallucination rate
  - Task success

This is intentionally lightweight (stdlib-only) and is designed to complement the
older Q/A-centric rubric in `benchmark.py` and the results aggregation in `scorer.py`.

If a ground truth JSON is provided, accuracy/hallucination are computed by set overlap of
canonicalized entities/conditions. Without a ground truth JSON, accuracy becomes a
field-quality heuristic and hallucination becomes a provenance heuristic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml


# Baseline "labels" (mirrors the legacy question set in `functions.py` / `benchmark.py`)
REQUIRED_SECTIONS: Tuple[str, ...] = (
    "target_materials",
    "acids_and_solvents",
    "resins_or_columns",
    "elution_conditions",
    "final_products",
)


def _safe_get_items(obj: Any) -> List[Dict[str, Any]]:
    """Return section['items'] as list[dict], or [] if missing/malformed."""
    if not isinstance(obj, dict):
        return []
    items = obj.get("items", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
    return out


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\.\-\+\(\)\/]+", re.UNICODE)


_UNICODE_TRANSLATE = str.maketrans(
    {
        "×": "x",
        "−": "-",
        "–": "-",
        "—": "-",
        "·": " ",
        "•": " ",
        "µ": "u",
        "Ω": "ohm",
        "Ω": "ohm",
    }
)


def _normalize_unicode(s: str) -> str:
    # NFKC collapses some compatibility characters; then we apply explicit translations.
    s = unicodedata.normalize("NFKC", s)
    return s.translate(_UNICODE_TRANSLATE)


# Domain mapping: common chemical names -> formula tokens.
_CHEM_SYNONYMS: Dict[str, str] = {
    "hydrochloric acid": "hcl",
    "hydrochloride acid": "hcl",
    "hydrogen chloride": "hcl",
    "nitric acid": "hno3",
    "sulfuric acid": "h2so4",
    "sulphuric acid": "h2so4",
    "ammonium hydroxide": "nh4oh",
    "ammonia solution": "nh4oh",
    "deionized water": "water",
    "ultrapure water": "water",
    "megaohm cm water": "water",
    "mohm cm water": "water",
    "mω cm water": "water",
}

_CHEM_TOKEN_MAP: Dict[str, str] = {
    "hcl": "hcl",
    "hno3": "hno3",
    "h2so4": "h2so4",
    "nh4oh": "nh4oh",
    "water": "water",
}


def canonicalize_chemical_name(s: Any) -> str:
    """
    Canonicalize a chemical name/formula into a compact token when possible.

    Examples:
    - "Hydrochloric acid" -> "hcl"
    - "HCl" -> "hcl"
    - "18 MΩ·cm water" -> "water"

    Falls back to ``canonicalize_text`` when no mapping applies.
    """
    t = canonicalize_text(s)
    if not t:
        return ""
    for phrase, tok in _CHEM_SYNONYMS.items():
        if phrase in t:
            return tok
    for tok in _CHEM_TOKEN_MAP:
        if re.search(rf"\b{re.escape(tok)}\b", t):
            return _CHEM_TOKEN_MAP[tok]
    return t


_ISO_NAME_RE = re.compile(r"^(?P<element>[a-z]+)\s*[- ]\s*(?P<mass>\d{1,3})$", re.IGNORECASE)
_ISO_SYMBOL_RE = re.compile(r"^(?P<mass>\d{1,3})\s*(?P<symbol>[a-z]{1,3})$", re.IGNORECASE)


def canonicalize_isotope(s: Any) -> str:
    """
    Canonicalize isotope identifiers/names into a compact form like "55co".

    Accepts:
    - "55Co", "55 co" -> "55co"
    - "cobalt-55", "cobalt 55" -> "55co" (best-effort)
    """
    t = canonicalize_text(s)
    if not t:
        return ""
    m = _ISO_SYMBOL_RE.match(t.replace(" ", ""))
    if m:
        return f"{int(m.group('mass'))}{m.group('symbol').lower()}"
    m2 = _ISO_NAME_RE.match(t)
    if m2:
        mass = int(m2.group("mass"))
        element = m2.group("element").lower()
        name_to_symbol = {
            "cobalt": "co",
            "nickel": "ni",
            "copper": "cu",
            "thorium": "th",
            "radium": "ra",
            "actinium": "ac",
        }
        sym = name_to_symbol.get(element, element[:2])
        return f"{mass}{sym}"
    return t


def canonicalize_eluent_name(s: Any) -> str:
    """
    Canonicalize an eluent to a chemical token, stripping embedded concentrations.

    Examples:
    - "0.1 M HCl" -> "hcl"
    - "Hydrochloric acid" -> "hcl"
    """
    t = canonicalize_text(s)
    if not t:
        return ""
    t = re.sub(r"^\d+(?:\.\d+)?\s*m\s+", "", t).strip()
    return canonicalize_chemical_name(t)


def canonicalize_text(s: Any) -> str:
    """Lowercase, normalize whitespace/punct; safe for None/non-strings."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = _normalize_unicode(s)
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    # Keep chemical-ish characters; normalize everything else to spaces.
    s = _PUNCT_RE.sub(" ", s)
    # Normalize common resin/column patterns like "1×8", "1-x8", "1 x 8" -> "1x8"
    s = re.sub(r"(\d)\s*[- ]?\s*x\s*[- ]?\s*(\d)", r"\1x\2", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def canonicalize_concentration(s: Any) -> str:
    """
    Normalize common concentration strings ("6 M", "6M", "0.1 M") to a canonical form.
    Leaves non-parsable strings canonicalized as text.
    """
    t = canonicalize_text(s)
    if not t:
        return ""

    # canonical patterns: "6 m", "0.1 m", "concentrated"
    if "concentrated" in t:
        return "concentrated"

    m = re.match(r"^(\d+(?:\.\d+)?)\s*m$", t)
    if m:
        num = float(m.group(1))
        # avoid "6.0 m"
        if math.isclose(num, round(num)):
            return f"{int(round(num))} m"
        return f"{num:g} m"

    # already tokenized like "6m" -> "6 m"
    m2 = re.match(r"^(\d+(?:\.\d+)?)m$", t)
    if m2:
        num = float(m2.group(1))
        if math.isclose(num, round(num)):
            return f"{int(round(num))} m"
        return f"{num:g} m"

    return t


def _item_has_provenance(item: Dict[str, Any]) -> bool:
    src = item.get("source_section")
    if isinstance(src, str) and src.strip():
        return True
    return False


def _item_key_target(item: Dict[str, Any]) -> str:
    name = canonicalize_text(item.get("name"))
    formula = canonicalize_text(item.get("chemical_formula"))
    isotope = canonicalize_isotope(item.get("isotope"))
    parts = [p for p in (isotope, formula, name) if p]
    return "|".join(parts)


def _item_key_acid_solvent(item: Dict[str, Any]) -> str:
    # key entity + key condition(s)
    name = canonicalize_chemical_name(item.get("name"))
    typ = canonicalize_text(item.get("type"))
    conc = canonicalize_concentration(item.get("concentration"))
    parts = [p for p in (name, typ, conc) if p]
    return "|".join(parts)


def _item_key_resin(item: Dict[str, Any]) -> str:
    name = canonicalize_text(item.get("name"))
    material = canonicalize_text(item.get("material"))
    parts = [p for p in (name, material) if p]
    return "|".join(parts)


def _item_key_elution(item: Dict[str, Any]) -> str:
    eluent = canonicalize_eluent_name(item.get("eluent"))
    conc = canonicalize_concentration(item.get("concentration"))
    parts = [p for p in (eluent, conc) if p]
    return "|".join(parts)


def _item_key_product(item: Dict[str, Any]) -> str:
    name = canonicalize_text(item.get("name"))
    isotope = canonicalize_isotope(item.get("isotope"))
    if not isotope:
        # If name is like "cobalt-55", use that as isotope token.
        iso_from_name = canonicalize_isotope(name)
        if re.match(r"^\d{1,3}[a-z]{1,3}$", iso_from_name):
            isotope = iso_from_name
    form = canonicalize_text(item.get("chemical_form"))
    parts = [p for p in (isotope, name, form) if p]
    return "|".join(parts)


SECTION_KEY_FN = {
    "target_materials": _item_key_target,
    "acids_and_solvents": _item_key_acid_solvent,
    "resins_or_columns": _item_key_resin,
    "elution_conditions": _item_key_elution,
    "final_products": _item_key_product,
}


def extract_section_keys(extraction_json: Dict[str, Any], section: str) -> Set[str]:
    fn = SECTION_KEY_FN.get(section)
    if fn is None:
        return set()
    sec_obj = extraction_json.get(section, {})
    items = _safe_get_items(sec_obj)
    keys = set()
    for it in items:
        k = fn(it)
        if k:
            keys.add(k)
    return keys


def count_section_items(extraction_json: Dict[str, Any], section: str) -> int:
    sec_obj = extraction_json.get(section, {})
    return len(_safe_get_items(sec_obj))


def protocol_completeness(extraction_json: Dict[str, Any], required_sections: Sequence[str] = REQUIRED_SECTIONS) -> float:
    r"""
    Compute a coarse "protocol completeness" score for a single extraction JSON.

    This metric is intentionally simple: for each section name in ``required_sections``
    (by default the five canonical sections used throughout this module), it checks
    whether *at least one item* is present under ``extraction_json[section]["items"]``.
    The final score is:

    \[
        \text{completeness} = \frac{\text{\# sections with ≥1 item}}{\text{\# required sections}}
    \]

    This tells you **how many of the expected sections the model populated at all**,
    regardless of *how many* items it added or how correct they are. It is used as a
    gating signal for task success and as a quick sanity-check that a run is producing
    minimally-structured protocols (e.g., not missing resins entirely).
    """
    present = 0
    for sec in required_sections:
        if count_section_items(extraction_json, sec) > 0:
            present += 1
    return present / max(1, len(required_sections))


def _set_prf(pred: Set[str], gold: Set[str], section: Optional[str] = None) -> Dict[str, float]:
    tp, fp, fn = _match_counts(pred, gold, section=section)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _key_component_match(pred_comp: str, gold_comp: str) -> bool:
    if pred_comp == gold_comp:
        return True
    if not pred_comp or not gold_comp:
        return False
    # Relaxed textual match: treat gold as correct if it appears inside predicted text.
    # This is primarily for name-like fields (e.g., "iron" vs "iron 58"), while still
    # requiring that the gold component contain alphabetic characters.
    if re.search(r"[a-z]", gold_comp) and gold_comp in pred_comp:
        return True
    return False


def _extract_isotope_token_from_key(key: str) -> str:
    for part in (p for p in key.split("|") if p):
        if re.match(r"^\d{1,3}[a-z]{1,3}$", part):
            return part
    return ""


def _key_match_relaxed(pred_key: str, gold_key: str, section: Optional[str] = None) -> bool:
    if pred_key == gold_key:
        return True
    if not pred_key or not gold_key:
        return False

    # For isotope-bearing sections, treat isotope agreement as sufficient even when
    # the textual name differs (e.g., "actinium" vs "actinium-228"/"228ac").
    if section in {"target_materials", "final_products"}:
        gold_iso = _extract_isotope_token_from_key(gold_key)
        pred_iso = _extract_isotope_token_from_key(pred_key)
        if gold_iso and pred_iso and gold_iso == pred_iso:
            return True

    pred_parts = [p for p in pred_key.split("|") if p]
    gold_parts = [g for g in gold_key.split("|") if g]
    if not gold_parts:
        return False

    used_pred: Set[int] = set()
    for g in gold_parts:
        matched = False
        for i, p in enumerate(pred_parts):
            if i in used_pred:
                continue
            if _key_component_match(p, g):
                used_pred.add(i)
                matched = True
                break
        if not matched:
            return False
    return True


def _match_counts(pred: Set[str], gold: Set[str], section: Optional[str] = None) -> Tuple[int, int, int]:
    pred_list = sorted(pred)
    gold_list = sorted(gold)
    used_gold: Set[int] = set()
    tp = 0
    fp = 0
    for p in pred_list:
        matched_idx = None
        for i, g in enumerate(gold_list):
            if i in used_gold:
                continue
            if _key_match_relaxed(p, g, section=section):
                matched_idx = i
                break
        if matched_idx is None:
            fp += 1
        else:
            used_gold.add(matched_idx)
            tp += 1
    fn = len(gold_list) - len(used_gold)
    return tp, fp, fn


def _jaccard(pred: Set[str], gold: Set[str], section: Optional[str] = None) -> float:
    tp, fp, fn = _match_counts(pred, gold, section=section)
    union_size = tp + fp + fn
    if union_size == 0:
        return 1.0
    return float(tp / union_size)


def jaccard_similarity(
    pred_json: Dict[str, Any],
    gold_json: Dict[str, Any],
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
) -> Dict[str, Any]:
    r"""
    Compute Jaccard similarity between predicted and gold extractions.

    Jaccard is defined as:

    \[
        J(A, B) = \frac{|A \cap B|}{|A \cup B|}
    \]

    Here, ``A`` and ``B`` are sets of **canonicalized item keys**, where each key
    encodes the important identifying fields for a section (e.g., eluent + concentration
    for elution conditions). We compute:

    - a **per-section Jaccard** over keys in each required section
    - a **micro Jaccard** over the union of all sections, with each key tagged by section

    This metric is more interpretable than F1 for some users, because it directly
    reflects “how much overlap” there is between pred and gold sets, penalizing both
    omissions and hallucinated items.
    """
    by_section: Dict[str, float] = {}
    pred_all: Set[str] = set()
    gold_all: Set[str] = set()
    for sec in required_sections:
        p = extract_section_keys(pred_json, sec)
        g = extract_section_keys(gold_json, sec)
        by_section[sec] = _jaccard(p, g, section=sec)
        pred_all |= {f"{sec}:{x}" for x in p}
        gold_all |= {f"{sec}:{x}" for x in g}
    return {"micro": _jaccard(pred_all, gold_all), "by_section": by_section}


def macro_prf_from_accuracy(acc: Dict[str, Any], required_sections: Sequence[str] = REQUIRED_SECTIONS) -> Dict[str, Any]:
    """
    Macro-average precision/recall/F1 across sections (each section weighted equally).
    Only valid when acc['mode'] == 'gold'.
    """
    if acc.get("mode") != "gold":
        return {"mode": "unsupported"}
    by_section = acc.get("by_section", {})
    ps: List[float] = []
    rs: List[float] = []
    f1s: List[float] = []
    for sec in required_sections:
        s = by_section.get(sec, {})
        ps.append(float(s.get("precision", 0.0)))
        rs.append(float(s.get("recall", 0.0)))
        f1s.append(float(s.get("f1", 0.0)))
    denom = max(1, len(required_sections))
    return {
        "precision": float(sum(ps) / denom),
        "recall": float(sum(rs) / denom),
        "f1": float(sum(f1s) / denom),
    }


def duplicate_rate(
    pred_json: Dict[str, Any],
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
) -> Dict[str, Any]:
    r"""
    Estimate how much the extraction **over-repeats** essentially identical items.

    We treat each section independently, derive canonical keys for each item using the
    same logic as the accuracy scorer, and then compute:

    - ``items``: number of items emitted in that section
    - ``unique``: number of distinct canonical keys
    - ``duplicates``: ``max(0, items - unique)``
    - ``duplicate_rate``: ``duplicates / items`` (or 0.0 if no items)

    The top-level return value aggregates this across all sections into a global
    duplicate rate:

    \[
        \text{duplicate\_rate} =
        \frac{\text{total items} - \text{total unique keys}}{\max(1, \text{total items})}
    \]

    This helps distinguish “verbose but diverse” predictions from ones that simply
    spam near-duplicate rows without adding new information.
    """
    total_items = 0
    total_unique = 0
    by_section: Dict[str, Any] = {}

    for sec in required_sections:
        items = _safe_get_items(pred_json.get(sec, {}))
        keys = extract_section_keys(pred_json, sec)
        n_items = len(items)
        n_unique = len(keys)
        total_items += n_items
        total_unique += n_unique
        by_section[sec] = {
            "items": n_items,
            "unique": n_unique,
            "duplicates": max(0, n_items - n_unique),
            "duplicate_rate": (float(max(0, n_items - n_unique)) / n_items) if n_items else 0.0,
        }

    duplicates = max(0, total_items - total_unique)
    return {
        "duplicate_rate": (float(duplicates) / total_items) if total_items else 0.0,
        "duplicates": duplicates,
        "items_total": total_items,
        "by_section": by_section,
    }


def _nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    return True


def schema_validity(
    pred_json: Dict[str, Any],
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
) -> Dict[str, Any]:
    r"""
    Compute a heuristic **schema validity** score for an extraction JSON.

    For each section, we define a small set of **required fields** that should be
    non-empty for an item to be considered structurally well-formed. Some sections
    use **one-of rules** for identity (e.g. name *or* isotope must be present), plus
    mandatory provenance via ``source_section``.

    For every item in every required section we:

    - evaluate a list of boolean requirements (per section),
    - count how many are satisfied (``scored``) out of how many were checked
      (``possible``),
    - accumulate these counts across sections.

    The top-level ``validity`` scalar is:

    \[
        \text{validity} = \frac{\text{total satisfied field checks}}{\text{total field checks}}
    \]

    This is **agnostic to semantic correctness**; it only checks that the extraction
    filled in the right structural fields. It is useful to differentiate “empty or
    malformed JSON blobs” from predictions that at least adhere to the agreed schema.
    """
    by_section: Dict[str, Any] = {}
    total_scored = 0
    total_possible = 0

    for sec in required_sections:
        items = _safe_get_items(pred_json.get(sec, {}))
        sec_scored = 0
        sec_possible = 0

        for it in items:
            # Define per-section requirements.
            if sec == "target_materials":
                # Need identity + provenance; physical_form is optional.
                has_identity = any(_nonempty(it.get(k)) for k in ("name", "chemical_formula", "isotope"))
                req_bools = [has_identity, _nonempty(it.get("source_section"))]
            elif sec == "acids_and_solvents":
                req_bools = [
                    _nonempty(it.get("name")),
                    _nonempty(it.get("type")),
                    _nonempty(it.get("role")),
                    _nonempty(it.get("source_section")),
                ]
            elif sec == "resins_or_columns":
                req_bools = [
                    _nonempty(it.get("name")),
                    _nonempty(it.get("role")),
                    _nonempty(it.get("source_section")),
                ]
            elif sec == "elution_conditions":
                req_bools = [
                    _nonempty(it.get("eluent")),
                    _nonempty(it.get("source_section")),
                ]
            elif sec == "final_products":
                has_identity = any(_nonempty(it.get(k)) for k in ("name", "isotope"))
                req_bools = [has_identity, _nonempty(it.get("source_section"))]
            else:
                req_bools = []

            scored = sum(1 for b in req_bools if b)
            possible = len(req_bools)
            sec_scored += scored
            sec_possible += possible

        by_section[sec] = {
            "items": len(items),
            "scored": sec_scored,
            "possible": sec_possible,
            "validity": (float(sec_scored) / sec_possible) if sec_possible else 0.0,
        }
        total_scored += sec_scored
        total_possible += sec_possible

    return {
        "validity": (float(total_scored) / total_possible) if total_possible else 0.0,
        "scored": total_scored,
        "possible": total_possible,
        "by_section": by_section,
    }


def extraction_accuracy(
    pred_json: Dict[str, Any],
    gold_json: Optional[Dict[str, Any]] = None,
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
) -> Dict[str, Any]:
    r"""
    Measure how accurately an extraction JSON captures the gold protocol, or,
    in the absence of gold, how well it fills in key fields.

    There are **two operating modes**:

    1. **Gold mode** (``gold_json`` provided):

       - For each section in ``required_sections``, we compute a set of canonical keys
         for predicted items and for gold items.
       - We then compute per-section precision/recall/F1 for these sets.
       - To summarize across sections, we construct a micro-averaged view by taking the
         union of all keys tagged with their section (``"sec:key"``) and computing a
         single PRF triplet.

       The return value has the shape::

           {
             "mode": "gold",
             "micro": { "precision": ..., "recall": ..., "f1": ... },
             "by_section": { section_name: { "precision": ..., "recall": ..., "f1": ... }, ... }
           }

       This mode is used whenever a ground-truth JSON is available.

    2. **Heuristic mode** (no ``gold_json``):

       - For each item, we check whether its “entity identity” fields and (where
         applicable) its “condition” fields are non-empty (e.g., eluent + concentration
         for elution conditions).
       - Items with identity but no condition get partial credit; items missing identity
         get zero.
       - We average these per-item scores across all items and sections to obtain an
         ``overall`` accuracy proxy.

       The return value in this case has the shape::

           {
             "mode": "heuristic",
             "overall": <0-1>,
             "by_section": {
               section_name: {
                 "items": <count>,
                 "avg_item_score": <0-1>
               },
               ...
             },
             "items_scored": <total_items_considered>
           }

    This function is intentionally **field-level** rather than embedding-level: it
    operates on discrete canonical keys so that metrics are reproducible, easy to
    inspect, and can be tied back to specific mis-extracted items.
    """
    if gold_json is not None:
        pred_all: Set[str] = set()
        gold_all: Set[str] = set()
        per_section: Dict[str, Any] = {}
        for sec in required_sections:
            p = extract_section_keys(pred_json, sec)
            g = extract_section_keys(gold_json, sec)
            per_section[sec] = _set_prf(p, g, section=sec)
            pred_all |= {f"{sec}:{x}" for x in p}
            gold_all |= {f"{sec}:{x}" for x in g}

        micro = _set_prf(pred_all, gold_all)
        return {"mode": "gold", "micro": micro, "by_section": per_section}

    # Heuristic mode (no gold):
    # Score each item on whether it has a non-empty entity identifier + condition where applicable.
    # Then average across all sections/items.
    scored = 0
    possible = 0
    by_section: Dict[str, Any] = {}

    def _score_item(has_entity: bool, has_condition: bool) -> float:
        # entity is mandatory; condition is bonus
        if not has_entity:
            return 0.0
        if has_condition:
            return 1.0
        return 0.6

    for sec in required_sections:
        items = _safe_get_items(pred_json.get(sec, {}))
        sec_scores: List[float] = []
        for it in items:
            possible += 1
            if sec == "target_materials":
                has_entity = bool(canonicalize_text(it.get("name")) or canonicalize_text(it.get("isotope")) or canonicalize_text(it.get("chemical_formula")))
                has_condition = bool(canonicalize_text(it.get("physical_form")))
            elif sec == "acids_and_solvents":
                has_entity = bool(canonicalize_text(it.get("name")))
                has_condition = bool(canonicalize_concentration(it.get("concentration")) or canonicalize_text(it.get("role")) or canonicalize_text(it.get("type")))
            elif sec == "resins_or_columns":
                has_entity = bool(canonicalize_text(it.get("name")))
                has_condition = bool(canonicalize_text(it.get("role")) or canonicalize_text(it.get("material")) or canonicalize_text(it.get("column_dimensions")))
            elif sec == "elution_conditions":
                has_entity = bool(canonicalize_text(it.get("eluent")))
                has_condition = bool(canonicalize_concentration(it.get("concentration")) or canonicalize_text(it.get("volume")) or canonicalize_text(it.get("pH")) or canonicalize_text(it.get("flow_rate")))
            elif sec == "final_products":
                has_entity = bool(canonicalize_text(it.get("name")) or canonicalize_text(it.get("isotope")))
                has_condition = bool(canonicalize_text(it.get("chemical_form")))
            else:
                has_entity = False
                has_condition = False

            s = _score_item(has_entity, has_condition)
            sec_scores.append(s)
            scored += s

        by_section[sec] = {
            "items": len(items),
            "avg_item_score": (sum(sec_scores) / len(sec_scores)) if sec_scores else 0.0,
        }

    overall = (scored / possible) if possible else 0.0
    return {"mode": "heuristic", "overall": overall, "by_section": by_section, "items_scored": possible}


def hallucination_rate(
    pred_json: Dict[str, Any],
    gold_json: Optional[Dict[str, Any]] = None,
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
) -> Dict[str, Any]:
    r"""
    Estimate how much an extraction **hallucinates** entities or conditions.

    There are two operating modes:

    1. **Gold mode** (``gold_json`` provided):

       - For each section we build sets of canonical keys for prediction and gold.
       - Any predicted key not in the gold set counts as a **fabrication** (false
         positive).
       - We aggregate across sections to compute:

         - ``fabricated``: total number of fabricated keys
         - ``pred_total``: total number of predicted keys
         - ``rate``: ``fabricated / max(1, pred_total)``

       We also expose per-section counts for debugging.

    2. **Heuristic mode** (no ``gold_json``):

       - For each item, we check two things:

         * the presence of a non-empty ``source_section`` (provenance),
         * the presence of a minimal identity field (e.g., a name or isotope).

       - Items that lack provenance or a core identity are flagged as potential
         hallucinations.

       The overall hallucination rate is:

       \[
           \text{rate} = \frac{\text{\# flagged items}}{\max(1, \text{\# total items})}
       \]

    This is a **penalty metric** complementary to accuracy: a model can have reasonable
    recall but an unacceptably high hallucination rate, which this function is designed
    to surface.
    """
    if gold_json is not None:
        fp = 0
        tp = 0
        fn = 0
        pred_total = 0
        by_section: Dict[str, Any] = {}
        for sec in required_sections:
            p = extract_section_keys(pred_json, sec)
            g = extract_section_keys(gold_json, sec)
            pred_total += len(p)
            sec_tp, sec_fp, sec_fn = _match_counts(p, g, section=sec)
            tp += sec_tp
            fp += sec_fp
            fn += sec_fn
            by_section[sec] = {"pred": len(p), "fabricated": sec_fp, "matched": sec_tp, "missed": sec_fn}
        rate = fp / max(1, pred_total)
        return {
            "mode": "gold",
            "rate": rate,
            "fabricated": fp,
            "pred_total": pred_total,
            "matched": tp,
            "missed": fn,
            "by_section": by_section,
        }

    flagged = 0
    total = 0
    by_section: Dict[str, Any] = {}

    for sec in required_sections:
        items = _safe_get_items(pred_json.get(sec, {}))
        sec_flagged = 0
        for it in items:
            total += 1
            # basic provenance + identity check
            has_src = _item_has_provenance(it)
            has_identity = False
            if sec == "target_materials":
                has_identity = bool(canonicalize_text(it.get("name")) or canonicalize_text(it.get("isotope")) or canonicalize_text(it.get("chemical_formula")))
            elif sec == "acids_and_solvents":
                has_identity = bool(canonicalize_text(it.get("name")))
            elif sec == "resins_or_columns":
                has_identity = bool(canonicalize_text(it.get("name")))
            elif sec == "elution_conditions":
                has_identity = bool(canonicalize_text(it.get("eluent")))
            elif sec == "final_products":
                has_identity = bool(canonicalize_text(it.get("name")) or canonicalize_text(it.get("isotope")))

            if (not has_src) or (not has_identity):
                sec_flagged += 1
                flagged += 1

        by_section[sec] = {"items": len(items), "flagged": sec_flagged}

    rate = flagged / max(1, total)
    return {"mode": "heuristic", "rate": rate, "flagged": flagged, "items_total": total, "by_section": by_section}


@dataclass(frozen=True)
class TaskSuccessThresholds:
    min_completeness: float = 0.8
    min_accuracy: float = 0.65
    max_hallucination_rate: float = 0.25


def task_success(
    completeness: float,
    accuracy: float,
    halluc_rate: float,
    thresholds: TaskSuccessThresholds = TaskSuccessThresholds(),
) -> Dict[str, Any]:
    r"""
    Decide whether an extraction run should be considered an overall **task success**.

    Instead of hard-coding pass/fail everywhere, we centralize the decision here using
    three scalar inputs:

    - ``completeness``: protocol completeness score \(\in [0, 1]\)
    - ``accuracy``: a single scalar derived from either micro F1 (gold mode) or the
      heuristic accuracy (no-gold mode)
    - ``halluc_rate``: hallucination rate \(\in [0, 1]\)

    These are compared to configurable thresholds (``TaskSuccessThresholds``):

    - ``min_completeness`` (default 0.8)
    - ``min_accuracy`` (default 0.7)
    - ``max_hallucination_rate`` (default 0.2)

    The function returns a dict with:

    - ``success``: boolean indicating whether all three criteria are met
    - ``thresholds``: the exact numeric cutoffs used, so downstream consumers can
      log or display them alongside decisions.

    This separation makes it easy to retune success criteria without touching the
    scoring internals, and encourages callers to reason about **trade-offs** between
    coverage, correctness, and hallucination.
    """
    ok = (
        completeness >= thresholds.min_completeness
        and accuracy >= thresholds.min_accuracy
        and halluc_rate <= thresholds.max_hallucination_rate
    )
    return {
        "success": bool(ok),
        "thresholds": {
            "min_completeness": thresholds.min_completeness,
            "min_accuracy": thresholds.min_accuracy,
            "max_hallucination_rate": thresholds.max_hallucination_rate,
        },
    }


def score_extraction_json(
    pred_json: Dict[str, Any],
    gold_json: Optional[Dict[str, Any]] = None,
    required_sections: Sequence[str] = REQUIRED_SECTIONS,
    thresholds: TaskSuccessThresholds = TaskSuccessThresholds(),
) -> Dict[str, Any]:
    """
    End-to-end scoring pipeline for a **single extraction JSON** (with optional gold).

    Given one predicted extraction and an optional gold-standard extraction, this
    function computes all of the core metrics used by the benchmark:

    - ``protocol_completeness``: fraction of required sections with ≥1 item.
    - ``extraction_accuracy``: either gold-aware micro PRF or heuristic field-quality
      accuracy (see ``extraction_accuracy`` for details).
    - ``macro_prf``: macro-averaged precision/recall/F1 across sections (gold mode).
    - ``jaccard_similarity``: Jaccard overlap between predicted and gold keys
      (gold mode).
    - ``hallucination_rate``: gold-based or heuristic hallucination estimate.
    - ``duplicate_rate``: extent to which items are over-repeated.
    - ``schema_validity``: how well items conform to the expected schema.
    - ``task_success``: a boolean “pass/fail” flag based on configurable thresholds.

    It also bundles a small ``meta`` block describing the scoring mode and which
    sections were considered. The intent is that all downstream reporting—the CSV,
    plots, CLI summaries—can be derived purely from this structured result without
    re-implementing any metric logic.
    """
    comp = protocol_completeness(pred_json, required_sections=required_sections)
    any_empty_component = any(count_section_items(pred_json, sec) == 0 for sec in required_sections)
    acc = extraction_accuracy(pred_json, gold_json=gold_json, required_sections=required_sections)
    hall = hallucination_rate(pred_json, gold_json=gold_json, required_sections=required_sections)
    dup = duplicate_rate(pred_json, required_sections=required_sections)
    validity = schema_validity(pred_json, required_sections=required_sections)

    # unify an "accuracy_scalar" for task_success gating
    if acc["mode"] == "gold":
        acc_scalar = float(acc["micro"]["f1"])
    else:
        acc_scalar = float(acc["overall"])

    hall_scalar = float(hall["rate"])

    success = task_success(comp, acc_scalar, hall_scalar, thresholds=thresholds)
    # Hard fail: if any required section has an empty `items` list, task_success is False.
    if any_empty_component:
        success = {**success, "success": False}

    return {
        "protocol_completeness": comp,
        "extraction_accuracy": acc,
        "macro_prf": macro_prf_from_accuracy(acc, required_sections=required_sections),
        "jaccard_similarity": (jaccard_similarity(pred_json, gold_json, required_sections=required_sections) if gold_json is not None else None),
        "hallucination_rate": hall,
        "duplicate_rate": dup,
        "schema_validity": validity,
        "task_success": {
            **success,
            "inputs": {"completeness": comp, "accuracy": acc_scalar, "hallucination_rate": hall_scalar},
            "rules": {"fail_if_any_required_section_empty": True},
            "derived": {"any_required_section_empty": bool(any_empty_component)},
        },
        "meta": {
            "required_sections": list(required_sections),
            "scoring_mode": "gold" if gold_json is not None else "heuristic",
        },
    }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root in {path}")
    return data


def _canonicalize_id(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\.pdf(\.json)?$", "", s)
    s = re.sub(r"\.json$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_doc_fingerprints(obj: Dict[str, Any], fallback_path: Optional[Path] = None) -> Dict[str, str]:
    meta = obj.get("paper_metadata", {}) if isinstance(obj.get("paper_metadata"), dict) else {}
    doi = _canonicalize_id(meta.get("doi"))
    title = _canonicalize_id(meta.get("title"))
    source = _canonicalize_id(meta.get("source"))
    stem = _canonicalize_id(fallback_path.stem if fallback_path else "")
    return {"doi": doi, "title": title, "source": source, "stem": stem}


def _exp_num_from_path(p: Path) -> Optional[int]:
    # Expect .../out/exp18/... ; returns 18
    for part in p.parts:
        m = re.match(r"^exp(\d+)$", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _pred_gold_stem_id(path: Path) -> str:
    # normalize common patterns: "S106....pdf.json" -> "s106..."
    return _canonicalize_id(path.name)


def _gold_stem_id(path: Path) -> str:
    # gold files are already .json, but normalize similarly
    return _canonicalize_id(path.name)


def _is_extraction_json(obj: Dict[str, Any], required_sections: Sequence[str] = REQUIRED_SECTIONS) -> bool:
    # Filter out non-extraction JSONs (e.g., timing/config) found under out/
    for sec in required_sections:
        v = obj.get(sec)
        if isinstance(v, dict) and isinstance(v.get("items"), list):
            return True
    return False


def _best_match_gold(
    pred_path: Path,
    pred_obj: Dict[str, Any],
    gold_index_by_doi: Dict[str, Path],
    gold_fingerprints: Dict[Path, Dict[str, str]],
    min_similarity: float = 0.65,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Prefer DOI exact match. Otherwise, fuzzy match using title/source/stem.
    Returns (gold_path|None, debug_info).
    """
    pred_fp = _extract_doc_fingerprints(pred_obj, fallback_path=pred_path)
    debug: Dict[str, Any] = {"pred_fingerprints": pred_fp, "mode": None}

    if pred_fp["doi"] and pred_fp["doi"] in gold_index_by_doi:
        gp = gold_index_by_doi[pred_fp["doi"]]
        debug["mode"] = "doi"
        debug["match"] = {"doi": pred_fp["doi"], "gold_path": str(gp)}
        return gp, debug

    candidates: List[Tuple[float, Path, str]] = []
    pred_keys = [pred_fp["title"], pred_fp["source"], pred_fp["stem"]]
    pred_keys = [k for k in pred_keys if k]

    if not pred_keys:
        debug["mode"] = "none"
        debug["match"] = None
        return None, debug

    for gold_path, gfp in gold_fingerprints.items():
        gold_keys = [gfp.get("title", ""), gfp.get("source", ""), gfp.get("stem", "")]
        best = 0.0
        best_reason = ""
        for pk in pred_keys:
            for gk in gold_keys:
                if not pk or not gk:
                    continue
                score = SequenceMatcher(None, pk, gk).ratio()
                if score > best:
                    best = score
                    best_reason = f"{pk}~{gk}"
        candidates.append((best, gold_path, best_reason))

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_path, best_reason = candidates[0]
    debug["mode"] = "fuzzy"
    debug["match"] = {"similarity": best_score, "reason": best_reason, "gold_path": str(best_path)}

    if best_score < min_similarity:
        return None, debug
    return best_path, debug


def score_out_vs_ground_truth(
    out_dir: Path,
    ground_truth_dir: Path,
    include_kinds: Sequence[str] = ("rag_output", "full_summary", "section_summaries"),
    min_similarity: float = 0.65,
    exp_min: Optional[int] = None,
    exp_max: Optional[int] = None,
    exps: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """
    Batch-score many prediction JSONs in an ``out/`` tree against a ``ground_truth/`` directory.

    This is the main entry point for **experiment-wide benchmarking**. It:

    1. Loads all ground-truth JSON files from ``ground_truth_dir`` and indexes them
       by DOI and by normalized filename stem.
    2. Discovers prediction JSONs under ``out_dir`` in the specified ``include_kinds``
       subdirectories (typically ``rag_output`` and ``full_summary``).
    3. Optionally filters predictions to a subset of experiments via:

       - ``exp_min`` / ``exp_max``: include only ``out/expN`` where
         ``exp_min ≤ N ≤ exp_max``.
       - ``exps``: explicit set of experiment numbers, which overrides the range
         filters if provided.

    4. For each prediction file:

       - matches it to a gold file by normalized filename stem, falling back to
         DOI-based or fuzzy title/source matching when necessary,
       - loads per-experiment metadata (model name, timing) from ``config.yaml`` and
         ``timing.json`` in the corresponding experiment directory,
       - runs ``score_extraction_json`` to compute all core metrics,
       - records a rich per-file record including experiment id, method, model,
         runtime, DOI, and metric bundle.

    The returned dict has two main parts:

    - ``aggregate``: experiment-level counts and mean metrics (micro/macro F1, Jaccard,
      hallucination, completeness, duplicate rate, schema validity, success rate).
    - ``per_file``: one entry per prediction file, suitable for CSV export and
      visualization.

    This function deliberately encapsulates **all filesystem and matching logic** so
    that the rest of the codebase can treat scoring as pure data processing.
    """
    gold_paths = sorted(ground_truth_dir.glob("*.json"))
    gold_index_by_doi: Dict[str, Path] = {}
    gold_fingerprints: Dict[Path, Dict[str, str]] = {}
    gold_index_by_stem: Dict[str, List[Path]] = {}

    for gp in gold_paths:
        try:
            gobj = _load_json(gp)
        except Exception:
            continue
        gfp = _extract_doc_fingerprints(gobj, fallback_path=gp)
        gold_fingerprints[gp] = gfp
        if gfp["doi"]:
            gold_index_by_doi[gfp["doi"]] = gp
        sid = _gold_stem_id(gp)
        gold_index_by_stem.setdefault(sid, []).append(gp)

    pred_paths: List[Path] = []
    for kind in include_kinds:
        pred_paths.extend(out_dir.glob(f"**/{kind}/*.json"))
    pred_paths = sorted(set(pred_paths))

    def _exp_allowed(p: Path) -> bool:
        n = _exp_num_from_path(p)
        if n is None:
            # If it isn't under an exp folder, exclude by default when filtering.
            return (exp_min is None and exp_max is None and not exps)
        if exps is not None:
            return n in exps
        if exp_min is not None and n < exp_min:
            return False
        if exp_max is not None and n > exp_max:
            return False
        return True

    pred_paths = [p for p in pred_paths if _exp_allowed(p)]

    per_file: List[Dict[str, Any]] = []
    matched = 0
    skipped = 0
    micro_precisions: List[float] = []
    micro_recalls: List[float] = []
    micro_f1s: List[float] = []
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []
    jaccards: List[float] = []
    halluc_rates: List[float] = []
    completeness: List[float] = []
    dup_rates: List[float] = []
    validity_scores: List[float] = []
    successes: List[bool] = []

    # Cache per-experiment metadata (model name, per-file runtimes).
    exp_meta: Dict[int, Dict[str, Any]] = {}

    def _load_exp_meta(exp_num: int) -> Dict[str, Any]:
        if exp_num in exp_meta:
            return exp_meta[exp_num]
        base = out_dir / f"exp{exp_num}"
        model_name: Optional[str] = None
        timing: Dict[str, Dict[str, float]] = {}

        # Model name from config.yaml (best-effort).
        cfg = base / "config.yaml"
        if cfg.exists():
            try:
                with cfg.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    if isinstance(data.get("model"), str):
                        model_name = data["model"]
                    else:
                        # fall back: search for first string value under keys containing 'model'
                        for k, v in data.items():
                            if "model" in str(k).lower() and isinstance(v, str):
                                model_name = v
                                break
            except Exception:
                pass

        # Timing from timing.json (multiple JSON objects concatenated).
        tj = base / "timing.json"
        if tj.exists():
            try:
                text = tj.read_text(encoding="utf-8").strip()
                if text:
                    # Make it a JSON array.
                    fixed = "[" + text.replace("}{", "},{") + "]"
                    arr = json.loads(fixed)
                    if isinstance(arr, list):
                        for obj in arr:
                            if not isinstance(obj, dict):
                                continue
                            fname = obj.get("file")
                            if not isinstance(fname, str):
                                continue
                            stem = _canonicalize_id(Path(fname).name)
                            timing[stem] = {
                                "section_summaries": float(obj.get("section_summary_time", 0.0) or 0.0),
                                "full_summary": float(obj.get("full_summary_time", 0.0) or 0.0),
                                "rag_output": float(obj.get("rag_time", 0.0) or 0.0),
                            }
            except Exception:
                pass

        exp_meta[exp_num] = {"model": model_name, "timing": timing}
        return exp_meta[exp_num]

    for pp in pred_paths:
        try:
            pobj = _load_json(pp)
        except Exception as e:
            skipped += 1
            per_file.append({"pred_path": str(pp), "error": str(e)})
            continue

        if not _is_extraction_json(pobj):
            skipped += 1
            per_file.append({"pred_path": str(pp), "error": "not_extraction_json"})
            continue

        # Prefer filename stem match; fall back to DOI/fuzzy matching if needed.
        stem_id = _pred_gold_stem_id(pp)
        stem_matches = gold_index_by_stem.get(stem_id, [])
        if len(stem_matches) == 1:
            gold_path = stem_matches[0]
            match_debug = {"mode": "filename", "stem_id": stem_id, "gold_path": str(gold_path)}
        elif len(stem_matches) > 1:
            # Disambiguate by fuzzy similarity on the filename itself.
            scored = [(SequenceMatcher(None, stem_id, _gold_stem_id(gp)).ratio(), gp) for gp in stem_matches]
            scored.sort(key=lambda t: t[0], reverse=True)
            gold_path = scored[0][1]
            match_debug = {"mode": "filename_ambiguous", "stem_id": stem_id, "candidates": [str(p) for _, p in scored[:5]], "gold_path": str(gold_path)}
        else:
            gold_path, match_debug = _best_match_gold(
                pred_path=pp,
                pred_obj=pobj,
                gold_index_by_doi=gold_index_by_doi,
                gold_fingerprints=gold_fingerprints,
                min_similarity=min_similarity,
            )

        exp_num = _exp_num_from_path(pp)
        exp_info = _load_exp_meta(exp_num) if exp_num is not None else {"model": None, "timing": {}}
        model_name = exp_info.get("model")
        timing_map = exp_info.get("timing") or {}
        stem_for_timing = _canonicalize_id(Path(pp).name.replace(".json", ""))
        file_id = Path(pp).name
        if file_id.lower().endswith(".json"):
            file_id = file_id[:-5]
        doi = None

        if gold_path is None:
            # In this benchmark, every prediction is expected to have a gold counterpart.
            raise SystemExit(f"No ground truth match for prediction: {pp}")

        matched += 1
        gobj = _load_json(gold_path)
        meta = gobj.get("paper_metadata", {}) if isinstance(gobj.get("paper_metadata"), dict) else {}
        doi = meta.get("doi") if isinstance(meta.get("doi"), str) else None
        report = score_extraction_json(pobj, gold_json=gobj)

        # Scalars for aggregation.
        prec = float(report["extraction_accuracy"]["micro"]["precision"])
        rec = float(report["extraction_accuracy"]["micro"]["recall"])
        f1 = float(report["extraction_accuracy"]["micro"]["f1"])
        macro = report.get("macro_prf") or {}
        macro_p = float(macro.get("precision", 0.0))
        macro_r = float(macro.get("recall", 0.0))
        macro_f = float(macro.get("f1", 0.0))
        jac = float((report.get("jaccard_similarity") or {}).get("micro", 0.0))
        hall = float(report["hallucination_rate"]["rate"])
        comp = float(report["protocol_completeness"])
        dup = float((report.get("duplicate_rate") or {}).get("duplicate_rate", 0.0))
        val = float((report.get("schema_validity") or {}).get("validity", 0.0))
        succ = bool(report["task_success"]["success"])
        micro_precisions.append(prec)
        micro_recalls.append(rec)
        micro_f1s.append(f1)
        macro_precisions.append(macro_p)
        macro_recalls.append(macro_r)
        macro_f1s.append(macro_f)
        jaccards.append(jac)
        halluc_rates.append(hall)
        completeness.append(comp)
        dup_rates.append(dup)
        validity_scores.append(val)
        successes.append(succ)

        kind = next((k for k in include_kinds if k in pp.parts), None)
        if kind == "rag_output":
            method = "RAG"
        elif kind == "section_summaries":
            method = "SECTION_SUMMARIES"
        elif kind == "full_summary":
            method = "FULL_SUMMARY"
        else:
            method = None
        rt = None
        if stem_for_timing in timing_map and kind in timing_map[stem_for_timing]:
            rt = float(timing_map[stem_for_timing][kind])

        per_file.append(
            {
                "pred_path": str(pp),
                "gold_path": str(gold_path),
                "kind": kind,
                "exp": exp_num,
                "file": file_id,
                "doi": doi,
                "model": model_name,
                "runtime_seconds": rt,
                "method": method,
                "match_debug": match_debug,
                "report": report,
            }
        )

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    aggregate = {
        "counts": {
            "gold_files": len(gold_paths),
            "pred_files_considered": len(pred_paths),
            "matched": matched,
            "skipped": skipped,
        },
        "metrics_on_matched": {
            "micro_precision_mean": _mean(micro_precisions),
            "micro_recall_mean": _mean(micro_recalls),
            "micro_f1_mean": _mean(micro_f1s),
            "macro_precision_mean": _mean(macro_precisions),
            "macro_recall_mean": _mean(macro_recalls),
            "macro_f1_mean": _mean(macro_f1s),
            "jaccard_micro_mean": _mean(jaccards),
            "hallucination_rate_mean": _mean(halluc_rates),
            "protocol_completeness_mean": _mean(completeness),
            "duplicate_rate_mean": _mean(dup_rates),
            "schema_validity_mean": _mean(validity_scores),
            "task_success_rate": (float(sum(1 for s in successes if s)) / len(successes)) if successes else 0.0,
        },
    }

    return {
        "meta": {
            "out_dir": str(out_dir),
            "ground_truth_dir": str(ground_truth_dir),
            "include_kinds": list(include_kinds),
            "min_similarity": float(min_similarity),
            "exp_min": exp_min,
            "exp_max": exp_max,
            "exps": sorted(exps) if exps is not None else None,
        },
        "aggregate": aggregate,
        "per_file": per_file,
    }


def _flatten_report_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a per_file entry into simple scalar columns for CSV.
    """
    out: Dict[str, Any] = {
        "exp": row.get("exp"),
        "file": row.get("file"),
        "doi": row.get("doi"),
        "method": row.get("method"),
        "model": row.get("model"),
        "runtime_seconds": row.get("runtime_seconds"),
    }
    # metadata
    report = row.get("report") or {}

    out["protocol_completeness"] = report.get("protocol_completeness")
    out["hallucination_rate"] = (report.get("hallucination_rate") or {}).get("rate")
    out["task_success"] = (report.get("task_success") or {}).get("success")

    acc = report.get("extraction_accuracy") or {}
    micro = acc.get("micro") or {}
    out["micro_precision"] = micro.get("precision")
    out["micro_recall"] = micro.get("recall")
    out["micro_f1"] = micro.get("f1")
    out["micro_accuracy"] = micro.get("accuracy")

    macro = report.get("macro_prf") or {}
    out["macro_precision"] = macro.get("precision")
    out["macro_recall"] = macro.get("recall")
    out["macro_f1"] = macro.get("f1")

    jac = report.get("jaccard_similarity")
    out["jaccard_micro"] = (jac or {}).get("micro") if isinstance(jac, dict) else None

    dup = report.get("duplicate_rate") or {}
    out["duplicate_rate"] = dup.get("duplicate_rate")
    out["duplicates"] = dup.get("duplicates")
    out["items_total"] = dup.get("items_total")

    val = report.get("schema_validity") or {}
    out["schema_validity"] = val.get("validity")

    return out


def write_csv_report(report: Dict[str, Any], csv_path: Path) -> None:
    """
    Materialize the per-file portion of a batch scoring report as a flat CSV.

    The input ``report`` is expected to be the dict returned by
    ``score_out_vs_ground_truth``. This function:

    - flattens each ``per_file`` entry via ``_flatten_report_for_csv`` into a dict
      with only scalar values and a stable set of keys,
    - writes a header row describing those fields,
    - emits one row per prediction/gold pair.

    The resulting CSV is designed to be **analysis-friendly**: you can load it into
    pandas, a spreadsheet, or any plotting tool and immediately group by experiment,
    model, method, or DOI without having to understand the nested JSON structure.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_report_for_csv(r) for r in (report.get("per_file") or [])]
    # stable columns
    fieldnames = [
        "exp",
        "file",
        "doi",
        "model",
        "runtime_seconds",
        "method",
        "protocol_completeness",
        "hallucination_rate",
        "task_success",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "jaccard_micro",
        "duplicate_rate",
        "duplicates",
        "items_total",
        "schema_validity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_plots(report: Dict[str, Any], plots_dir: Path, plot_exp: Optional[int] = None) -> None:
    """
    Generate a suite of comparison plots (boxplots + scatters) from a batch scoring report.

    This helper turns the CSV-like ``per_file`` portion of the report into a pandas
    DataFrame and then produces:

    The plot suite is designed for *model/method comparison*, not distribution archaeology.
    Concretely, it produces:

    - **boxplots grouped by extraction method** for the key quality metrics
      (e.g. micro F1, Jaccard, hallucination rate) and runtime,
    - a **scatter plot of runtime vs quality** (e.g. runtime_seconds vs micro_f1),
      colored by method, to visualize the speed/quality trade-off,
    - a small number of **summary bars** (e.g. task success rate).

    All plots are:

    - restricted to valid pred–gold pairs (this benchmark assumes all rows are matched),
    - optionally further restricted to a single experiment number when ``plot_exp`` is
      provided (matching the integer ``exp`` column),
    - saved as PNG files under ``plots_dir`` with stable, human-readable filenames.

    Axis labels and titles are kept explicit so that the plots are interpretable even
    when viewed out of context (e.g., in a notebook or slide deck).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Avoid "More than 20 figures opened" warnings when generating many exp-specific plots.
    mpl.rcParams["figure.max_open_warning"] = 0

    rows = [_flatten_report_for_csv(r) for r in (report.get("per_file") or [])]
    if not rows:
        return
    df = pd.DataFrame(rows)
    if plot_exp is not None and "exp" in df.columns:
        df = df[df["exp"] == plot_exp].copy()
    if df.empty:
        return

    # Ensure we don't retain figures between calls (especially in plot_each_exp loops).
    plt.close("all")

    def _boxplot_by_method(metric: str, ylabel: str, fname: str, ylim: Optional[Tuple[float, float]] = None) -> None:
        if "method" not in df.columns or metric not in df.columns:
            return
        tmp = df[["method", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric, "method"])
        if tmp.empty:
            return
        plt.figure(figsize=(7, 4))
        tmp.boxplot(column=metric, by="method", grid=False)
        plt.suptitle("")  # remove pandas default supertitle
        title = f"{metric.replace('_', ' ')} by method"
        if plot_exp is not None:
            title += f" (exp{plot_exp})"
        plt.title(title)
        plt.xlabel("method")
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(plots_dir / fname, dpi=160)
        plt.close()

    # Core quality metrics by method (boxplots).
    _boxplot_by_method("micro_f1", ylabel="micro F1", fname="box_micro_f1_by_method.png", ylim=(0, 1))
    _boxplot_by_method("jaccard_micro", ylabel="Jaccard (micro)", fname="box_jaccard_micro_by_method.png", ylim=(0, 1))
    _boxplot_by_method("hallucination_rate", ylabel="hallucination rate", fname="box_hallucination_rate_by_method.png", ylim=(0, 1))
    _boxplot_by_method("schema_validity", ylabel="schema validity", fname="box_schema_validity_by_method.png", ylim=(0, 1))
    _boxplot_by_method("protocol_completeness", ylabel="protocol completeness", fname="box_protocol_completeness_by_method.png", ylim=(0, 1))
    _boxplot_by_method("duplicate_rate", ylabel="duplicate rate", fname="box_duplicate_rate_by_method.png", ylim=(0, 1))

    # Time by method (boxplot).
    _boxplot_by_method("runtime_seconds", ylabel="runtime (seconds)", fname="box_runtime_seconds_by_method.png")

    # Task success rate (bar chart).
    if "task_success" in df.columns:
        s = df["task_success"].astype("bool")
        success_rate = float(s.mean()) if len(s) else 0.0
        plt.figure(figsize=(4.5, 4))
        plt.bar(["success_rate"], [success_rate])
        plt.ylim(0, 1)
        plt.ylabel("rate")
        title = "task success rate"
        if plot_exp is not None:
            title += f" (exp{plot_exp})"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plots_dir / "task_success_rate.png", dpi=160)
        plt.close()

    # Runtime vs quality trade-off (scatter), colored by method.
    if "runtime_seconds" in df.columns and "micro_f1" in df.columns and "method" in df.columns:
        x = pd.to_numeric(df["runtime_seconds"], errors="coerce")
        y = pd.to_numeric(df["micro_f1"], errors="coerce")
        m = df["method"].astype(str)
        ok = x.notna() & y.notna() & df["method"].notna()
        if ok.any():
            plt.figure(figsize=(7, 4.5))
            methods = sorted(set(m[ok].tolist()))
            for meth in methods:
                mask = ok & (m == meth)
                plt.scatter(x[mask], y[mask], alpha=0.8, label=meth)
            plt.xlabel("runtime (seconds)")
            plt.ylabel("micro F1")
            title = "runtime vs micro F1 (colored by method)"
            if plot_exp is not None:
                title += f" (exp{plot_exp})"
            plt.title(title)
            plt.ylim(0, 1)
            plt.legend(title="method", loc="best")
            plt.tight_layout()
            plt.savefig(plots_dir / "scatter_runtime_vs_micro_f1_by_method.png", dpi=160)
            plt.close()

    # Runtime by method (mean bars) — keep as a quick, glanceable summary alongside boxplots.
    if "method" in df.columns and "runtime_seconds" in df.columns:
        rt = pd.to_numeric(df["runtime_seconds"], errors="coerce")
        ok = df["method"].notna() & rt.notna()
        if ok.any():
            by_method = (
                df.loc[ok, ["method"]]
                .assign(runtime_seconds=rt[ok])
                .groupby("method")["runtime_seconds"]
                .mean()
                .sort_values()
            )
            plt.figure(figsize=(6, 4))
            by_method.plot(kind="bar")
            plt.ylabel("runtime (seconds)")
            plt.xlabel("method")
            title = "average runtime by method"
            if plot_exp is not None:
                title += f" (exp{plot_exp})"
            plt.title(title)
            plt.tight_layout()
            fname = "runtime_by_method.png" if plot_exp is None else f"runtime_by_method_exp{plot_exp}.png"
            plt.savefig(plots_dir / fname, dpi=160)
            plt.close()
            
def plot_each_exp(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Generate **per-experiment** plot suites under ``{plots_dir}/exp{i}``.

    This is a thin wrapper around ``write_plots``:

    - It discovers all distinct experiment numbers present in ``report["per_file"]``.
    - For each experiment ``i``, it creates a directory ``plots_dir / f"exp{i}"``.
    - It calls ``write_plots(report, plots_dir=..., plot_exp=i)`` so that *all* plots
      are generated using only rows from that experiment.

    This is meant to be run in addition to the global plots, so you can quickly
    compare distribution shifts across experiments (e.g., model changes).
    """
    per_file = report.get("per_file") or []
    exps: Set[int] = set()
    for r in per_file:
        n = r.get("exp")
        if isinstance(n, int):
            exps.add(n)
    for n in sorted(exps):
        exp_dir = plots_dir / f"exp{n}"
        write_plots(report, plots_dir=exp_dir, plot_exp=n)

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for running the benchmark script.

    There are two usage modes:

    1. **Single-file mode** (no ``--out-dir``):

       - ``--pred``: path to a single prediction JSON.
       - ``--gold``: optional path to its corresponding gold JSON.

       In this mode we simply load both JSONs (when gold is provided), call
       ``score_extraction_json``, and print the resulting report JSON to stdout
       (pretty-printed when ``--pretty`` is given).

    2. **Batch mode** (``--out-dir`` provided):

       - ``--out-dir``: root of the experiment outputs (e.g. ``out``).
       - ``--ground-truth-dir``: directory containing gold JSON files.
       - ``--include-kinds``: which subdirectories under each ``expN`` to treat as
         extraction outputs (e.g. ``rag_output,full_summary``).
       - ``--exp-min`` / ``--exp-max`` / ``--exps``: optional experiment filters.
       - ``--bench-dir``: directory where the CSV and plots will be written
         (defaults to ``./benchmarks``).
       - ``--csv-path`` / ``--write-report``: optional override for the CSV path.
       - ``--plots-dir``: optional override for the plots directory.
       - ``--plot-exp``: if set, plots only use rows from that experiment number.

       Batch mode delegates to ``score_out_vs_ground_truth``, then:

       - writes a CSV via ``write_csv_report``,
       - generates plots via ``write_plots``,
       - prints a compact JSON summary of aggregate metrics to stdout.

    The function returns an integer exit code suitable for use in shell scripts
    (0 on success, or raises ``SystemExit`` with an error message for misuse).
    """
    ap = argparse.ArgumentParser(description="Score separation-protocol extraction JSON.")
    ap.add_argument("--pred", required=False, default=None, help="Path to predicted extraction JSON.")
    ap.add_argument("--gold", required=False, default=None, help="Optional path to gold extraction JSON.")
    ap.add_argument("--out-dir", required=False, default=None, help="Score all predictions under this out/ directory.")
    ap.add_argument("--ground-truth-dir", required=False, default="ground_truth", help="Directory of gold JSON files.")
    ap.add_argument(
        "--include-kinds",
        required=False,
        default="rag_output,full_summary,section_summaries",
        help="Comma-separated subdirectory names under out/ to include (default: rag_output,full_summary).",
    )
    ap.add_argument(
        "--min-similarity",
        required=False,
        default=0.65,
        type=float,
        help="Minimum fuzzy-match similarity when DOI is missing (default: 0.65).",
    )
    ap.add_argument(
        "--write-report",
        required=False,
        default=None,
        help="Deprecated alias for --csv-path (writes CSV).",
    )
    ap.add_argument(
        "--csv-path",
        required=False,
        default=None,
        help="Optional path to write per-file CSV report (recommended for --out-dir mode).",
    )
    ap.add_argument(
        "--plots-dir",
        required=False,
        default=None,
        help="Optional directory to save PNG plots (requires pandas+matplotlib).",
    )
    ap.add_argument(
        "--exp-min",
        required=False,
        default=None,
        type=int,
        help="Only include out/expN where N >= this value.",
    )
    ap.add_argument(
        "--exp-max",
        required=False,
        default=None,
        type=int,
        help="Only include out/expN where N <= this value.",
    )
    ap.add_argument(
        "--exps",
        required=False,
        default=None,
        help="Comma-separated experiment numbers to include (overrides --exp-min/--exp-max). Example: 18,20,21",
    )
    ap.add_argument(
        "--bench-dir",
        required=False,
        default="benchmarks",
        help="Directory to write benchmark artifacts (CSV, plots). Defaults to ./benchmarks.",
    )
    ap.add_argument(
        "--plot-exp",
        required=False,
        default=None,
        type=int,
        help="If set, plots only metrics for this experiment number (expN).",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    args = ap.parse_args(list(argv) if argv is not None else None)
    
    

    if args.out_dir:
        out_dir = Path(args.out_dir)
        gt_dir = Path(args.ground_truth_dir)
        include_kinds = [s.strip() for s in str(args.include_kinds).split(",") if s.strip()]
        bench_dir = Path(args.bench_dir)
        bench_dir.mkdir(parents=True, exist_ok=True)
        exps: Optional[Set[int]] = None
        if args.exps:
            exps = set()
            for tok in str(args.exps).split(","):
                tok = tok.strip()
                if tok:
                    exps.add(int(tok))
        report = score_out_vs_ground_truth(
            out_dir=out_dir,
            ground_truth_dir=gt_dir,
            include_kinds=include_kinds,
            min_similarity=float(args.min_similarity),
            exp_min=args.exp_min,
            exp_max=args.exp_max,
            exps=exps,
        )

        csv_path = Path(args.csv_path) if args.csv_path else (Path(args.write_report) if args.write_report else bench_dir / "benchmark_report.csv")
        if csv_path:
            write_csv_report(report, csv_path=csv_path)

        plots_dir = Path(args.plots_dir) if args.plots_dir else (bench_dir / "plots")
        write_plots(report, plots_dir=plots_dir, plot_exp=args.plot_exp)
        # By default, also generate per-experiment plot suites under {plots_dir}/exp{i}.
        plot_each_exp(report, plots_dir=plots_dir)

        # Always print a concise summary to stdout in batch mode.
        agg = report["aggregate"]
        m = agg["metrics_on_matched"]
        c = agg["counts"]
        summary = {
            "matched": c["matched"],
            "skipped": c["skipped"],
            "micro_precision_mean": m["micro_precision_mean"],
            "micro_recall_mean": m["micro_recall_mean"],
            "micro_f1_mean": m["micro_f1_mean"],
            "macro_f1_mean": m["macro_f1_mean"],
            "jaccard_micro_mean": m["jaccard_micro_mean"],
            "hallucination_rate_mean": m["hallucination_rate_mean"],
            "protocol_completeness_mean": m["protocol_completeness_mean"],
            "duplicate_rate_mean": m["duplicate_rate_mean"],
            "schema_validity_mean": m["schema_validity_mean"],
            "task_success_rate": m["task_success_rate"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True) if args.pretty else json.dumps(summary))
        return 0

    if not args.pred:
        raise SystemExit("Must provide either --pred (single-file mode) or --out-dir (batch mode).")

    pred_path = Path(args.pred)
    gold_path = Path(args.gold) if args.gold else None

    pred = _load_json(pred_path)
    gold = _load_json(gold_path) if gold_path else None

    report = score_extraction_json(pred, gold_json=gold)
    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

