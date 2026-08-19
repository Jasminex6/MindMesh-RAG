"""Medical query normalization — safe preprocessing that preserves clinical meaning.

Rules (in priority order):
1. Preserve exact clinical acronyms: ICS, SABA, LABA, MART, FeNO, FEV1, PEF, OCS, AERD, EIB.
2. Preserve drug names and numeric values with units (mg, mcg, %, μg).
3. Preserve age-range patterns (e.g. "5-11 years", "≥12 years").
4. Normalize Arabic punctuation/diacritics (harakat removal, alef normalization, ta-marbuta).
5. Mixed Arabic-English: detect and segment, normalize each part independently.
6. Spelling variants: salbutamol ↔ albuterol, sulphate ↔ sulfate.
7. Strip redundant whitespace.
"""
from __future__ import annotations
import re, unicodedata

_PROTECTED_CLINICAL_TERMS: set[str] = {
    "ICS","SABA","LABA","MART","LTRA","FeNO","FEV1","FEV","PEF","FVC",
    "OCS","AERD","EIB","NIV","CPAP","NICE","WHO","GINA",
}

_DRUG_SYNONYMS: dict[str, str] = {
    "albuterol": "salbutamol",
    "sulphate": "sulfate",
    "sulphuric": "sulfuric",
    "sulphur": "sulfur",
}

_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
_ALEF_VARIANTS_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")
_TA_MARBUTA_RE = re.compile(r"\u0629(?=\s|$)")
_ARABIC_PUNCTUATION_RE = re.compile(r"[،؛؟٪]")
_TATWEEL_RE = re.compile(r"\u0640+")
_UNIT_PATTERN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|μg|mg|ml|L|%|years?|yr|mEq|mmol)", re.IGNORECASE)


def _protect_terms(query: str) -> tuple[str, dict[str, str]]:
    token_map: dict[str, str] = {}
    result = query
    for i, term in enumerate(sorted(_PROTECTED_CLINICAL_TERMS, key=len, reverse=True)):
        pattern = re.compile(r"(?<![a-zA-Z])" + re.escape(term) + r"(?![a-zA-Z])")
        if pattern.search(result):
            found = pattern.search(result).group(0)
            placeholder = f"__PROT{i}__"
            token_map[placeholder] = found
            result = pattern.sub(placeholder, result)
    return result, token_map


def _restore_terms(query: str, token_map: dict[str, str]) -> str:
    for placeholder, original in token_map.items():
        query = query.replace(placeholder, original)
    return query


def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _ALEF_VARIANTS_RE.sub("\u0627", text)
    text = _TA_MARBUTA_RE.sub("\u0647", text)
    text = _TATWEEL_RE.sub("", text)
    text = _ARABIC_PUNCTUATION_RE.sub(" ", text)
    return text


def _is_arabic_char(ch: str) -> bool:
    return "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"


def _has_arabic(text: str) -> bool:
    return any(_is_arabic_char(c) for c in text)


def normalize_medical_query(query: str) -> str:
    if not query or not query.strip():
        return query
    protected, token_map = _protect_terms(query)
    if _has_arabic(protected):
        protected = normalize_arabic(protected)
    words = protected.split()
    normalized_words = []
    for w in words:
        if "__PROT" in w:
            normalized_words.append(w)
            continue
        lower_stripped = w.lower().rstrip(".,;:!?)")
        canonical = _DRUG_SYNONYMS.get(lower_stripped, lower_stripped)
        normalized_words.append(canonical)
    protected = " ".join(normalized_words)
    protected = unicodedata.normalize("NFC", protected)
    protected = re.sub(r"\s{2,}", " ", protected).strip()
    result = _restore_terms(protected, token_map)
    return result


def extract_clinical_units(query: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).lower()) for m in _UNIT_PATTERN_RE.finditer(query)]


def has_arabic_content(query: str) -> bool:
    return _has_arabic(query)
