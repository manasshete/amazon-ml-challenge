"""Country-agnostic text normalization for names and addresses.

Every function here must work without knowing the record's country, because
France appears only in the test set (Section 5.4 of the plan: nothing may
branch on country == "US" or "India"). Abbreviation/legal-form dictionaries
below simply include entries from all three regions; a French record just
won't match the India-only entries, which is harmless.

Views produced (Section 5.2 / 5.3 of the plan):
  name_clean   -- cleaned full name
  name_core    -- name with legal suffix removed
  legal_form   -- the canonical legal suffix found (or "" if none)
  name_tokens_sorted -- tokens sorted alphabetically (handles word-order swaps)
  acronym      -- first letters of name_core's tokens
  addr_clean   -- cleaned address, abbreviations expanded
  postal       -- extracted postal code, or ""
  numbers      -- sorted list of numeric tokens found in the address
  landmark     -- text following near/opp/behind/... or ""
  addr_no_landmark -- address with the landmark phrase removed
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Generic text cleaning
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_KEEP_RE = re.compile(r"[^a-z0-9/\- ]")  # keep letters, digits, / and -


def strip_accents(text: str) -> str:
    """Fold accented characters to their base form: 'é' -> 'e', 'ç' -> 'c'.

    NFKD ("compatibility decomposition") splits an accented character into
    a base letter + a separate combining-mark codepoint (e.g. 'é' becomes
    'e' + U+0301). We then drop every combining-mark codepoint, leaving
    plain ASCII-ish text. This is the single most important step for
    France, whose business names/addresses are full of accents that never
    appear in the US/India training data.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """Lowercase, strip accents, normalize '&' , drop stray punctuation."""
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = _PUNCT_KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Name normalization
# ---------------------------------------------------------------------------

# Legal-form suffixes across US / India / France. Order matters: longer/more
# specific tokens are checked as whole words via regex word boundaries, so
# order doesn't actually cause false matches here, but grouping keeps the
# dictionary auditable per the "disclose hand-built dictionaries" rule.
LEGAL_FORMS = [
    "incorporated", "inc", "llc", "ltd", "limited", "corporation", "corp",
    "co", "company", "pvt", "private", "plc", "llp", "lp", "pllc",
    "gmbh", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ste",
    "societe", "ets", "etablissements",
]
_LEGAL_FORM_RE = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_FORMS, key=len, reverse=True)) + r")\b"
)

_AND_SYNONYMS = {"et"}  # French "et" == "and", used only for token comparison


def split_legal_form(name_clean: str) -> tuple[str, str]:
    """Return (name_core, legal_form). legal_form is "" if none found.

    Legal suffixes are often compound ("Pvt Ltd", "Private Limited"), so we
    remove *every* legal-form token found, not just the first -- stripping
    only one and leaving the other in name_core would contaminate every
    downstream name comparison. legal_form joins all found tokens in the
    order they appeared, e.g. "pvt ltd".
    """
    matches = _LEGAL_FORM_RE.findall(name_clean)
    if not matches:
        return name_clean, ""
    legal_form = " ".join(matches)
    core = _LEGAL_FORM_RE.sub(" ", name_clean)
    core = _WS_RE.sub(" ", core).strip()
    return core, legal_form


def tokens_sorted(text: str) -> str:
    """Sort a cleaned string's tokens alphabetically.

    Handles word-order transpositions: "Store Lakshmi" and "Lakshmi Store"
    both become "lakshmi store", so an exact-string or Jaccard comparison
    on this view is order-invariant.
    """
    return " ".join(sorted(text.split()))


def acronym(name_core: str) -> str:
    """First letters of each token: 'international business machines' -> 'ibm'.

    Only useful when the acronym has >= 2 letters (a single-token name has
    a meaningless one-letter "acronym").
    """
    tokens = name_core.split()
    if len(tokens) < 2:
        return ""
    return "".join(t[0] for t in tokens if t)


def normalize_name(raw_name: str) -> dict:
    """Produce every name view for one raw business_name string."""
    name_clean = clean_text(raw_name)
    name_core, legal_form = split_legal_form(name_clean)
    return {
        "name_clean": name_clean,
        "name_core": name_core,
        "legal_form": legal_form,
        "name_tokens_sorted": tokens_sorted(name_core),
        "acronym": acronym(name_core),
    }


# ---------------------------------------------------------------------------
# 3. Address normalization
# ---------------------------------------------------------------------------

# word -> expansion. Applied as whole-word regex substitutions so "st" only
# expands when it's a standalone token, not inside another word.
ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "apt": "apartment", "ste": "suite", "fl": "floor", "opp": "opposite",
    "nr": "near", "bldg": "building", "mkt": "market",
    # French address abbreviations
    "r": "rue", "pl": "place", "fbg": "faubourg", "chem": "chemin",
    "imp": "impasse", "all": "allee",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(ADDRESS_ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)

# Landmark introducer phrases; everything after one is split out as a
# separate "landmark" view so it doesn't drown out the real street address
# in a similarity comparison.
_LANDMARK_RE = re.compile(
    r"\b(near|opp|opposite|behind|beside|next to|pres de|en face)\b(.*)$"
)

# A standalone 5- or 6-digit number near the end of the string: covers
# India's 6-digit PIN, US 5-digit ZIP, and France's 5-digit postal code,
# without branching on which country it is.
_POSTAL_RE = re.compile(r"\b(\d{5,6})\b")

_NUMBER_RE = re.compile(r"\d+")


def expand_abbreviations(text: str) -> str:
    def repl(m):
        return ADDRESS_ABBREVIATIONS[m.group(1)]
    return _ABBREV_RE.sub(repl, text)


def extract_landmark(addr_clean: str) -> tuple[str, str]:
    """Split off a landmark phrase. Returns (addr_no_landmark, landmark)."""
    match = _LANDMARK_RE.search(addr_clean)
    if not match:
        return addr_clean, ""
    landmark = match.group(0).strip()
    addr_no_landmark = addr_clean[: match.start()].strip()
    return addr_no_landmark, landmark


def extract_postal(addr_clean: str) -> str:
    """Return the last standalone 5-6 digit token, treated as a postal code."""
    matches = _POSTAL_RE.findall(addr_clean)
    return matches[-1] if matches else ""


def extract_numbers(addr_clean: str) -> list:
    """All numeric tokens in the address (house/plot/shop numbers, etc.), sorted."""
    return sorted(set(_NUMBER_RE.findall(addr_clean)))


def normalize_address(raw_address: str) -> dict:
    """Produce every address view for one raw business_address string."""
    addr_clean = clean_text(raw_address)
    addr_clean = expand_abbreviations(addr_clean)
    addr_no_landmark, landmark = extract_landmark(addr_clean)
    return {
        "addr_clean": addr_clean,
        "addr_no_landmark": addr_no_landmark,
        "landmark": landmark,
        "postal": extract_postal(addr_clean),
        "numbers": extract_numbers(addr_clean),
    }


def normalize_record(business_name: str, business_address: str) -> dict:
    """Convenience wrapper: all name views + all address views for one record."""
    out = {}
    out.update(normalize_name(business_name))
    out.update(normalize_address(business_address))
    return out


if __name__ == "__main__":
    samples = [
        ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC"),
        ("International South Consultants Private Ltd", ""),
        ("Café Lumière SARL", "12 Rue de la Paix, Near Opera, 75002 Paris"),
        ("S. Lakshmi & Sons Pvt Ltd", "Rd No 5, Opp SBI ATM, Hyderabad, 500001"),
    ]
    for name, addr in samples:
        print(f"\nRAW name={name!r} addr={addr!r}")
        for k, v in normalize_record(name, addr).items():
            print(f"  {k:20s} = {v!r}")
