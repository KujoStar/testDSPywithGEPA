"""
haiku_metric.py
---------------
A spaCy-based haiku evaluator packaged as a DSPy GEPA metric.

Runs 20 sub-checks (structural, lexical, syntactic, semantic) on a
candidate haiku, aggregates them into a weighted score in [0, 1], and
returns rich textual feedback that GEPA's reflection LM can introspect
to mutate the upstream prompt.

Dependencies:
    pip install spacy dspy pyphen
    python -m spacy download en_core_web_md

Usage with dspy.GEPA:

    import dspy
    from haiku_metric import haiku_metric

    gepa = dspy.GEPA(
        metric=haiku_metric,
        reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
        auto="light",
    )
    optimized = gepa.compile(student, trainset=trainset, valset=valset)

Notes
-----
* This is sketch-quality code. The lexicons (kigo, kireji) are starter
  sets; replace with a curated saijiki for serious work.
* `_extract_haiku_text` assumes the prediction has a `haiku` field;
  change to match your dspy.Signature.
* Weights in CHECKS are a starting point — tune them on a small
  human-labeled dev set before trusting the aggregate.
"""

from __future__ import annotations


import time; t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] importing spacy..."); import spacy
print(f"[{time.time()-t0:5.1f}s] importing dspy...");  import dspy
print(f"[{time.time()-t0:5.1f}s] loading model...");   nlp = spacy.load("en_core_web_md")
print(f"[{time.time()-t0:5.1f}s] ready.")

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import spacy
from spacy.tokens import Doc, Token

try:
    import pyphen
    _HYPH = pyphen.Pyphen(lang="en_US")
except ImportError:  # pragma: no cover
    _HYPH = None

import dspy


# ---------------------------------------------------------------------------
# spaCy pipeline (loaded once at import time)
# ---------------------------------------------------------------------------

# md or lg is required for token.similarity / doc.similarity.
_MODEL = "en_core_web_md"
_NLP = spacy.load(_MODEL)


# ---------------------------------------------------------------------------
# Lexicons & anchors
# ---------------------------------------------------------------------------

# Tiny starter kigo set. Extend with a real saijiki for production.
KIGO_LEMMAS: set[str] = {
    # spring
    "blossom", "cherry", "plum", "thaw", "swallow", "warbler", "sapling",
    "bud", "mist", "robin", "daffodil",
    # summer
    "cicada", "lotus", "firefly", "monsoon", "thunder", "humid", "swelter",
    "dragonfly", "watermelon",
    # autumn
    "harvest", "chrysanthemum", "maple", "persimmon", "geese", "stubble",
    "acorn", "scarecrow", "moonlight",
    # winter
    "snow", "frost", "ice", "icicle", "owl", "wolf", "ash", "shiver",
    "bare", "hearth",
}

SENSORY_ANCHORS: list[str] = ["see", "hear", "smell", "taste", "touch"]

KIREJI_PUNCT: set[str] = {"—", "–", ":", ";", "…", "--"}

FIRST_PERSON_LEMMAS: set[str] = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def _syllables(word: str) -> int:
    """Count syllables in one word: pyphen if available, vowel-cluster fallback."""
    m = _WORD_RE.findall(word.lower())
    if not m:
        return 0
    w = m[0]
    if _HYPH is not None:
        parts = [p for p in _HYPH.inserted(w).split("-") if p]
        return max(1, len(parts))
    # fallback heuristic
    vowels = "aeiouy"
    count, prev_v = 0, False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _line_syllables(line: str) -> int:
    return sum(_syllables(tok) for tok in line.split())


def _max_dep_depth(doc: Doc) -> int:
    def depth(tok: Token) -> int:
        d, cur = 0, tok
        while cur.head.i != cur.i:
            d += 1
            cur = cur.head
        return d
    return max((depth(t) for t in doc), default=0)


# ---------------------------------------------------------------------------
# The 20 checks. Each returns (score in [0, 1], one-line feedback string).
# Signature: (doc, lines, line_docs) -> (score, feedback)
# ---------------------------------------------------------------------------

def _c01_syllables(doc, lines, line_docs):
    target = [5, 7, 5]
    counts = [_line_syllables(ln) for ln in lines]
    counts = (counts + [0, 0, 0])[:3]
    diffs = [abs(c - t) for c, t in zip(counts, target)]
    score = max(0.0, 1.0 - sum(diffs) / 9)
    return score, f"Syllable counts {counts} vs target [5,7,5] (diffs {diffs})."


def _c02_line_count(doc, lines, line_docs):
    n = len(lines)
    if n == 3:
        return 1.0, "Line count is 3 (correct)."
    return max(0.0, 1.0 - abs(n - 3) / 3), f"Line count is {n}; expected 3."


def _c03_pos_distribution(doc, lines, line_docs):
    content = [t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}]
    if not content:
        return 0.0, "No content tokens detected."
    nv = sum(1 for t in content if t.pos_ in {"NOUN", "VERB", "PROPN"})
    ratio = nv / len(content)
    return min(1.0, ratio / 0.7), (
        f"Noun+verb share of content tokens: {ratio:.2f} (target >=0.70)."
    )


def _c04_adjective_scarcity(doc, lines, line_docs):
    adj = [t for t in doc if t.pos_ == "ADJ"]
    n = len(adj)
    score = max(0.0, 1.0 - max(0, n - 1) / 3)
    return score, f"Adjective count: {n} ({[t.text for t in adj]}); haiku favor sparse adjectives."


def _c05_present_tense(doc, lines, line_docs):
    verbs = [t for t in doc if t.pos_ == "VERB"]
    if not verbs:
        return 1.0, "No finite verbs (vacuously present, or verbless — both acceptable)."
    pres = sum(1 for t in verbs if "Pres" in t.morph.get("Tense"))
    return pres / len(verbs), f"{pres}/{len(verbs)} verbs are present-tense."


def _c06_noun_chunks(doc, lines, line_docs):
    chunks = list(doc.noun_chunks)
    return min(1.0, len(chunks) / 2), (
        f"{len(chunks)} noun chunks: {[c.text for c in chunks]}."
    )


def _c07_kigo(doc, lines, line_docs):
    hits = [t.text for t in doc if t.lemma_.lower() in KIGO_LEMMAS]
    if hits:
        return 1.0, f"Seasonal reference (kigo) found: {hits}."
    return 0.0, "No seasonal reference (kigo) detected in starter lexicon."


def _c08_kireji(doc, lines, line_docs):
    found = []
    for i, ld in enumerate(line_docs[:2]):
        if len(ld) and ld[-1].is_punct and ld[-1].text in KIREJI_PUNCT:
            found.append((i + 1, ld[-1].text))
    if found:
        return 1.0, f"Cutting word (kireji) punctuation present: {found}."
    return 0.3, "No em-dash/colon/ellipsis at end of line 1 or 2 — juxtaposition cue weak."


def _c09_stop_ratio(doc, lines, line_docs):
    total = sum(1 for t in doc if not t.is_punct and not t.is_space)
    if not total:
        return 0.0, "Empty doc."
    stops = sum(1 for t in doc if t.is_stop)
    ratio = stops / total
    score = max(0.0, 1.0 - max(0.0, ratio - 0.4) / 0.4)
    return score, f"Stop-word ratio: {ratio:.2f} (target <=0.40)."


def _c10_lexical_density(doc, lines, line_docs):
    total = sum(1 for t in doc if not t.is_punct and not t.is_space)
    if not total:
        return 0.0, "Empty doc."
    content = sum(1 for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"})
    density = content / total
    return min(1.0, density / 0.6), f"Lexical density: {density:.2f} (target >=0.60)."


def _c11_avg_token_length(doc, lines, line_docs):
    toks = [t for t in doc if t.is_alpha]
    if not toks:
        return 0.0, "No alphabetic tokens."
    avg = sum(len(t.text) for t in toks) / len(toks)
    score = max(0.0, 1.0 - max(0.0, avg - 5.5) / 3.0)
    return score, f"Average alpha-token length: {avg:.2f} chars."


def _c12_ner_rarity(doc, lines, line_docs):
    ents = list(doc.ents)
    score = 1.0 if not ents else max(0.0, 1.0 - len(ents) / 3)
    return score, f"Named entities: {[(e.text, e.label_) for e in ents]}."


def _c13_first_person_absence(doc, lines, line_docs):
    fp = [
        t.text for t in doc
        if t.pos_ == "PRON" and t.lemma_.lower() in FIRST_PERSON_LEMMAS
    ]
    if not fp:
        return 1.0, "No first-person pronouns (classical preference)."
    return max(0.0, 1.0 - len(fp) / 3), f"First-person pronouns present: {fp}."


def _c14_line_juxtaposition(doc, lines, line_docs):
    if len(line_docs) < 3 or not all(ld.has_vector for ld in line_docs):
        return 0.5, "Could not compute line vectors (need en_core_web_md/lg and 3 lines)."
    sim = line_docs[0].similarity(line_docs[2])
    # 1.0 at sim<=0.3, 0.0 at sim>=0.9
    score = max(0.0, min(1.0, (0.9 - sim) / 0.6))
    return score, f"Line1<->Line3 similarity: {sim:.2f} (lower = stronger juxtaposition)."


def _c15_dep_depth(doc, lines, line_docs):
    depth = _max_dep_depth(doc)
    score = max(0.0, 1.0 - max(0, depth - 4) / 4)
    return score, f"Max dependency depth: {depth} (target <=4)."


def _c16_sentence_count(doc, lines, line_docs):
    n = len(list(doc.sents))
    if n in (1, 2):
        return 1.0, f"Sentence count: {n} (ideal)."
    return max(0.0, 1.0 - abs(n - 1.5) / 3), f"Sentence count: {n} (ideal 1-2)."


def _c17_lemma_repetition(doc, lines, line_docs):
    content_lemmas = [
        t.lemma_.lower() for t in doc
        if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
    ]
    if not content_lemmas:
        return 1.0, "No content lemmas to check."
    dupes = len(content_lemmas) - len(set(content_lemmas))
    return max(0.0, 1.0 - dupes / 3), f"Content-lemma duplicates: {dupes}."


def _c18_article_frequency(doc, lines, line_docs):
    arts = sum(1 for t in doc if t.lemma_.lower() in {"a", "an", "the"})
    total = sum(1 for t in doc if t.is_alpha)
    if not total:
        return 0.0, "Empty doc."
    ratio = arts / total
    score = max(0.0, 1.0 - max(0.0, ratio - 0.15) / 0.25)
    return score, f"Article ratio: {ratio:.2f} ({arts}/{total}); target <=0.15."


def _c19_sensory_similarity(doc, lines, line_docs):
    if not doc.has_vector:
        return 0.5, "No vectors available for sensory check."
    anchors = [_NLP.vocab[w] for w in SENSORY_ANCHORS if _NLP.vocab[w].has_vector]
    content = [t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ"} and t.has_vector]
    if not content or not anchors:
        return 0.5, "Could not compute sensory similarity."
    best = max(max(t.similarity(a) for a in anchors) for t in content)
    return min(1.0, best / 0.5), f"Max sensory-anchor similarity: {best:.2f}."


def _c20_corpus_similarity(doc, lines, line_docs, reference: Optional[Doc] = None):
    if reference is None or not reference.has_vector or not doc.has_vector:
        return 0.5, "No reference corpus provided — neutral score."
    sim = doc.similarity(reference)
    score = min(1.0, max(0.0, (sim - 0.3) / 0.5))
    return score, f"Similarity to reference haiku corpus: {sim:.2f}."


# ---------------------------------------------------------------------------
# Registry (name, weight, fn). Tune weights to taste.
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, float, Callable]] = [
    ("syllables_5_7_5",      3.0, _c01_syllables),
    ("line_count_3",         2.0, _c02_line_count),
    ("pos_distribution",     1.0, _c03_pos_distribution),
    ("adjective_scarcity",   1.0, _c04_adjective_scarcity),
    ("present_tense",        1.0, _c05_present_tense),
    ("noun_chunks_imagery",  1.5, _c06_noun_chunks),
    ("kigo_seasonal",        1.5, _c07_kigo),
    ("kireji_cutting_word",  1.0, _c08_kireji),
    ("stop_word_ratio",      0.5, _c09_stop_ratio),
    ("lexical_density",      1.0, _c10_lexical_density),
    ("avg_token_length",     0.5, _c11_avg_token_length),
    ("ner_rarity",           0.5, _c12_ner_rarity),
    ("first_person_absence", 0.5, _c13_first_person_absence),
    ("line_juxtaposition",   1.5, _c14_line_juxtaposition),
    ("dep_tree_depth",       0.5, _c15_dep_depth),
    ("sentence_count",       0.5, _c16_sentence_count),
    ("lemma_repetition",     0.5, _c17_lemma_repetition),
    ("article_frequency",    0.5, _c18_article_frequency),
    ("sensory_similarity",   1.0, _c19_sensory_similarity),
    ("corpus_similarity",    0.5, _c20_corpus_similarity),
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    score: float
    per_check: list[tuple[str, float, str]] = field(default_factory=list)

    def feedback_text(self) -> str:
        lines = [
            f"  - {name:24s} score={s:.2f}  {fb}"
            for name, s, fb in self.per_check
        ]
        weakest = sorted(self.per_check, key=lambda x: x[1])[:5]
        weak = "\n".join(f"  - {n}: {fb}" for n, _, fb in weakest)
        return (
            f"Aggregate score: {self.score:.3f}\n\n"
            f"Per-check breakdown:\n" + "\n".join(lines)
            + f"\n\nWeakest 5 checks:\n{weak}"
        )


def evaluate_haiku(text: str, reference: Optional[Doc] = None) -> EvalResult:
    """Run all 20 checks on a haiku string and return an EvalResult."""
    lines = _split_lines(text)
    doc = _NLP(text)
    line_docs = [_NLP(ln) for ln in lines]

    total_w, weighted_sum = 0.0, 0.0
    per_check: list[tuple[str, float, str]] = []

    for name, weight, fn in CHECKS:
        if name == "corpus_similarity":
            s, fb = fn(doc, lines, line_docs, reference=reference)
        else:
            s, fb = fn(doc, lines, line_docs)
        s = max(0.0, min(1.0, float(s)))
        weighted_sum += weight * s
        total_w += weight
        per_check.append((name, s, fb))

    return EvalResult(
        score=weighted_sum / total_w if total_w else 0.0,
        per_check=per_check,
    )


# ---------------------------------------------------------------------------
# DSPy GEPA metric
# ---------------------------------------------------------------------------

def _extract_haiku_text(pred, field_name: str = "haiku") -> str:
    """Pull the haiku string out of a dspy.Prediction. Adjust for your signature."""
    if hasattr(pred, field_name):
        val = getattr(pred, field_name)
        if isinstance(val, str):
            return val
    # fall back to first string attribute
    for v in (getattr(pred, "__dict__", {}) or {}).values():
        if isinstance(v, str):
            return v
    return str(pred)


def haiku_metric(
    gold,
    pred,
    trace=None,
    pred_name: Optional[str] = None,
    pred_trace=None,
):
    """
    GEPA-compatible feedback metric for haiku generation.

    Returns dspy.Prediction(score=float, feedback=str). Score is the weighted
    average of 20 spaCy-based sub-checks; feedback is a structured breakdown
    that GEPA's reflection LM uses to mutate the upstream prompt.
    """
    text = _extract_haiku_text(pred)
    if not text or not text.strip():
        return dspy.Prediction(
            score=0.0,
            feedback="Empty output — no haiku text to evaluate.",
        )

    result = evaluate_haiku(text)
    feedback = (
        f"Candidate haiku:\n{text}\n\n"
        + result.feedback_text()
        + "\n\nGuidance: to raise the score, prioritize fixing the weakest "
          "checks above without regressing the strongest ones. The 5-7-5 "
          "syllable constraint, three-line structure, and a concrete "
          "seasonal image are weighted most heavily."
    )
    return dspy.Prediction(score=result.score, feedback=feedback)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "an old silent pond—\n"
        "a frog jumps into the pond\n"
        "splash, silence again"
    )
    res = evaluate_haiku(sample)
    print(f"Score: {res.score:.3f}\n")
    for name, s, fb in res.per_check:
        print(f"{name:24s} {s:.2f}  {fb}")

    