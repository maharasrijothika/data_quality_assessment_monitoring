"""Central application configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Upload limits
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Semantic understanding
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
# Candidate depth. Raised from 5 to 8 when the KB grew to families + expanded
# specific concepts (Part C of the Stage 04 upgrade): the union of embedding
# top-K and lexical top-K candidates needs more room so that a lexically
# exact concept can never be crowded out of the candidate list.
SEMANTIC_TOP_K = 8

# Version tag for the TEXT REPRESENTATION embedded for KB concepts. Bump when
# build_concept_text changes (e.g. raw JSON no longer embedded). Stored KB
# embeddings are keyed by "<model>/<repr-version>" so stale embeddings are
# regenerated exactly once and never per request.
SEMANTIC_EMBEDDING_REPRESENTATION_VERSION = 2

# Semantic confidence thresholds (score is a weighted evidence score, not a
# calibrated probability). DERIVED from the evaluation harness
# (backend/tools/semantic_eval.py) threshold sweep on the dev split of
# Calibrated on backend/data/semantic_eval/semantic_gold_labels.csv
# (359 dev rows, expanded KB run, tools/semantic_eval.py threshold
# sweep; final numbers in data/semantic_eval/eval_expanded/report.md).
# Acceptance bar: wrong-confident <= 1%. Dev-split sweep over the
# 0.55-0.85 grid measures W = 14/6/5/3/2/1/1 for probable =
# 0.55/0.60/0.65/0.70/0.75/0.80/0.85, i.e. 4.8%/2.4%/2.5%/
# 1.6%/1.2%/0.6%/0.7% wrong-confident. The bar is first met at
# probable=0.80 (W=1/155 = 0.6%, strict coverage 71.0%, the
# coverage-maximal feasible point; 0.85 has the same W at lower
# coverage). Holdout stays W=0/19 at 0.80. STRONG sits 5pp above
# PROBABLE so the Probable band stays reachable (confidence sweep:
# strong=0.85 keeps the same 1/2 wrong-labelled-Strong count as
# 0.80 while still labelling ~84% of correct matches Strong at
# margin 0.05). ambiguous/family=0.40 (grid showed 0.35-0.55
# changes W by <1pp; 0.40 keeps the most family proposals). These
# are NOT calibrated probabilities; re-run the harness after
# KB/weight edits.
SEMANTIC_STRONG_THRESHOLD = 0.85
SEMANTIC_PROBABLE_THRESHOLD = 0.80
SEMANTIC_AMBIGUOUS_THRESHOLD = 0.40

# Evidence-coverage threshold: a candidate must actually clear a substantial
# fraction of the concept's applicable checks to reach "Probable".
SEMANTIC_PROBABLE_EVIDENCE_THRESHOLD = 0.50

# Margin (top-1 score minus top-2 score) required for a confident level.
# A candidate whose runner-up is within the margin is downgraded to
# "Ambiguous" regardless of its absolute score. The harness confidence sweep
# (dev split) shows margin 0.05-0.15 changes the wrong-labelled-Strong count
# by <1 while 0.80-strong keeps ~91% of correct matches Strong; 0.12 is kept
# as the conservative end of that flat region.
SEMANTIC_STRONG_MARGIN = 0.12
SEMANTIC_PROBABLE_MARGIN = 0.05

# Deterministic semantic scoring weights. The score is a weighted mean over
# the APPLICABLE evidence features only (see semantic_features.py), so the
# weights below are relative, not absolute: missing evidence (no description,
# no dataset domain) is excluded from both numerator and denominator instead
# of silently capping the achievable score. Weights were tuned ONLY through
# the evaluation harness ablation on the dev split (leave-one-feature-out and
# weight presets); name evidence was raised and description lowered relative
# to the pre-harness draft after the sweep showed name evidence contributes
# most precision at this label volume. value/context/relationship are small
# by design: they disambiguate, they do not carry the decision.
SEMANTIC_SCORING_WEIGHTS = {
    "name_similarity": 0.28,
    "embedding_similarity": 0.22,
    "description_similarity": 0.15,
    "profile_compatibility": 0.12,
    "datatype_compatibility": 0.08,
    "value_evidence": 0.07,
    "context_similarity": 0.05,
    "relationship_evidence": 0.03,
}

# A candidate must also clear this share of its APPLICABLE evidence to be
# more than "Ambiguous". Structurally absent evidence (no description given,
# no dataset domain configured) does not count as failed evidence.
SEMANTIC_APPLICABLE_FEATURES = (
    "name_similarity",
    "embedding_similarity",
    "description_similarity",
    "profile_compatibility",
    "datatype_compatibility",
    "value_evidence",
    "context_similarity",
    "relationship_evidence",
)

# Minimum non-null sample for profile "unique" evidence. On very small
# samples 100% distinctness is not meaningful evidence (5 rows of 5 distinct
# values would otherwise make any column look like an identifier).
SEMANTIC_MIN_UNIQUE_SAMPLE = 10

# Minimum coverage share of non-null rows for a pattern expectation
# (email-like, phone-like, postal-like, currency-like) to PASS. Replaces the
# old "count > 0" rule where one matching value made a 1000-row column
# "email-like". Derived from the harness: 0.6 separates the labelled
# email/phone/postal columns (>95% coverage) from incidental matches.
SEMANTIC_PATTERN_MIN_COVERAGE = 0.6

# KB retrieval gate: candidates below this embedding similarity are not
# considered suitable KB matches. If every candidate is below it, the column
# goes to Open Discovery instead of being forced into the closest concept.
SEMANTIC_KB_MIN_EMBEDDING_SIMILARITY = 0.30

# Top-K alternatives retained for the review UI.
SEMANTIC_ALTERNATIVE_LIMIT = 5

# Integer version of the evidence/decision payload stored in
# SemanticPrediction.evidence_json / alternatives_json / features_json.
# Bumped when the stored shape changes; readers use .get() with defaults so
# payloads from older versions keep loading.
SEMANTIC_EVIDENCE_VERSION = 2

# Minimum approved semantic/rule/remediation feedback examples required before
# offline model training is attempted.
MIN_FEEDBACK_FOR_TRAINING = 20

# ---------------------------------------------------------------------------
# Relationship discovery (deterministic PK/FK candidate evidence)
# ---------------------------------------------------------------------------

# Deterministic relationship scoring weights (weighted mean of applicable
# evidence, same convention as SEMANTIC_SCORING_WEIGHTS). Documented here so
# the ranking is reproducible and configurable.
RELATIONSHIP_SCORING_WEIGHTS = {
    "name_similarity": 0.10,
    "semantic_compatibility": 0.20,
    "datatype_compatibility": 0.05,
    "parent_uniqueness": 0.15,
    "value_containment": 0.20,
    "value_overlap": 0.10,
    "cardinality_compatibility": 0.10,
    "child_completeness": 0.05,
    "structural_evidence": 0.05,
}

# Semantic compatibility is downgraded to a NEUTRAL value (neither helping
# nor hurting) when no semantic mapping exists for a side. A value below
# neutral treats conflicting approved concepts as evidence AGAINST a pair.
RELATIONSHIP_SEMANTIC_NEUTRAL = 0.5

# Composite PK candidates: uniqueness threshold and per-member distinctness
# floor for profiling-driven key-candidate enumeration (pairs preferred;
# triples only when strongly evidenced).
RELATIONSHIP_COMPOSITE_UNIQUENESS = 98.0
RELATIONSHIP_KEY_MIN_MEMBER_UNIQUENESS = 5.0

# Candidates below this evidence score are not surfaced (except manual adds).
RELATIONSHIP_MIN_CANDIDATE_SCORE = 0.35

# Minimum lexical similarity for a (child, parent) column pair to be evaluated
# at all, unless the parent column is a strong PK candidate.
RELATIONSHIP_MIN_NAME_SIMILARITY = 0.40

# Affirmative-evidence floor: a pair with ZERO containment AND zero overlap
# (no value support at all) must at least show this name similarity to remain
# a candidate. Pairs failing it are coincidence (identifier-vs-identifier
# lookalikes), not orphan situations. Low-but-nonzero containment and
# zero-containment-with-matching-name both still survive: a real relationship
# with orphans keeps matching names.
RELATIONSHIP_MIN_NAME_SUPPORT = 0.60

# Safety cap on evaluated (parent, child) pairs per discovery run.
RELATIONSHIP_MAX_PAIRS = 300

# Top ranked candidates retained per child column.
RELATIONSHIP_TOP_PER_CHILD = 3

# Held-out fraction used during offline model evaluation.
HOLDOUT_FRACTION = 0.2

# Relationship discovery thresholds (deterministic evidence, not proof).
RELATIONSHIP_CANDIDATE_MIN_SCORE = 0.35
RELATIONSHIP_MIN_PARENT_UNIQUENESS = 95.0
RELATIONSHIP_HIGH_CONTAINMENT = 90.0
