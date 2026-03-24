# Evaluation System (Grounding-First)

This challenge is designed to reward **verifiable, grounded RAG**. Grounding is the strongest signal in the final score because it is a *multiplier* applied to everything else.

---

## 1. Dataset overview

**Warm-up set:** 100 questions / ~30 docs (for local debugging)  
**Final set:** 900 questions / ~300 docs (final ranking)

Each phase ships its **own document corpus** via `GET /documents`. The two corpora may partially overlap, but you must answer each phase's questions using **only its own corpus** — index them independently.

Question types include single-document extraction, clause analysis, multi-document reasoning, negative/adversarial cases, and uncertainty handling.

---

## 2. Answer types

Each question includes an `answer_type` that defines the scoring rule.

### Deterministic types

- `number`: JSON number (integer or float — both valid, e.g. `1000` or `1000.0`), **±1% tolerance**
- `boolean`: JSON boolean `true` / `false` (exact match)
- `name`: normalized exact match (string); names are designed to have a single unambiguous expected answer — no aliases
- `names`: JSON array of strings, scored with Jaccard similarity over normalized sets; no aliases in expected values
- `date`: ISO 8601 (`YYYY-MM-DD`) exact match

**Special value:** for any deterministic type, the JSON value `null` is a valid answer meaning "this information is absent from the corpus and cannot be found or inferred". If both reference and participant answer are `null` → **1 point**. If only one is `null` → **0 points**.

### Free text type

`free_text` is scored via an LLM judge using multiple criteria (see below).  
Expected format: coherent text of 1–3 paragraphs, **max 280 characters**.

For unanswerable `free_text` questions, return a natural-language statement such as *"There is no information on this question in the provided documents."* and set `retrieved_chunk_pages` to `[]`.

---

## 3. Metric blocks

Each submission computes:

1. **Deterministic score** `S_det`  
2. **Assistant score** `S_asst` (LLM judge)  
3. **Grounding score** `G` (retrieval quality)  
4. **Telemetry factor** `T`  
5. **TTFT factor** `F`

Final score (0–1):

```
Total = (0.7 * S_det + 0.3 * S_asst) * G * T * F
```

The platform reports `total_score` as a 0–1 value.

---

## 4. LLM-as-Judge (free_text)

`free_text` answers are evaluated by an internal LLM-based judge pipeline.

### Criteria (5)

1. **Correctness** — does the response contain the key information from the reference answer, answer the question, and contain no factual errors?
2. **Completeness** — does the response address all aspects of the question and cover the key points present in the reference answer, leaving no important parts unanswered?
3. **Grounding** — is every statement supported by information from the retrieved context? Does the response avoid claims that contradict or add specific material facts not present in the context?
4. **Confidence calibration** — is confidence adequately expressed? Does the system properly convey uncertainty when the answer is not fully certain or information is missing?
5. **Clarity & relevance** — is the answer clear, concise, directly addressing the question, and logically structured?

Each criterion returns a score in **{0, 1}**.  

The **assistant score** for a question is the mean of the 5 criteria (0–1).  
The **submission assistant score** is the mean of all free-text questions.

---

## 5. Grounding (retrieval quality) — **most important**

Grounding is computed from `telemetry.retrieval.retrieved_chunk_pages`.
Each entry is a `(doc_id, page_number)` pair. Gold references in the dataset
use the same `doc_id` and `page_numbers` format.

### Scoring rule (F‑beta, β = 2.5)

Let `P` be predicted page refs and `G` be gold page refs:

```
precision = |P ∩ G| / |P|
recall = |P ∩ G| / |G|
F_beta = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
```

With `beta = 2.5`, **recall is prioritized**, but extra pages still reduce precision.

**Edge cases:**
- Both sets are empty → **1.0**
- One set is empty and the other is not → **0.0**

**For unanswerable questions**: set `retrieved_chunk_pages` to an empty array `[]`. This will match an empty gold set and yield grounding **1.0**.

**Important:** `Grounding` is a multiplier in the final score.  
Even perfect answers collapse if grounding is low.

---

## 6. Telemetry factor (T)

Telemetry is required for every answer. The platform validates:

- `timing` present, non-negative, `ttft_ms <= total_time_ms`
- `usage` present, non-negative token counts
- `retrieval.retrieved_chunk_pages` present and non-empty
- `doc_id` exists in corpus mapping

If any of these fail, the answer is **malformed** and gets:

```
telemetry_factor = 0.9
```

The submission telemetry score `T` is the **mean** of all telemetry factors.

---

## 7. TTFT factor (F)

`TTFT` is the per-answer *time to first token* from telemetry.  
The score uses this mapping:

| TTFT (ms) | Factor |
|---|---|
| `< 1000` | `1.05` |
| `< 2000` | `1.02` |
| `< 3000` | `1.00` |
| `> 3000` | `0.85–0.99` |

The submission `multiplier` is the **mean** of all TTFT factors.

---

## 8. Leaderboard fields

- `deterministic` — deterministic accuracy (0–1)
- `assistant` — LLM-judge score (0–1)
- `grounding` — average grounding (0–1)
- `telemetry` — telemetry availability (0.90–1.00)
- `ttft_ms` — mean TTFT in milliseconds
- `ttft_multiplier` — TTFT factor applied to total score
- `total_score` — final composite score (0–1)

---

## 9. Optimization priorities

1. **Grounding first** — improve retrieval precision/recall with tight context
2. **Correct answers** — deterministic accuracy is the base signal
3. **LLM quality** — free-text judged on 5 criteria
4. **Telemetry health** — avoid malformed telemetry
5. **Speed** — TTFT boosts total score only if everything above is solid
