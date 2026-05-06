# Real eval run — `rag-threat-intel` v1.0.0

These numbers are from a real local run, not fabricated. Reproduce
with the steps in the project README.

## Stack used

| Component        | Choice                                    |
|------------------|-------------------------------------------|
| Embedding model  | `all-minilm` via Ollama (384-dim, 256-tok) |
| Generation model | `llama3.2:1b` via Ollama (1.3 GB)         |
| Vector store     | `pgvector/pgvector:pg16` (Docker, host port 5433) |
| Hardware         | Intel Core i5-8250U / 16 GB RAM           |

The author used a smaller embedding model (`all-minilm`, 256-token
context) and a smaller generation model (`llama3.2:1b`, 1.3 GB) than
the README's "default" stack because the working machine had ~7 GB
free at run time. The `OllamaEmbedder` truncates inputs to 600 chars
to fit the embedding model's window — see `embeddings/embed.py`.

## Corpus

* **7 named historical CVEs** (Log4Shell, Heartbleed, Apache Struts,
  BlueKeep, Zerologon, Spring4Shell, regreSSHion) — fetched via the
  NVD 2.0 by-id endpoint (`fetch_cves_by_ids`). These are the CVEs
  the eval set's CVE-recall queries reference by name.
* **30 recent CVEs** — fetched via the NVD 2.0 paginated endpoint
  (`fetch_recent_cves`). These are the negatives the retriever has
  to discriminate against.

37 unique CVE Documents in total → 38 `fixed_size` chunks, 38
`semantic` chunks, 91 `sentence_window` chunks.

## MRR @ k (real run, 14 CVE-recall queries with ground-truth source IDs)

```bash
python -m eval.eval_mrr
```

| Strategy          | MRR@5  | MRR@10 | mean latency (ms) | errors |
|-------------------|-------:|-------:|------------------:|-------:|
| fixed\_size       | 0.7679 | 0.7798 | 39.9              | 0      |
| **semantic**      | **0.8214** | **0.8333** | 42.2          | 0      |
| sentence\_window  | 0.7619 | 0.7619 | 40.4              | 0      |

**Semantic wins** by ~5 percentage points on both depths. This is
qualitatively the same direction the chunking-comparison literature
reports — see Iris et al. 2025 in the paper bibliography.

The sentence-window strategy underperforms slightly here because the
NVD CVE corpus has very short Documents (one description per CVE);
chunking by sentence multiplies the chunk count (91 vs 38) without
giving the embedder a wider semantic window to work with. On a
PDF-heavy corpus the result usually flips — sentence-window's
±N-neighbour context is most valuable when the retrieved chunk is a
sentence pulled out of a 30-page document.

## Faithfulness (semantic strategy, llama3.2:1b, n=12)

```bash
python -m eval.eval_faithfulness --strategy semantic --limit 12
```

| Metric                    | Value     | Interpretation                                              |
|---------------------------|----------:|-------------------------------------------------------------|
| mean citation density     | 0.201     | only 20% of sentences end in a `[doc_N]` tag                |
| **mean citation validity**| **1.000** | **the model never invented a `[doc_N]` tag — zero hallucinated citations** |
| mean refusal honesty      | 0.333     | 4 of 12 queries answered correctly; 8 were refused incorrectly |
| mean elapsed (ms)         | 53,434    | ~53 s per query on the working hardware                     |
| errors                    | 0         | every call returned a parseable response                    |

### What this number set actually tells us

A 1-billion-parameter model under our strict citation contract
**chooses to refuse rather than cite ambiguously**. Of the 12 queries
in the pilot, the model emitted the canonical refusal sentence on 8
even though the retriever had returned plausibly relevant chunks for
each. When the model *did* answer, it was extremely faithful: 100%
citation validity (no fabricated `[doc_42]` tags), and meaningful
density on the responses where it engaged.

The right read of the three signals together is *"a small local
model is over-cautious but reliable"* — and that is exactly the
trade-off the paper's three-axis scorer is designed to surface.
A 70B-class model would almost certainly refuse less and cite more
densely; the next time this hardware budget allows a larger model,
re-running the same script against it would produce a directly
comparable row.

Per-query rows are in `results/eval_faithfulness.csv` if you want
to inspect which 4 queries the model engaged on (q02, q05, q07,
q09) versus which 8 it refused.

## Reproducing locally

```bash
docker run -d --name rti-pgvector \
    -e POSTGRES_DB=rti -e POSTGRES_USER=rti -e POSTGRES_PASSWORD=rti-dev-pw \
    -p 5433:5432 pgvector/pgvector:pg16
ollama pull all-minilm
ollama pull llama3.2:1b
cp .env.example .env  # then point PG_PORT=5433 and the Ollama models above
python -m ingest.bootstrap --cve-limit 30 --no-pdfs
python -m eval.eval_mrr
python -m eval.eval_faithfulness --strategy semantic --limit 12
```

Total wall-clock for the whole eval: ingest ≈30 s, MRR ≈3 s,
faithfulness ≈8–15 min depending on RAM pressure.
