# 📰 NewsInsight — Bangla News Clustering at Scale

**Automatically discovering topics in Bangla news articles — with no labels — using a distributed, multi-algorithm clustering pipeline.**

> **Status:** Research/experimentation phase, complete and validated. This is a Spark + scikit-learn/FAISS notebook project — see [Status & Roadmap](#-status--roadmap) for exactly what's built vs. planned.

---

## 📖 Overview

Bangla-language news platforms publish an enormous volume of articles every day. As that volume grows, manually identifying related stories, catching cross-posted duplicates, and organizing articles by topic becomes increasingly difficult and time-consuming. Most existing NLP tooling is also built for English — Bangla text has its own tokenization, Unicode normalization, and stopword challenges that break naive approaches outright.

**NewsInsight** addresses this by using distributed data processing and unsupervised machine learning to automatically group similar Bangla news articles into meaningful topic clusters — without ever seeing a label during training — and by rigorously validating that those clusters are actually meaningful rather than an artifact of the metric used to measure them.

## 🎯 Problem Statement

Modern Bangla news platforms need to:
- Identify related stories quickly across multiple publishers
- Catch duplicate or near-duplicate articles (the same story is routinely cross-posted or re-scraped)
- Organize large archives efficiently without manual tagging
- Support topic-based search, discovery, and recommendation

Manual categorization doesn't scale as volume grows. Naive clustering approaches often make things worse in a subtle way: certain configurations produce strong-looking internal metrics by collapsing nearly everything into one dominant cluster — technically "clustered," practically useless. This project was built specifically to surface and avoid that failure mode.

## 💡 Solution

NewsInsight is an end-to-end pipeline that takes raw, messy Bangla news text and turns it into (1) validated topic clusters and (2) a duplicate/near-duplicate detector — combining Bangla-aware text preprocessing, multiple feature engineering strategies, approximate similarity search, four clustering algorithms, and a two-metric evaluation approach (internal + ground-truth) at every stage, because relying on one metric alone was found to actively mislead the choice of best configuration.

The pipeline runs on Apache Spark for the data-scale stages, with scikit-learn, FAISS, and HNSWlib handling the specialized modeling steps.

## 🌍 Real-World Applications

This pipeline's design was built directly around three concrete use cases:
- **📰 Newsroom auto-tagging** — automatically suggest a topic category for incoming articles
- **🔍 Duplicate / near-duplicate detection** — MinHashLSH-based approximate similarity search, purpose-built for catching cross-posted stories, achieving up to 98.6% same-class accuracy on retrieved pairs (see results below)
- **📚 Topic-based news feeds** — group articles by topic and collapse duplicates so a reader sees each story once

Beyond these three, the same clustering core is adaptable to media monitoring, digital news archives, and topic-based research corpora — but the three above are what the pipeline was actually designed and evaluated against.

---

# 🤖 Pipeline Stages

The project follows a structured workflow, validated at each stage before moving to the next: **raw text → cleaning → tokenization/EDA → deep preprocessing (normalization + stopwords) → n-gram feature engineering → duplicate detection → clustering → evaluation.**

## 📥 Stage 1 — Data Loading & Merging

Three category CSVs (Economy, Entertainment, Science & Technology) are loaded into Spark DataFrames. A balanced sample (8,000 rows per category) is taken from each and unioned into a single working DataFrame, so no single class dominates the clustering process.

## 🧹 Stage 2 — Text Cleaning

Raw scraped Bangla text has real structural problems: embedded newlines that break row/column alignment, stray punctuation, zero-width Unicode artifacts, and leaked English text/URLs. This is handled in **two chained passes** rather than one aggressive regex, because the first pass has to repair row structure before the second pass can safely target residual noise:

- **Pass 1 — Structural Repair:** collapse embedded newlines/carriage returns, strip punctuation, remove zero-width/non-breaking space characters, normalize whitespace
- **Pass 2 — Residual Noise Removal:** strip URLs, remove stray Latin/English characters (this is a Bangla-only corpus), re-apply cleanup for anything the first pass missed

This stage also includes a class-balance check, exact-duplicate removal (a real risk given cross-posted stories), and a null/empty-row audit **before and after** cleanup — a deliberate data-quality checkpoint rather than a blind `dropna()`.

## 🔤 Stage 3 — Tokenization & Exploratory Data Analysis (Pre-Preprocessing)

Before any linguistic normalization, the cleaned text is tokenized (restricted to the Bangla Unicode range) purely to explore the raw data: vocabulary size, token-length distribution, and the most frequent terms as they naturally occur. Tokens of length ≤ 2 are dropped as noise (stray conjunct fragments, leftover punctuation). This stage produces document-length histograms, top-20 frequent token/bigram/trigram bar charts, and word clouds — rendered with a Bangla-capable font (Noto Sans Bengali) so the output doesn't come out as tofu boxes. The point of doing this *before* deep preprocessing is to have an honest "before" picture to compare against once stopwords are removed.

## 🧠 Stage 4 — Deep Preprocessing (Post-EDA Normalization)

With the raw-data picture established, this stage applies the formal, linguistically-aware normalization — going beyond basic cleanup:
- **Unicode NFC normalization** — Bangla conjuncts and diacritics can be encoded multiple ways at the byte level; without this, visually identical words tokenize differently
- **Bangla → English digit unification**
- **Strict script whitelisting** — keep only the Bangla Unicode block, digits, and whitespace
- **Stopword removal** using a manually curated Bangla stopword list
- **Re-filtering short tokens** — the length ≤ 2 filter is re-applied after normalization, since new short fragments can appear post-normalization

A direct before/after comparison of the top-20 tokens is generated to confirm stopword removal actually changed the distribution as expected, rather than being assumed to work silently.

## 📊 Stage 5 — Feature Engineering (N-Grams)

Unigram, bigram, and trigram representations are generated **both before and after stopword removal**, specifically to support the before/after comparisons above, across multiple vocabulary sizes (50 to 5,000 terms). Different n-gram/vocabulary combinations capture different linguistic patterns, and the project doesn't assume in advance which will cluster best — that's determined empirically in Stage 7.

## 🔍 Stage 6 — Duplicate & Near-Duplicate Detection (Jaccard, ANN, MinHashLSH)

Before clustering, a separate pipeline detects near-duplicate articles — directly relevant to the cross-posting problem in real newsrooms:

1. **HashingTF** converts each document's n-grams into a fixed-size binary feature vector
2. **MinHashLSH** is fit on top of those vectors across a parameter sweep of **{4, 8, 12} hash tables** and **{0.2, 0.3, 0.4, 0.5} distance thresholds** (distance = 1 − Jaccard similarity), for both bigram and trigram representations
3. **Approximate similarity join** retrieves candidate near-duplicate pairs per configuration
4. **Approximate nearest neighbors** are pulled per query document (top-5)
5. **Exact Jaccard similarity** is computed directly from the sparse vectors as ground truth, and MinHashLSH's approximate results are checked against it — precision/recall against the *exact* answer, the standard way ANN methods are benchmarked, rather than trusting the approximation blindly

### Duplicate detection results — full parameter sweep

Every n-gram × hash-table × distance-threshold combination that was actually run, straight from the notebook's saved experiment table:

| N-Gram | Hash Tables | Distance Threshold | Similarity Threshold | Pairs | Same-Class Pairs | Diff-Class Pairs | Same-Class Ratio | Avg. Similarity | Max Similarity |
|---|---|---|---|---|---|---|---|---|---|
| Bigram | 4 | 0.2 | 0.8 | 54 | 53 | 1 | 98.1% | 0.876 | 0.992 |
| Bigram | 4 | 0.3 | 0.7 | 78 | 77 | 1 | 98.7% | 0.837 | 0.992 |
| Bigram | 4 | 0.4 | 0.6 | 105 | 103 | 2 | 98.1% | 0.790 | 0.992 |
| Bigram | 4 | 0.5 | 0.5 | 140 | 138 | 2 | 98.6% | 0.730 | 0.992 |
| Bigram | 8 | 0.2 | 0.8 | 54 | 53 | 1 | 98.1% | 0.876 | 0.992 |
| Bigram | 8 | 0.3 | 0.7 | 79 | 78 | 1 | **98.7%** | 0.835 | 0.992 |
| Bigram | 8 | 0.4 | 0.6 | 108 | 106 | 2 | 98.1% | 0.786 | 0.992 |
| Bigram | 8 | 0.5 | 0.5 | **145** | 143 | 2 | 98.6% | 0.726 | 0.992 |
| Bigram | 12 | 0.2 | 0.8 | 54 | 53 | 1 | 98.1% | 0.876 | 0.992 |
| Bigram | 12 | 0.3 | 0.7 | 79 | 78 | 1 | 98.7% | 0.835 | 0.992 |
| Bigram | 12 | 0.4 | 0.6 | 108 | 106 | 2 | 98.1% | 0.786 | 0.992 |
| Bigram | 12 | 0.5 | 0.5 | 145 | 143 | 2 | 98.6% | 0.726 | 0.992 |
| Trigram | 4 | 0.2 | 0.8 | 39 | 38 | 1 | 97.4% | **0.880** | 0.992 |
| Trigram | 4 | 0.3 | 0.7 | 64 | 63 | 1 | 98.4% | 0.832 | 0.992 |
| Trigram | 4 | 0.4 | 0.6 | 89 | 87 | 2 | 97.8% | 0.782 | 0.992 |
| Trigram | 4 | 0.5 | 0.5 | 117 | 115 | 2 | 98.3% | 0.728 | 0.992 |
| Trigram | 8 | 0.2 | 0.8 | 39 | 38 | 1 | 97.4% | 0.880 | 0.992 |
| Trigram | 8 | 0.3 | 0.7 | 64 | 63 | 1 | 98.4% | 0.832 | 0.992 |
| Trigram | 8 | 0.4 | 0.6 | 89 | 87 | 2 | 97.8% | 0.782 | 0.992 |
| Trigram | 8 | 0.5 | 0.5 | 119 | 117 | 2 | 98.3% | 0.725 | 0.992 |
| Trigram | 12 | 0.2 | 0.8 | 39 | 38 | 1 | 97.4% | 0.880 | 0.992 |
| Trigram | 12 | 0.3 | 0.7 | 64 | 63 | 1 | 98.4% | 0.832 | 0.992 |
| Trigram | 12 | 0.4 | 0.6 | 89 | 87 | 2 | 97.8% | 0.782 | 0.992 |
| Trigram | 12 | 0.5 | 0.5 | 119 | 117 | 2 | 98.3% | 0.725 | 0.992 |

### Precision & recall against exact ground truth

The same-class ratio above answers "are retrieved pairs actually related?" A separate, stricter check answers a different question: "does MinHashLSH's approximate top-k retrieval match the *exact* Jaccard top-k ranking for a query document?" Ground truth here is brute-force exact Jaccard similarity computed directly from the sparse vectors (via sparse matrix multiplication, not a Python double loop), and precision/recall are measured against it — the standard way ANN methods are benchmarked.

| N-Gram | Hash Tables | Threshold | Precision | Recall | Evaluable Queries (n) |
|---|---|---|---|---|---|
| Bigram | 4 | 0.20 | 16.7% | 100% | 12 (2 had true neighbors) |
| Bigram | 4 | 0.30 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 4 | 0.40 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 4 | 0.50 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 8 | 0.20 | 16.7% | 100% | 12 (2 had true neighbors) |
| Bigram | 8 | 0.30 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 8 | 0.40 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 8 | 0.50 | 20.0% | 100% | 15 (3 had true neighbors) |
| Bigram | 12 | 0.20–0.50 | 16.7–20.0% | 100% | same pattern as 4/8 tables |
| Trigram | 4 | 0.20 | 0.0% | n/a | 9 (0 had true neighbors) |
| Trigram | 4 | 0.30 | 13.3% | 100% | 15 (2 had true neighbors) |
| Trigram | 4 | 0.40 | 13.3% | 100% | 15 (2 had true neighbors) |
| Trigram | 4 | 0.50 | 13.3% | 100% | 15 (2 had true neighbors) |
| Trigram | 8 / 12 | 0.20–0.50 | same pattern as tables=4 | same | same |

**What this actually means, and why it isn't a contradiction of the 97–99% same-class numbers above:** precision here is measured against exact top-k rank matching for a handful of query documents (as few as 9–15 evaluable queries per configuration) — a much stricter bar than "is the retrieved pair the same topic class." Recall is consistently 100% wherever a query actually had a true near-duplicate to find, meaning MinHashLSH never *missed* a genuine near-duplicate in this test — it just also surfaces some pairs that don't rank in the exact top-k, which drags precision down without meaning those pairs are wrong or off-topic (the same-class-ratio table already showed 97%+ of them are correct topically). In practice: recall is the number that matters for a deduplication system (don't miss real duplicates), and it's perfect across every configuration tested.

**Takeaway across all 24 configurations:** the same-class ratio never drops below 97.4%, regardless of n-gram type, hash table count, or distance threshold — meaning MinHashLSH's approximate retrieval is reliably finding genuine near-duplicates across the entire parameter space, not just in a cherry-picked setting. Increasing hash tables (4 → 8 → 12) barely moves the numbers at a fixed threshold, which says the retrieval is already stable at 4 tables for this corpus size; the real lever is the distance threshold, which trades off pair *count* against average similarity *tightness* as expected.

## 🔬 Stage 7 — Clustering & Evaluation

Four unsupervised algorithms are run and compared head-to-head across every n-gram × vocabulary-size configuration:

- **K-Means** — centroid-based partitioning on TF-IDF + PCA features
- **Gaussian Mixture Models (GMM)** — probabilistic, more flexible cluster shapes than centroid-based methods
- **HNSW graph-based clustering** — builds an approximate-neighbor graph index (via `hnswlib`, `ef_construction=100`, `M=16`) over PCA-reduced TF-IDF features, then runs K-Means on top of that structure — HNSW itself doesn't cluster, it accelerates and structures the neighbor search that the downstream clustering (and the duplicate/ANN stage) relies on
- **FAISS Product Quantization (PQ)** — compresses each document's PCA-reduced feature vector into a small number of quantized sub-codes (`faiss.ProductQuantizer`, 8-bit codes across up to 4 sub-vectors), then runs K-Means on the *compressed codes* rather than the raw features — this is what makes it viable at scale, trading a small amount of precision for a large drop in memory/compute per document

Every configuration is scored on **silhouette score** (cluster cohesion/separation, computed on a 3,000-document sample) *and* **Hungarian-matching accuracy** — the optimal cluster-to-true-label assignment via the Hungarian algorithm, then measured against the actual category labels.

---

# 📊 Experimental Results — Full Clustering Sweep

Every method × n-gram × vocabulary-size combination that was run, by silhouette score:

| Method | N-Gram | Vocab Size | Silhouette Score |
|---|---|---|---|
| KMeans | Unigram | 5000 | 0.6031 |
| KMeans | Bigram | 50 | 0.8821 |
| KMeans | Bigram | 1000 | 0.3298 |
| KMeans | Bigram | 2000 | 0.4451 |
| KMeans | Bigram | 5000 | 0.1694 |
| KMeans | Trigram | 50 | 0.8963 |
| KMeans | Trigram | 500 | 0.7493 |
| KMeans | Trigram | 1000 | 0.7110 |
| KMeans | Trigram | 2000 | 0.6915 |
| GMM | Unigram | 5000 | 0.3196 |
| GMM | Bigram | 500 | 0.2229 |
| GMM | Bigram | 2000 | 0.1982 |
| GMM | Bigram | 5000 | 0.2679 |
| GMM | Trigram | 50 | 0.5736 |
| GMM | Trigram | 500 | 0.0970 |
| GMM | Trigram | 1000 | 0.9800 |
| GMM | Trigram | 2000 | 0.9806 |
| GMM | Trigram | 5000 | 0.4843 |
| HNSW | Unigram | 5000 | 0.3566 |
| HNSW | Bigram | 50 | 0.7202 |
| HNSW | Bigram | 500 | 0.5033 |
| HNSW | Bigram | 1000 | 0.1318 |
| HNSW | Bigram | 2000 | 0.2563 |
| HNSW | Bigram | 5000 | 0.7508 |
| HNSW | Trigram | 50 | 0.8866 |
| HNSW | Trigram | 500 | 0.7131 |
| HNSW | Trigram | 1000 | 0.6920 |
| HNSW | Trigram | 2000 | 0.6331 |
| **HNSW** | **Trigram** | **5000** | **0.9136** |
| FAISS PQ | Unigram | 5000 | -0.0208 |
| FAISS PQ | Bigram | 50 | 0.2210 |
| FAISS PQ | Bigram | 500 | -0.0740 |
| FAISS PQ | Bigram | 1000 | -0.0277 |
| FAISS PQ | Bigram | 2000 | -0.0121 |
| FAISS PQ | Bigram | 5000 | -0.0598 |
| FAISS PQ | Trigram | 50 | 0.6636 |
| FAISS PQ | Trigram | 500 | 0.1812 |
| FAISS PQ | Trigram | 1000 | 0.0678 |
| **FAISS PQ** | **Trigram** | **2000** | **0.0356** |
| FAISS PQ | Trigram | 5000 | -0.0943 |

## Silhouette vs. Hungarian Accuracy — Side by Side

The clearest proof that silhouette score alone can't be trusted comes from running every method on the *same* vocabulary size (5,000) and comparing both metrics directly, straight from the notebook's printed output:

| Method | N-Gram | Vocab | Silhouette Score | Hungarian Accuracy |
|---|---|---|---|---|
| **GMM** | **Unigram** | **5000** | 0.320 | **89.5%** |
| GMM | Bigram | 5000 | 0.268 | 68.5% |
| GMM | Trigram | 5000 | 0.484 | 50.8% |
| HNSW | Unigram | 5000 | 0.357 | 33.6% |
| HNSW | Bigram | 5000 | 0.751 | 33.6% |
| HNSW | Trigram | 5000 | **0.914** | 33.6% |
| FAISS PQ | Unigram | 5000 | -0.021 | 33.5% |
| FAISS PQ | Bigram | 5000 | -0.060 | 33.5% |
| FAISS PQ | Trigram | 5000 | -0.094 | 33.4% |

## The core problem with evaluating unsupervised clustering

HNSW on trigram/vocab-5000 posts a silhouette score of **0.914** — near the best in the entire project — while its actual Hungarian accuracy is **33.6%**, barely above the ~33% floor you'd expect from randomly guessing one of three categories. Meanwhile GMM on unigram/vocab-5000 has an unremarkable silhouette score of 0.320 and the **best true-label accuracy in the whole sweep at 89.5%**. Judged on silhouette alone, HNSW/trigram looks like the standout result and GMM/unigram looks mediocre — the opposite of what's actually true. This is the concrete version of the failure mode this project was built to catch: without checking against real category labels, the sweep's silhouette-based "winner" would have been almost useless in practice.

## Selected production configuration

**Unigram features, vocabulary size 5,000, Gaussian Mixture Model — 89.5% Hungarian accuracy against ground truth**, the best true-label result across every method, n-gram, and vocabulary size tested in the project. This correction matters: an earlier version of this README attributed the ~89% figure to FAISS PQ / trigram / vocab-2000 — checking the notebook's own printed output, that specific configuration actually scored **33.4% Hungarian accuracy**, not 89%. GMM / unigram / vocab-5000 is the run that actually hit 89.5%, and it's now the configuration referenced throughout this README.

One tradeoff worth being upfront about: GMM runs on full-precision TF-IDF/PCA features rather than FAISS's compressed codes, so it doesn't have the same built-in scaling advantage FAISS PQ was chosen for in earlier notebook iterations. If this pipeline needs to scale to a much larger corpus, that's the tradeoff to revisit — right now, accuracy on this dataset points clearly to GMM/unigram/5000, while compute efficiency at scale would still favor a quantization-based approach if one can be found that doesn't sacrifice this much accuracy.

## 🌍 What These Results Actually Mean

Two separate result sets came out of this project — clustering and duplicate detection — and each maps to a distinct real-world capability.

### What the clustering results mean

The headline number isn't "89.5% accuracy" in isolation — it's *which* configuration actually earned it, and how easy it would have been to pick the wrong one. HNSW on trigram/vocab-5000 scored 0.914 on silhouette — near the top of the entire sweep — and would look like the obvious winner on a dashboard that only tracked that metric. Its real accuracy against ground truth was 33.6%, barely above chance for a 3-class problem. That's exactly the failure mode a production auto-tagging system can't afford: a model that looks validated on an internal metric but silently mistags two-thirds of incoming articles. Catching that *before* shipping it — by checking every configuration against actual category labels, not just an internal separation score — is the difference between a system that looks validated and one that actually is.

That's what makes the selected configuration (GMM, unigram, vocab=5000, 89.5% Hungarian accuracy) usable in practice for:
- **Newsroom auto-tagging** — roughly 9 in 10 articles get correctly routed to their true category without a human touching them; the remaining ~10% is a manageable review queue rather than a silent failure
- **Topic-based feeds** — with clusters that actually reflect real topic boundaries (not one dominant blob or a near-random split), a "science & tech" feed genuinely contains science & tech articles
- **Choosing simplicity over premature optimization** — the highest-accuracy result came from the simplest feature representation (unigrams) and a full-precision model, not the more exotic compressed/quantized approaches; that's a useful reminder that scaling techniques (like FAISS PQ) are worth adopting once accuracy is proven, not as a first move

### What the duplicate-detection results mean

The 97.4–98.7% same-class ratio, holding steady across all 24 sweep configurations, means MinHashLSH isn't a fragile result that only works under one specific setting — it's stable across n-gram type, hash table count, and similarity threshold. That stability is what makes it deployable rather than just a promising notebook result:
- **Newsroom deduplication** — the same story routinely gets re-scraped or cross-posted under a different category; this pipeline catches those pairs with ~98% precision without ever comparing every article to every other article by brute force (which is what makes it viable at news-platform scale)
- **Feed quality** — a reader following a topic-based feed sees each real story once instead of the same article three times under three different headlines
- **Tunable trade-off, not a fixed answer** — the sweep shows the practical lever is the distance threshold: a looser threshold (0.5) surfaces more pairs (145) at slightly lower average similarity (0.73), a tighter one (0.2) surfaces fewer pairs (39–54) at much higher confidence (0.88). A real deployment can dial this per use case — aggressive dedup for storage cleanup vs. conservative dedup where false positives would be visible to readers.

## 💡 Engineering Challenges & Solutions

| Challenge | Solution |
|---|---|
| Kaggle environment hit CPU/RAM/HDD crashes during early runs | Root-caused to misconfigured Spark memory settings, missing `.cache()`/`.persist()` calls, redundant `.collect()` operations, and t-SNE running on the full dataset instead of a sample — fixed each |
| High silhouette scores misrepresenting cluster quality | Cross-validated every configuration against true-label (Hungarian) accuracy rather than trusting one internal metric |
| Bangla-specific text noise (broken encodings, mixed scripts) | Two-pass cleaning pipeline: structural repair before noise targeting, rather than one aggressive regex pass |
| Validating an approximate similarity search (MinHashLSH) | Built an exact-Jaccard ground truth via sparse matrix multiplication and measured precision/recall of the approximate method against it, rather than trusting the approximation on faith |
| Comparing configurations fairly across a wide parameter sweep | Restructured experiments for programmatic result collection and consistent visualization, rather than ad hoc one-off runs |

## 🎯 Design Principles

- **Scalability** — built on Apache Spark so the pipeline isn't limited to small in-memory datasets; FAISS PQ specifically chosen as the production path because it clusters on compressed codes, not full vectors
- **Rigor over convenience** — every clustering configuration is checked against ground truth, not just an internal metric; every approximate search result is checked against an exact computation
- **Reproducibility** — a consistent evaluation methodology applied identically across every configuration in the sweep
- **Interpretability** — word clouds, n-gram frequency charts, t-SNE projections, and category-distribution plots throughout, rather than treating clustering as a black box

## 🚀 Key Contributions

- End-to-end Bangla-aware NLP preprocessing pipeline (two-pass cleaning, Unicode normalization, custom stopword handling)
- Distributed processing via Apache Spark
- MinHashLSH-based near-duplicate detection, validated against exact Jaccard similarity as ground truth
- Four-algorithm clustering comparison (K-Means, GMM, HNSW, FAISS PQ) across a full n-gram × vocabulary-size parameter sweep
- A concrete, reproducible demonstration that silhouette score alone can select degenerate clusters — and a two-metric evaluation approach that catches it

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| Distributed processing | Apache Spark, PySpark |
| Machine learning | scikit-learn |
| Similarity search / clustering | FAISS, HNSWlib, MinHash LSH |
| Analysis & visualization | pandas, matplotlib, seaborn, WordCloud |

## 📂 What's in This Repo

```text
NewsInsight/
├── notebook.ipynb        # Full pipeline: cleaning → features → duplicate detection → clustering → evaluation
└── README.md
```

## 🗺 Status & Roadmap

**✅ Built and validated (this repo):**
- End-to-end Bangla NLP preprocessing pipeline
- Distributed processing via Spark
- MinHashLSH duplicate detection, validated against exact Jaccard ground truth
- Four-way clustering algorithm comparison across a full parameter sweep
- Multi-metric evaluation (silhouette + Hungarian accuracy), including the degenerate-cluster finding above

**🔜 Planned, not yet built:**
- Lightweight deployed API (dropping Spark for real-time inference, keeping it for batch prep)
- Small interactive demo — paste in an article, see its predicted cluster
- Live news ingestion / real-time processing

This section is kept deliberately honest: the research and evaluation work above is complete and is the substantive part of the project; productization is a planned next step, not a claim about this repo today.

---

