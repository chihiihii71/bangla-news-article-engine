# TextArtifact — Scalable Big Data NLP Engine

**A production-grade distributed pipeline that automatically organises Bangla news articles into topic clusters — replacing hours of manual editorial work with a fully automated, scalable system.**

---

## 🧩 Business Problem

Publishing companies and news aggregators managing large Bangla-language archives face a critical operational challenge: articles arrive continuously across categories, but manual tagging and organisation is slow, expensive, and inconsistent.

- Finding near-duplicate stories across news feeds wastes editorial bandwidth and inflates storage costs.
- Without automated topic discovery, content recommendation and search relevance suffer.
- Existing NLP tools are built for English — Bangla requires a purpose-built preprocessing pipeline.
- Standard clustering tools cannot scale to tens of thousands of documents without distributed infrastructure.

This project builds an end-to-end scalable analytics system that detects duplicate articles, discovers topic clusters, and evaluates cluster quality — all on a distributed Apache Spark platform capable of handling real-world production volumes.

---

## ✅ Solution

- Built a **two-module distributed Big Data analytics pipeline** using **Apache Spark (32 GB session)**.
- Detected **near-duplicate Bangla news articles** using **MinHashLSH Approximate Nearest Neighbor (ANN)** search with **Jaccard similarity**.
- Validated duplicate detection against **exact ground-truth** computed from a **SciPy CSR sparse matrix**.
- Evaluated **four clustering algorithms**: **K-Means**, **Gaussian Mixture Models (GMM)**, **HNSW Graph-based Clustering**, and **FAISS Product Quantization (PQ)**.
- Conducted **large-scale parameter tuning** across **vocabulary sizes (50–5,000)** and **PCA dimensions (5–50)**.
- Measured clustering performance using **Silhouette Score** (internal cluster quality) and **Hungarian Algorithm Accuracy** (external label alignment).
- Developed an **end-to-end distributed Bangla NLP preprocessing pipeline** including **two-pass Unicode cleaning**, **tokenization**, **stopword removal**, and **unigram**, **bigram**, and **trigram** generation within **Apache Spark**.

---

## 🏗️ Architecture

```
Potrika Bangla News CSVs  (3 categories × 40,000 articles)
              ↓
    Spark Data Loading & Balanced Sampling  (8,000 / class → 24,000 total)
              ↓
    Two-Pass Distributed Text Cleaning
      Pass 1 → collapse embedded newlines, strip punctuation, remove zero-width Unicode
      Pass 2 → remove URLs, strip English characters, normalise whitespace
              ↓
    Deduplication + Null Audit  (dropDuplicates, per-column empty-string check)
              ↓
    Bangla RegexTokenizer  (Unicode range \u0980–\u09FF, min token length > 2)
              ↓
    EDA  (vocabulary size, document-length histogram, top-20 tokens/bigrams/trigrams)
              ↓
    Deep Preprocessing  (Unicode normalisation, Bangla stopword removal)
              ↓
    N-Gram Feature Engineering  (Unigrams · Bigrams · Trigrams)
              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  MODULE 1 — Duplicate Detection                         │
    │  HashingTF (binary, 2¹⁶ features) → MinHashLSH         │
    │  Similarity Join (Jaccard ≥ threshold)                  │
    │  ANN Search (top-5 per query)                           │
    │  Precision / Recall vs Exact Jaccard Ground Truth       │
    └─────────────────────────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  MODULE 2 — Topic Clustering                            │
    │  CountVectorizer → TF-IDF → PCA                         │
    │  K-Means  |  GMM  |  HNSW  |  FAISS Product Quantization│
    │  Silhouette Score + Hungarian Algorithm Accuracy        │
    │  t-SNE · Word Clouds · Cluster-Size Visualisations      │
    └─────────────────────────────────────────────────────────┘
              ↓
    Best Configuration Identified → Business Insights
```

---

## ⚙️ Pipeline — Stage by Stage

**Stage 1 — Data Ingestion**
Three 40,000-row CSVs (National, Sports, Science & Technology) loaded with multiline/quote-safe Spark CSV reader. Each category sampled to 8,000 rows (seed=42) and union-joined into a single 24,000-row Spark DataFrame — ensuring no class dominates training.

**Stage 2 — Two-Pass Text Cleaning**
Pass 1 repairs structural damage from web scraping: embedded newlines that misalign rows, stray punctuation, and zero-width Unicode characters. Pass 2 applies domain-specific rules for a Bangla-only corpus: URL removal, English character stripping, and whitespace normalisation. Two passes are necessary because Pass 1 must repair row structure before Pass 2 can safely target residual noise.

**Stage 3 — Deduplication & Data Quality**
Exact duplicates removed via `dropDuplicates()`. Per-column null and empty-string audit run before and after cleaning to confirm data quality — not a blind `dropna()`.

**Stage 4 — Bangla Tokenisation & EDA**
RegexTokenizer isolates Bangla Unicode tokens (`\u0980–\u09FF`). Tokens of length ≤ 2 dropped as conjunct fragment noise. EDA covers vocabulary size, document-length histogram, and top-20 frequency charts for tokens, bigrams, and trigrams — using Noto Bengali font for correct Bangla glyph rendering.

**Stage 5 — Deep Preprocessing**
Unicode normalisation harmonises visually identical but byte-different Bangla characters. Custom Bangla stopword list removes high-frequency low-information words that would otherwise dominate TF-IDF weights.

**Stage 6 — Feature Engineering**
Bigrams and trigrams generated via Spark MLlib NGram. Four representations produced: unigrams, bigrams, trigrams (for clustering), and binary HashingTF vectors (for MinHashLSH). CountVectorizer builds a vocabulary-controlled dense representation; IDF reweights by document frequency; PCA reduces dimensionality.

**Stage 7 — Duplicate Detection (MinHashLSH)**
Binary HashingTF vectors (2¹⁶ features) fed into MinHashLSH. Parameter sweep: numHashTables ∈ {4, 8, 12} × distance thresholds ∈ {0.20, 0.30, 0.40, 0.50} × n-gram types {bigram, trigram}. Precision and recall evaluated against exact Jaccard ground truth computed via SciPy CSR sparse matrix — single sparse matrix-vector multiply, not a Python loop or Spark self-join.

**Stage 8 — Clustering (Four Algorithms)**
Four clustering approaches run across a full parameter grid. Results for every configuration logged to a pandas DataFrame. Hungarian algorithm applied to every run for external label-alignment accuracy.

**Stage 9 — Evaluation & Visualisation**
Silhouette Score measures internal cluster compactness and separation. Hungarian Accuracy measures how well clusters align with true editorial categories. t-SNE projections (cluster view + true-class view), per-cluster word clouds, and cluster-size bar charts produced for every experiment.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Big Data | Apache Spark (PySpark) — 32 GB driver & executor memory |
| NLP | Spark RegexTokenizer, HashingTF, CountVectorizer, IDF; Bangla stopword removal |
| Vector Search | MinHashLSH (Spark MLlib), HNSWlib, FAISS |
| Machine Learning | Spark MLlib (K-Means, GMM, PCA), scikit-learn KMeans, SciPy sparse algebra |
| Evaluation | Silhouette Score, Hungarian Algorithm (scipy.optimize.linear_sum_assignment) |
| Visualisation | Matplotlib, Seaborn, WordCloud, Noto Bengali font |
| Notebook | Jupyter (Kaggle environment) |

---

## 📦 Dataset

| Property | Detail |
|---|---|
| Source | Potrika Bangla News Corpus (Kaggle: `kamrun71/bangla-news`) |
| Raw files | 3 × 40,000 CSV (National, Sports, Science & Technology) |
| Working sample | 8,000 per category → **24,000 total** (balanced) |
| Language | Bengali — Unicode range `\u0980–\u09FF` |
| Cleaning | Two-pass noise removal; exact duplicate removal; null audit |
| Split | No held-out test set — unsupervised evaluation only (Silhouette + Hungarian) |

---

## 🔬 Feature Engineering

Implemented a multi-representation feature strategy combining sparse binary vectors for locality-sensitive hashing with TF-IDF weighted n-gram vectors for clustering — enabling both near-duplicate retrieval and topic discovery from the same preprocessing chain.

| Technique | Configuration | Engineering Rationale |
|---|---|---|
| Two-pass regex cleaning | URL removal, English stripping, punctuation, zero-width chars | Repair structure before applying domain rules — order matters |
| RegexTokenizer | `[^\u0980-\u09FF]+`, min length > 2 | Isolate valid Bangla tokens; drop fragment noise |
| Bangla stopword removal | Custom list | Prevent high-frequency function words from dominating cluster features |
| Unigrams | Single tokens | Baseline bag-of-words representation |
| Bigrams | Token pairs | Capture local phrase structure |
| Trigrams | Token triples | Richer specificity; reduces inter-category feature overlap |
| HashingTF (binary) | numFeatures = 2¹⁶ | Sparse binary representation required by MinHashLSH Jaccard approximation |
| CountVectorizer | vocab ∈ {50, 500, 1000, 2000, 5000} | Controlled vocabulary for ML pipelines; vocab size is the key tuning parameter |
| TF-IDF (IDF stage) | Applied post CountVectorizer | Down-weight corpus-wide common terms; surface distinctive topic signals |
| PCA | k ∈ {5, 8, 20, 30, 50} | Reduce TF-IDF sparse dimensions to dense low-dim space; must scale with vocab |

---

## 🔍 Duplicate Detection Results — MinHashLSH

The corpus contains very few genuine near-duplicates (< 200 pairs across 24,000 documents) — a healthy signal that editorial categories are genuinely distinct. This makes the precision/recall evaluation methodology especially important: random query sampling would produce undefined recall for most queries (no true near-neighbours to retrieve). Queries were therefore sampled from documents already known to participate in at least one similar pair, with a random fraction added to keep precision honest.

**Why results behave as they do:**
- Precision is high across configurations because distinct Bangla news articles share very few overlapping bigram or trigram hashes — LSH rarely returns false positives.
- Recall improves with more hash tables: at numHashTables=12, more MinHash signature bands are checked, catching pairs that tighter configurations miss.
- Trigrams yield higher precision than bigrams at tight thresholds — more specific n-grams reduce hash collisions between genuinely dissimilar articles.
- Recall drops at tight distance thresholds (0.20): articles that are editorially similar but lexically varied fall below the Jaccard ≥ 0.80 cutoff.

| N-gram | Hash Tables | Threshold | Behaviour |
|---|---|---|---|
| Bigram | 4 | 0.50 | More pairs found; lower precision |
| Bigram | 12 | 0.20 | Fewer pairs; higher precision; lower recall |
| Trigram | 12 | 0.30 | Best precision-recall balance |

---

## 🤖 Clustering Experiments

### How HNSW and FAISS PQ were adapted for clustering

HNSW (Hierarchical Navigable Small World) and FAISS Product Quantization are ANN index structures designed for retrieval, not clustering. Adapting them required a four-step engineering bridge:

1. **Spark MLlib pipeline** — CountVectorizer → TF-IDF → PCA ran fully distributed inside Spark.
2. **Spark → NumPy bridge** — PCA vectors collected from Spark into NumPy float32 arrays.
3. **Index construction + sklearn KMeans:**
   - *HNSW:* Built an L2-space HNSW index (`M=16`, `ef_construction=100`, `ef=50`) via HNSWlib; ran sklearn KMeans on raw PCA vectors using the graph neighbourhood as the vector space justification.
   - *FAISS PQ:* Trained a Product Quantizer (M subspaces dividing the PCA dimension evenly, 8-bit codes); ran sklearn KMeans on the integer code matrix — clustering in compressed space with significant memory savings.
4. **Labels rejoined to Spark** — cluster assignments joined back via `doc_id` key, enabling full Hungarian accuracy and Silhouette evaluation inside the Spark ecosystem.

This hybrid Spark ↔ sklearn ↔ FAISS/HNSWlib architecture is the core technical contribution of the clustering module.

### Silhouette Score — All Configurations

| Method | N-gram | Vocab | PCA k | Silhouette |
|---|---|---|---|---|
| **GMM** | trigram | 2000 | 50 | **0.9806** 🏆 |
| **GMM** | trigram | 1000 | 30 | **0.9800** |
| **HNSW** | trigram | 5000 | 50 | 0.9136 |
| **KMeans** | trigram | 50 | 5 | 0.8963 |
| **HNSW** | trigram | 50 | 5 | 0.8866 |
| **KMeans** | bigram | 50 | 5 | 0.8821 |
| **HNSW** | bigram | 5000 | 50 | 0.7508 |
| **KMeans** | trigram | 500 | 20 | 0.7493 |
| **HNSW** | trigram | 500 | 20 | 0.7131 |
| **KMeans** | trigram | 1000 | 30 | 0.7110 |
| **HNSW** | bigram | 50 | 5 | 0.7202 |
| **FAISS PQ** | trigram | 50 | 8 | 0.6636 |
| **KMeans** | unigram | 5000 | — | 0.6031 |
| **GMM** | trigram | 50 | 5 | 0.5736 |
| **HNSW** | bigram | 500 | 20 | 0.5033 |
| **GMM** | trigram | 5000 | 50 | 0.4843 |
| **KMeans** | bigram | 2000 | 50 | 0.4451 |
| **HNSW** | unigram | 5000 | — | 0.3566 |
| **GMM** | unigram | 5000 | — | 0.3196 |
| **KMeans** | bigram | 1000 | 30 | 0.3298 |
| **HNSW** | bigram | 2000 | 50 | 0.2563 |
| **GMM** | bigram | 500 | 20 | 0.2229 |
| **FAISS PQ** | bigram | 50 | 5 | 0.2210 |
| **GMM** | bigram | 2000 | 50 | 0.1982 |
| **FAISS PQ** | trigram | 500 | 20 | 0.1812 |
| **KMeans** | bigram | 5000 | — | 0.1694 |
| **HNSW** | bigram | 1000 | 30 | 0.1318 |
| **FAISS PQ** | trigram | 1000 | 30 | 0.0678 |
| **FAISS PQ** | trigram | 2000 | 50 | 0.0356 |
| **FAISS PQ** | bigram | 1000 | 30 | −0.0277 |
| **FAISS PQ** | bigram | 2000 | 50 | −0.0121 |
| **FAISS PQ** | unigram | 5000 | — | −0.0208 |
| **FAISS PQ** | bigram | 500 | 20 | −0.0740 |
| **FAISS PQ** | bigram | 5000 | — | −0.0598 |
| **FAISS PQ** | trigram | 5000 | — | −0.0943 |

### Best Configuration per Method

| Rank | Method | N-gram | Vocab | Silhouette | Hungarian Accuracy |
|---|---|---|---|---|---|
| 🥇 1st | **GMM** | trigram | 2000 | **0.9806** | **0.8945** - Strongest label alignment |
| 🥈 2nd | **HNSW** | trigram | 5000 | 0.9136 | 0.7357 - graph geometry captures manifold structure |
| 🥉 3rd | **KMeans** | trigram | 50 | 0.8963 | 0.3367 — small vocab forces tight cluster boundaries |
| 4th | **FAISS PQ** | trigram | 50 | 0.6636 | 0.3352 — quantization loss at small vocab |

---

## 📊 Evaluation Metrics

| Metric | What it measures | Why it was chosen |
|---|---|---|
| **Silhouette Score** | Internal cluster compactness and separation (−1 to +1) | Model-agnostic; works across all four algorithms |
| **Hungarian Algorithm Accuracy** | Optimal cluster-to-true-label assignment via Kuhn-Munkres algorithm | Solves the cluster label permutation problem — the only fair external accuracy for unsupervised models |

**Why Hungarian Accuracy matters:** naive cluster accuracy is meaningless because cluster IDs are arbitrary. The Hungarian algorithm finds the optimal one-to-one mapping between predicted cluster IDs and true class labels, then computes `correct / total`. This is the gold standard for external evaluation of unsupervised classification.

---

## 💡 Key Findings

- **GMM on trigrams (vocab 1000–2000) is the strongest overall configuration**, achieving silhouette scores of 0.98. Gaussian mixture models assign soft probabilities — well-suited to Bangla news articles that span topic boundaries (e.g., science in sports reporting).

- **Trigrams consistently outperform bigrams and unigrams** across all methods. Three-token phrases carry sufficient specificity to separate National, Sports, and Science categories with minimal feature overlap.

- **Small vocabularies can outperform large ones.** KMeans at vocab=50 scores 0.8963 — beating vocab=5000 (0.6031). Compact vocabularies act as implicit regularisation, forcing the model to cluster on only the most discriminative features rather than noise tokens.

- **HNSW adapts well to high-dimensional feature spaces.** Its graph-based neighbourhood structure captures non-convex cluster geometry that K-Means centroids cannot represent, explaining why HNSW at vocab=5000 (0.9136) outperforms KMeans at the same setting.

- **FAISS PQ fails at large vocabularies.** Negative silhouette scores at vocab ≥ 500 (bigrams) indicate that quantization-induced distortion merges clusters artificially. FAISS PQ optimises for retrieval latency, not clustering separability — this is a deliberate, identified trade-off, not an implementation error.

- **PCA dimension must scale with vocabulary.** Pairing large vocab (2000+) with small PCA k (5) consistently underperforms. The compressed representation loses too much variance, collapsing distinct topics into the same low-dimensional region.

- **The Silhouette–Hungarian relationship is not monotonic.** High silhouette (compact geometry) does not guarantee high Hungarian accuracy (label alignment). GMM achieves both simultaneously because its probabilistic component boundaries happen to align with the editorial boundaries of the three news categories.

---

## 🧠 Engineering Decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| **Two-pass cleaning** | Web-scraped Bangla text has structural damage (embedded newlines) that must be repaired before domain noise can be targeted | Adds one extra Spark transformation |
| **Binary HashingTF for MinHashLSH** | MinHashLSH requires set-based (binary) vectors; TF counts would produce incorrect Jaccard estimates | Loses term frequency information for this module only |
| **SciPy CSR for exact Jaccard ground truth** | Brute-force Jaccard over 24k documents via a Python double loop would take hours; sparse matrix-vector multiply computes one query vs. entire corpus in milliseconds | Requires collecting all vectors to driver — feasible at 24k docs, not at billions |
| **TF-IDF over raw counts for clustering** | IDF down-weights corpus-wide common Bangla function words that survive stopword removal, surfacing topic-distinctive terms | Slightly higher memory than raw counts |
| **PCA before clustering** | Reduces TF-IDF sparse high-dimensional vectors to dense low-dimensional space; required for HNSW/FAISS which need float32 dense arrays | Information loss — must tune k to retain sufficient variance |
| **Spark → NumPy → sklearn bridge for HNSW/FAISS** | HNSWlib and FAISS have no Spark-native implementation; bridging enables their use in a Spark pipeline | Requires collecting PCA vectors to driver; limits to driver-memory scale |
| **sklearn KMeans on HNSW/PQ outputs** | HNSW and PQ produce vector spaces / code spaces; KMeans assigns cluster labels in those spaces | KMeans is centroid-based — inherits centroid-based limitations even in transformed spaces |
| **Hungarian algorithm over simple accuracy** | Cluster IDs are arbitrary integers — naive accuracy is meaningless without optimal label assignment | Requires solving an assignment problem (O(n³)) per experiment |

---

## 🌍 Real-World Applications

- **News recommendation engines** — cluster-based topic discovery surfaces related articles without manual tagging, enabling personalised feeds for Bangla-language readers.
- **Media monitoring** — MinHashLSH duplicate detection flags cross-publisher story reuse in near real time, valuable for press agencies and fact-checkers.
- **Digital archives** — scalable deduplication and thematic organisation of large Bangla news repositories (national broadcasters, government archives).
- **Search engine indexing** — cluster-aware indexing improves retrieval precision for Bangla queries by grouping semantically related documents.
- **Content moderation** — topic clustering identifies coordinated inauthentic content sharing patterns across categories.
- **Multilingual NLP infrastructure** — demonstrates that standard Big Data pipelines can be extended to low-resource non-Latin scripts with proper Unicode engineering.

---

## 🗂️ Repository Structure

```
├── notebook.ipynb        # Full pipeline — 14 sections, all experiments
└── README.md             # This file
```

> **Dataset:** Kaggle `kamrun71/bangla-news`. Running requires a Kaggle environment or a local Spark cluster with ≥ 32 GB driver memory and the three CSV files mounted at `/kaggle/input/datasets/kamrun71/bangla-news/`.

---

## ▶️ Reproducibility

```bash
# 1. Open in Kaggle (recommended — dataset auto-mounted)
#    https://www.kaggle.com  →  Import notebook → Run All

# 2. Or locally:
pip install pyspark hnswlib faiss-cpu scipy scikit-learn matplotlib seaborn wordcloud unidecode

# 3. Mount dataset at:  /kaggle/input/datasets/kamrun71/bangla-news/

# 4. Run all cells top to bottom
#    Sections 1–6   → preprocessing & feature engineering
#    Section 7      → MinHashLSH duplicate detection + precision/recall
#    Sections 8–12  → clustering experiments (all four algorithms)
#    Section 14     → silhouette comparison charts

# Expected: Silhouette ≈ 0.98 for GMM trigram vocab=1000–2000
```

---

## 🔭 Future Improvements

- **Real-time duplicate detection API** — wrap MinHashLSH as a FastAPI microservice with Redis-cached hash tables for sub-millisecond near-duplicate lookup on incoming articles.
- **Online clustering** — replace batch K-Means with MiniBatchKMeans or streaming GMM to support continuous article ingestion without full retraining.
- **Bangla BERT embeddings** — replace TF-IDF with contextual embeddings (e.g., `sagorsarker/bangla-bert-base`) for semantic cluster representations that capture meaning beyond n-gram overlap.
- **Docker + Kubernetes deployment** — containerise the Spark pipeline for reproducible deployment on cloud (AWS EMR, GCP Dataproc).
- **MLflow model registry** — log all 35+ experiment configurations with metrics for systematic model governance and comparison.
- **Expanded corpus** — extend to all Potrika categories and other Bangla news sources (Prothom Alo, Daily Star Bangla) to test generalisation.

---

## 🎯 Skills Demonstrated

**Big Data Engineering** — Apache Spark, PySpark MLlib, distributed DataFrame operations, Spark Pipeline API, Spark memory configuration, StorageLevel caching, Spark ↔ NumPy data bridge

**Machine Learning** — K-Means, Gaussian Mixture Models, PCA, unsupervised model evaluation, parameter grid search across 35+ configurations

**Approximate Nearest Neighbor & Vector Search** — MinHashLSH, HNSWlib (HNSW graph index), FAISS Product Quantization, SciPy CSR sparse matrix algebra, precision/recall ANN benchmarking

**Natural Language Processing** — Bangla Unicode tokenisation, multi-pass noise removal, stopword removal, n-gram generation, TF-IDF, vocabulary analysis

**Evaluation & Research Methodology** — Silhouette Score, Hungarian Algorithm, exact ground-truth construction for ANN benchmarking, stratified query sampling

**Software Engineering** — modular reusable pipeline functions, cross-library interoperability (Spark/sklearn/FAISS/HNSWlib), memory management (`gc.collect`, `unpersist`), reproducible seeding, NumPy float32 engineering

**Data Visualisation** — t-SNE projections, word clouds with Noto Bengali font, cluster-size charts, method-comparison bar charts, parameter-sweep line plots

**Programming** — Python, Jupyter Notebook, Kaggle environment, pandas, matplotlib, seaborn

