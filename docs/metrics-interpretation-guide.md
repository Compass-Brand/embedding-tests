# Retrieval Metrics Interpretation Guide

This guide explains how to interpret the metrics produced by the embedding-tests framework, helping you understand embedding model quality and make informed comparisons.

## Table of Contents

1. [Quick Reference Card](#quick-reference-card)
2. [Core Ranking Metrics](#core-ranking-metrics)
3. [Multi-k Metrics](#multi-k-metrics)
4. [Statistical Measures](#statistical-measures)
5. [Performance Metrics](#performance-metrics)
6. [Metric Relationships](#metric-relationships)
7. [Decision Framework](#decision-framework)
8. [Common Patterns](#common-patterns)
9. [Benchmarking Best Practices](#benchmarking-best-practices)

---

## Quick Reference Card

| Metric | Range | Higher is Better? | Primary Use |
|--------|-------|-------------------|-------------|
| MRR | 0-1 | Yes | First relevant result ranking |
| MAP | 0-1 | Yes | Overall ranking quality |
| NDCG@k | 0-1 | Yes | Graded relevance ranking |
| Recall@k | 0-1 | Yes | Coverage of relevant docs |
| Precision@k | 0-1 | Yes | Accuracy of top-k results |
| F1@k | 0-1 | Yes | Balance of P and R |
| Hit Rate@k | 0-1 | Yes | Success rate (at least one hit) |
| R-Precision | 0-1 | Yes | Precision at R relevant docs |

**Rule of Thumb for BEIR Benchmarks:**
- NDCG@10 > 0.5: Excellent
- NDCG@10 0.3-0.5: Good
- NDCG@10 0.1-0.3: Fair
- NDCG@10 < 0.1: Poor

---

## Core Ranking Metrics

### MRR (Mean Reciprocal Rank)

**What it measures:** How high the *first* relevant document appears in the results.

**Formula:** Average of 1/rank across all queries, where rank is the position of the first relevant document.

**Interpretation:**
| MRR Value | Meaning |
|-----------|---------|
| 1.0 | First result is always relevant |
| 0.5 | First relevant result is typically at position 2 |
| 0.33 | First relevant result is typically at position 3 |
| 0.1 | First relevant result is typically at position 10 |

**When to use:** When users typically only look at the first few results (search engines, chatbots, Q&A systems).

**Example:**
```
Query 1: First relevant at position 1 → 1/1 = 1.0
Query 2: First relevant at position 3 → 1/3 = 0.33
Query 3: First relevant at position 2 → 1/2 = 0.5
MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

---

### MAP (Mean Average Precision)

**What it measures:** Overall ranking quality across *all* relevant documents, not just the first.

**Formula:** Average of Average Precision (AP) scores. AP is the average of precision values at each position where a relevant document is found.

**Interpretation:**
| MAP Value | Meaning |
|-----------|---------|
| 1.0 | All relevant docs ranked at top positions |
| 0.5 | Moderate ranking quality |
| < 0.1 | Relevant docs scattered throughout results |

**When to use:** When you care about finding *all* relevant documents, not just one (research, legal discovery, medical literature).

**MAP vs MRR:**
- MRR: "How quickly do I find *something* relevant?"
- MAP: "How well are *all* relevant documents ranked?"

**Example:**
```
Relevant docs: {A, B, C}
Retrieved: [A, X, B, Y, C]

Position 1: A is relevant → Precision = 1/1 = 1.0
Position 3: B is relevant → Precision = 2/3 = 0.67
Position 5: C is relevant → Precision = 3/5 = 0.6

AP = (1.0 + 0.67 + 0.6) / 3 = 0.76
```

---

### NDCG@k (Normalized Discounted Cumulative Gain)

**What it measures:** Ranking quality with position-based discounting (higher positions matter more).

**Formula:** DCG@k / IDCG@k, where DCG sums relevance scores divided by log2(position+1).

**Interpretation:**
| NDCG@10 Value | Quality Level | Typical Use Case |
|---------------|---------------|------------------|
| > 0.7 | Excellent | Production-ready |
| 0.5 - 0.7 | Good | Acceptable for most uses |
| 0.3 - 0.5 | Fair | May need improvement |
| 0.1 - 0.3 | Poor | Significant issues |
| < 0.1 | Very Poor | Model not suitable |

**Why it's the primary BEIR metric:**
1. Handles graded relevance (not just binary)
2. Penalizes relevant docs at lower positions
3. Normalized to [0,1] for easy comparison

**Position discount visualization:**
```
Position 1: 1.00 (full credit)
Position 2: 0.63
Position 3: 0.50
Position 5: 0.39
Position 10: 0.29
Position 20: 0.23
```

---

### R-Precision

**What it measures:** Precision when you retrieve exactly R documents, where R = number of relevant documents.

**Interpretation:**
- R-Precision = 1.0: Top R results are exactly the R relevant documents
- R-Precision = 0.5: Half of top R are relevant
- Equal to Recall@R by definition

**When to use:** When the number of relevant documents varies significantly across queries.

---

## Multi-k Metrics

### Understanding k Values

The framework computes metrics at k = 1, 3, 5, 10, 20 to show how performance changes with result set size.

| k Value | Typical Use Case |
|---------|------------------|
| k=1 | Single-answer systems (chatbots) |
| k=3 | Quick glance results (mobile search) |
| k=5 | Standard search result page |
| k=10 | Extended search, RAG context |
| k=20 | Comprehensive retrieval |

### Recall@k

**What it measures:** What fraction of relevant documents appear in the top-k results.

**Interpretation:**
```
Recall@1 = 0.02  → Only 2% of relevant docs in first result
Recall@5 = 0.06  → 6% of relevant docs in top 5
Recall@10 = 0.10 → 10% of relevant docs in top 10
Recall@20 = 0.10 → No improvement beyond k=10 (plateau)
```

**Key insight:** When Recall@k stops increasing, you've captured all retrievable relevant docs.

### Precision@k

**What it measures:** What fraction of top-k results are relevant.

**Interpretation:**
```
Precision@1 = 0.30  → 30% of queries have relevant first result
Precision@5 = 0.28  → 28% of top-5 results are relevant
Precision@10 = 0.27 → 27% of top-10 results are relevant
```

**Key insight:** Precision typically decreases as k increases (you start including less relevant docs).

### F1@k

**What it measures:** Harmonic mean of Precision@k and Recall@k.

**Formula:** 2 × (P@k × R@k) / (P@k + R@k)

**When to use:** When you need to balance precision and recall.

**Interpretation:**
- High P, Low R → F1 is moderate (you're precise but missing docs)
- Low P, High R → F1 is moderate (you find docs but with noise)
- Balanced P and R → F1 is maximized

### Hit Rate (Success@k)

**What it measures:** Fraction of queries where *at least one* relevant document appears in top-k.

**Interpretation:**
```
Hit Rate@1 = 0.30  → 30% of queries have a relevant first result
Hit Rate@5 = 0.60  → 60% of queries have a hit in top 5
Hit Rate@10 = 0.66 → 66% of queries have a hit in top 10
```

**Key insight:** Hit Rate shows user success rate. If Hit Rate@5 = 0.60, 40% of users won't find what they need in the top 5.

---

## Statistical Measures

Each metric includes aggregate statistics to help understand variance:

| Statistic | Meaning | Use |
|-----------|---------|-----|
| mean | Average across queries | Primary comparison value |
| std | Standard deviation | Consistency measure |
| min | Worst query performance | Identify failure modes |
| max | Best query performance | Identify strengths |
| median | Middle value (p50) | Robust central tendency |
| p25 | 25th percentile | Lower quartile boundary |
| p75 | 75th percentile | Upper quartile boundary |

### Interpreting Standard Deviation

| Pattern | Interpretation |
|---------|----------------|
| Low std, high mean | Consistently good |
| Low std, low mean | Consistently poor |
| High std, high mean | Good on average but unpredictable |
| High std, moderate mean | Very inconsistent results |

**Example:**
```
Model A: NDCG@10 mean=0.35, std=0.15
Model B: NDCG@10 mean=0.32, std=0.05

→ Model A is better on average but less consistent
→ Model B is more predictable (lower variance)
→ For critical applications, Model B might be preferred
```

### Percentile Analysis

```
p25 = 0.10, median = 0.25, p75 = 0.45

This tells you:
- 25% of queries have NDCG < 0.10 (struggling)
- 50% of queries have NDCG < 0.25
- 25% of queries have NDCG > 0.45 (doing well)
- The distribution is likely right-skewed
```

---

## Performance Metrics

### Total Time

Total wall-clock time for the entire benchmark.

### Embedding Time

Time spent generating embeddings (corpus + queries).

**Insight:** If embedding_time ≈ total_time, the bottleneck is the model, not retrieval.

### Queries per Second

Throughput metric for latency-sensitive applications.

| QPS | Interpretation |
|-----|----------------|
| > 100 | Real-time capable |
| 10-100 | Interactive use |
| 1-10 | Batch processing |
| < 1 | Very slow (large models) |

### Corpus Chunks

Number of text chunks indexed. Useful for:
- Understanding memory requirements
- Comparing chunking strategies
- Estimating index size

---

## Metric Relationships

### The Precision-Recall Tradeoff

```
↑ k  →  ↑ Recall  →  ↓ Precision

As you retrieve more documents:
- You find more relevant ones (recall increases)
- You also include more irrelevant ones (precision decreases)
```

### MRR vs Hit Rate

```
MRR is always ≤ Hit Rate

MRR = 1.0 implies Hit Rate@1 = 1.0
Hit Rate@1 = 1.0 implies MRR = 1.0

But:
Hit Rate@5 = 0.6 does not tell you MRR
(relevant doc could be at position 1, 2, 3, 4, or 5)
```

### NDCG vs MAP

Both measure ranking quality, but:

| Aspect | NDCG | MAP |
|--------|------|-----|
| Relevance | Graded (0-1) | Binary (relevant/not) |
| Focus | Position-weighted | All relevant positions |
| Normalization | Against ideal | Against relevant count |

**When they diverge:**
- High NDCG, low MAP: Top results are excellent, but some relevant docs are buried
- High MAP, lower NDCG: Good coverage but suboptimal ordering

---

## Decision Framework

### Choosing the Right Metric

| Your Priority | Primary Metric | Secondary Metrics |
|---------------|----------------|-------------------|
| User finds *something* fast | MRR, Hit Rate@k | Precision@1 |
| User finds *everything* | Recall@k, MAP | R-Precision |
| Best overall ranking | NDCG@10 | MAP, MRR |
| Balanced retrieval | F1@k | Precision@k, Recall@k |
| Consistency matters | std of NDCG@10 | p25, min |

### Model Selection Criteria

**For RAG (Retrieval-Augmented Generation):**
```
Primary: Recall@10 (need relevant context)
Secondary: NDCG@10 (better docs first = better generation)
Threshold: Hit Rate@10 > 0.8 (most queries should succeed)
```

**For Search:**
```
Primary: NDCG@10 (ranking quality)
Secondary: MRR (first result quality)
Threshold: NDCG@10 > 0.4, MRR > 0.5
```

**For Q&A:**
```
Primary: MRR (single best answer)
Secondary: Hit Rate@1 (first result success)
Threshold: MRR > 0.6
```

---

## Common Patterns

### Pattern 1: High Precision, Low Recall
```
Precision@10 = 0.8, Recall@10 = 0.1

Diagnosis: Model retrieves accurate results but misses many relevant documents.
Cause: Embeddings are too specific/narrow.
Action: Consider larger k, query expansion, or hybrid search.
```

### Pattern 2: High Recall, Low Precision
```
Precision@10 = 0.1, Recall@10 = 0.8

Diagnosis: Model finds most relevant docs but includes too much noise.
Cause: Embeddings are too general/broad.
Action: Fine-tune model, add reranker, or use stricter thresholds.
```

### Pattern 3: High MRR, Low MAP
```
MRR = 0.7, MAP = 0.2

Diagnosis: First relevant doc ranked well, but others are buried.
Cause: Model captures primary topic but misses related content.
Action: Acceptable for Q&A, problematic for comprehensive retrieval.
```

### Pattern 4: Recall Plateau
```
Recall@5 = 0.10, Recall@10 = 0.10, Recall@20 = 0.10

Diagnosis: No additional relevant docs found beyond k=5.
Cause:
  a) All retrievable relevant docs already found (good)
  b) Remaining relevant docs not retrievable (bad)
Action: Check if relevant docs have very different vocabulary.
```

### Pattern 5: High Variance
```
NDCG@10: mean=0.35, std=0.40, min=0.0, max=1.0

Diagnosis: Very inconsistent performance across queries.
Cause: Model excels on some query types, fails on others.
Action: Analyze failure cases, consider query classification.
```

---

## Benchmarking Best Practices

### 1. Use Multiple Metrics

Never rely on a single metric. A comprehensive view requires:
- At least one ranking metric (NDCG or MAP)
- At least one coverage metric (Recall)
- At least one efficiency metric (MRR or Hit Rate)

### 2. Consider the k Values

Choose k based on your application:
```yaml
# Search application
primary_k: 10
compare_at: [1, 5, 10]

# RAG application
primary_k: 5
compare_at: [3, 5, 10]
```

### 3. Check Statistical Significance

With standard deviation, you can estimate significance:
```
If |mean_A - mean_B| > 2 × max(std_A, std_B) / sqrt(n_queries)
Then the difference is likely significant
```

### 4. Report Confidence Intervals

```
NDCG@10 = 0.35 ± 0.05 (95% CI)

More meaningful than just the mean.
```

### 5. Document Dataset Characteristics

Results depend heavily on:
- Corpus size (larger = harder)
- Query complexity
- Relevance density (relevant docs per query)
- Domain (technical vs general)

---

## Interpreting Real Results

### Example Analysis

```
Model: qwen3-embedding-0.6b (fp16)
Dataset: nano-nfcorpus (50 queries)

Aggregate:
  MRR:         0.4210
  MAP:         0.0563
  R-Precision: 0.2725 (±0.3051)

NDCG@k:        @1=0.30  @3=0.31  @5=0.29  @10=0.24
Recall@k:      @1=0.02  @3=0.03  @5=0.06  @10=0.10
Hit Rate@k:    @1=0.30  @3=0.50  @5=0.60  @10=0.66
```

**Analysis:**

1. **MRR = 0.42**: First relevant doc typically appears around position 2-3. Decent for a 0.6B model.

2. **MAP = 0.056**: Very low. Many relevant documents are not being found or are buried deep in results.

3. **Hit Rate progression (0.30 → 0.50 → 0.60 → 0.66)**:
   - 30% of queries succeed at k=1
   - Jumps to 50% at k=3 (big improvement)
   - Plateaus around 66% at k=10
   - 34% of queries have no relevant doc in top 10

4. **Low Recall (<10% at k=10)**: The corpus has many relevant documents per query, but the model only finds ~10% of them.

5. **High R-Precision std (0.31)**: Very inconsistent - some queries work well, others fail completely.

**Recommendations:**
- This model is suitable for quick Q&A (decent MRR)
- Not suitable for comprehensive retrieval (low MAP/Recall)
- Consider larger models (4B, 8B) for better coverage
- Consider adding a reranker to improve top results

---

## Glossary

| Term | Definition |
|------|------------|
| Relevant | A document that satisfies the information need |
| Retrieved | Documents returned by the system |
| Ground truth | Known relevant documents (qrels) |
| k | Number of top results considered |
| Graded relevance | Relevance on a scale (0-3) vs binary |
| Binary relevance | Relevant (1) or not (0) |
| Corpus | Collection of documents to search |
| Query | User's information request |

---

## Further Reading

- [BEIR Benchmark Paper](https://arxiv.org/abs/2104.08663)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [IR Evaluation Measures](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
