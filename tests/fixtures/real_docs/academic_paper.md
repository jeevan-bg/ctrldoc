# Sparse Index Compression for Bounded-Memory Retrieval

*Anonymous authors. Submitted to a fictional retrieval workshop, 2026.*

## Abstract

We present a sparse-index compression scheme for bounded-memory
retrieval over technical document collections. Existing inverted
indexes assume working-set sizes that exceed available memory on
edge devices. Our approach combines block-aware delta coding with
posting-list pruning guided by query-frequency statistics. On a
12k-document evaluation corpus we observe a 4.7× reduction in
on-disk index size with a 0.6-point drop in recall@10 relative to
an uncompressed baseline.

## 1. Introduction

Inverted indexes remain the dominant data structure for full-text
retrieval. Production deployments assume that the per-shard posting
lists fit entirely in memory, but this assumption breaks down on
edge hardware where the working-set budget is measured in tens of
megabytes. Prior work on compressed indexes targets server-class
memory hierarchies and either pays a steep query-time decompression
cost or stores auxiliary structures that themselves dominate the
budget.

We make three contributions:

1. A block-aware delta coding scheme that aligns posting blocks to
   the page cache and amortizes decompression across consecutive
   queries.
2. A pruning policy that drops postings ranked below the 95th
   percentile of historical query frequency for their term.
3. An evaluation on a 12k-document technical-documentation corpus
   showing that combined block-aware coding and pruning reduces
   index size by 4.7× while keeping recall@10 within 0.6 points of
   an uncompressed baseline.

## 2. Method

Let *T* denote the vocabulary of indexed terms and *p(t)* the
posting list for term *t*. We first apply block-aware delta coding
to each *p(t)*: postings are partitioned into fixed-size blocks of
128 entries, and within each block we store deltas relative to the
block's first document id using variable-byte coding.

We then apply query-frequency pruning. For each term *t* we observe
the per-posting click-through count over a 30-day query trace and
drop postings whose count falls below the 95th percentile for that
term. Terms whose entire posting list falls below threshold are
retained at their head — at least one posting per term remains.

## 3. Results

On the evaluation corpus we measure index size, recall@10, and
median query latency across 1k held-out queries. The uncompressed
baseline occupies 312 MB and achieves recall@10 of 0.84. Block-aware
coding alone reduces size to 138 MB with no recall loss. Pruning
alone reduces size to 96 MB with a 0.4-point recall drop. The
combined scheme reaches 66 MB with a 0.6-point recall drop.

Median query latency rises from 7 ms (uncompressed) to 11 ms
(combined). The latency overhead is dominated by block
decompression and is amortized over consecutive queries within the
same block.

## 4. Limitations

Our pruning policy depends on a 30-day query trace; cold-start
deployments fall back to retaining every posting. The block-aware
scheme assumes a 4 KB page size; on systems with larger pages the
block boundary may be re-tuned. We have not evaluated on
non-English corpora.

## 5. Conclusion

Block-aware coding and frequency-guided pruning are complementary
and can be applied together without compromising end-to-end recall
on a bounded memory budget.
