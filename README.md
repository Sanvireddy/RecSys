# Movie Recommendation System

A progression of recommender system techniques implemented from scratch in Python, spanning non-personalised popularity baselines, memory-based collaborative filtering, and a deep-learning Two-Tower neural model.

---

## Repository Contents

| Notebook | Dataset | Techniques |
|---|---|---|
| `content-based-filtering.ipynb` | MovieLens 100K | Popularity baselines, genre filtering, Precision@K |
| `collaborative-filtering.ipynb` | MovieLens 100K | User-User CF, Item-Item CF, Cosine / Pearson / Adjusted Cosine |
| `twotower.ipynb` | MovieLens 1M | Two-Tower neural model, in-batch negatives, FAISS retrieval |

---

## Notebook Breakdown

### 1. `content-based-filtering.ipynb` — Popularity & Genre Baselines

The starting point: non-personalised recommenders that establish a performance floor.

**Data loaded:** `u.data` (100K ratings), `u.item` (movie metadata + 19 binary genre flags), `u.user` (demographics).

**EDA:**
- Rating distribution (1–5 stars)
- User activity distribution (ratings per user histogram)
- Genre trends over time (top-5 genres plotted by release year)

**Recommenders implemented:**
- **Global Top-N** — ranks movies by average rating with a minimum-ratings threshold (default ≥ 50) to avoid obscure single-rating outliers.
- **Per-User Top-N** — same as above but excludes movies the target user has already rated.
- **Genre-Based Top-N** — filters the candidate set to a single genre before ranking by average rating; the simplest form of content-based filtering using item attributes.

**Evaluation:**
- Metric: Precision@K
- Protocol: per-user 80/20 train/test split (ensures every test user also has training data)
- Demonstrates the cold-start failure of a naive global split and why per-user splitting is necessary
- Baseline result: **Precision@10 ≈ 0.07–0.08**

---

### 2. `collaborative-filtering.ipynb` — Memory-Based CF

Moves to personalised recommendations by learning from user-item interaction patterns. All similarity functions are implemented from scratch without ML library wrappers.

**Data:** same MovieLens 100K tables as above, pivoted into a 943 × 1682 user-item rating matrix (~6% dense).

**Similarity metrics (all implemented from scratch):**

| Metric | Key property |
|---|---|
| Cosine similarity | NaN-filled with 0; fast but ignores rating scale bias |
| Pearson correlation | Computed on co-rated items only; accounts for user rating bias |
| Adjusted cosine similarity | Mean-centres by user before cosine; removes both user and item bias |

**User-User CF:**
- Weighted average prediction (plain)
- Mean-centered (bias-corrected) prediction — preferred variant
- Top-N recommendation by predicting ratings for all unseen items and ranking

**Item-Item CF:**
- Mean-centers the rating matrix before computing adjusted cosine between items
- Predicts via weighted average of the user's ratings on similar items

**Similarity caching:**
- Pre-computes the full user × user similarity matrix to avoid redundant computation during evaluation
- Pre-computes the full item × item similarity matrix with per-user mean-centering applied before storage

**Evaluation:**
- Metric: Precision@K
- Two protocols: per-user 80/20 split and leave-one-out (hold out most recent rating per user by timestamp)
- Evaluates on a sampled subset of users using the cached similarity matrices
- Expected results: User-User CF (Pearson) **~0.10–0.15**, Item-Item CF (Adjusted Cosine) **~0.12–0.18**
- Includes a note on out-of-range predictions from adjusted cosine and the fix (clip to [1, 5] or use for ranking only)

---

### 3. `twotower.ipynb` — Neural Two-Tower Model

Scales up to MovieLens 1M and replaces hand-crafted similarity with learned dense embeddings. Uses PyTorch throughout.

**Data:** MovieLens 1M — `movies.dat`, `ratings.dat`, `users.dat` (approx. 1M ratings, 6K users, 4K movies). Genres are parsed from pipe-delimited strings and encoded into integer IDs.

**Architecture:**

```
UserTower
  ├── nn.Embedding(num_users, 32)          ← user ID embedding
  ├── nn.Embedding(7, 8)                   ← age-bucket embedding (7 buckets)
  └── MLP [40 → 128 → 64 → 512]           ← BatchNorm + ReLU + Dropout

ItemTower
  ├── nn.Embedding(num_items, 32)          ← item ID embedding
  ├── nn.EmbeddingBag(18, 8, mode='mean') ← multi-hot genre embedding
  └── MLP [40 → 128 → 64 → 512]           ← BatchNorm + ReLU + Dropout

TwoTowerModel
  └── dot(user_emb, item_emb) / exp(log_temperature)   ← learnable temperature
```

**Training:**
- Loss: in-batch negative cross-entropy (`F.cross_entropy` over a batch_size × batch_size similarity matrix, diagonal = positives)
- Negative sampling: 4 negatives per positive sampled from items outside the user's interaction history
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Schedule: 5-epoch linear warmup → cosine decay over 50 total epochs
- DataLoader: batch size 512, shuffled, num_workers=2

**Retrieval at inference:**
- All item embeddings are pre-computed via `build_item_index`
- FAISS `IndexFlatIP` (exact inner product search) is built over the item embedding matrix
- Embeddings are L2-normalised before indexing, making inner product equivalent to cosine similarity

**Evaluation:**
- Protocol: leave-one-out (hold out each user's most recent rating by timestamp as the test item)
- Metrics: Hit Rate@K (HR@K) and NDCG@K

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, manipulation, matrix operations |
| `matplotlib` | EDA visualisations |
| `scikit-learn` | `train_test_split` for evaluation splits |
| `torch` (`nn`, `optim`, `DataLoader`) | Two-Tower model definition and training |
| `faiss-cpu` | Approximate nearest-neighbour retrieval index |

---

## Dataset

- **MovieLens 100K** — 100,000 ratings by 943 users on 1,682 movies. Used in the baseline and CF notebooks. Source: [GroupLens](https://grouplens.org/datasets/movielens/100k/).
- **MovieLens 1M** — ~1 million ratings by 6,040 users on 3,706 movies, with richer user demographics. Used in the Two-Tower notebook. Source: [GroupLens](https://grouplens.org/datasets/movielens/1m/).

Both datasets use explicit 1–5 star ratings.

---

## Results Summary

| Model | Metric | Score |
|---|---|---|
| Global popularity baseline | Precision@10 | ~0.07–0.08 |
| User-User CF (Pearson, mean-centered) | Precision@10 | ~0.10–0.15 |
| Item-Item CF (Adjusted Cosine) | Precision@10 | ~0.12–0.18 |
| Two-Tower (MovieLens 1M, FAISS) | HR@10 / NDCG@10 | evaluated in notebook |

---

## How to Run

Each notebook is self-contained. The dataset path in each notebook points to a Kaggle input directory:

```
/kaggle/input/datasets/trishna8/movielens-100k-dataset/ml-100k/     ← notebooks 1 & 2
/kaggle/input/datasets/odedgolden/movielens-1m-dataset/              ← notebook 3
```

To run locally, download the datasets from GroupLens and update the `basepath` variable at the top of each notebook.

For the Two-Tower notebook, install FAISS before running:
```bash
pip install faiss-cpu
```
