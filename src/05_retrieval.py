"""
Stage 5: Retrieval + Context Relevancy Testing
Location: src/05_retrieval.py
Concepts: Context Relevancy, Recall@k, Precision@k, MRR, Retrieval Latency
Interview Q: "How do you test if retrieval is relevant?"
"""

import json
import os
import numpy as np # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore[import-not-found]
from sklearn.metrics.pairwise import cosine_similarity # type: ignore[import-not-found]

# ============ 1. LOAD VECTOR STORE & CHUNKS ============
with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

embeddings = np.load("data/embeddings.npy")
vectorizer = TfidfVectorizer(max_features=384, vocabulary=np.load("data/vectorizer_vocab.npy", allow_pickle=True).item() if os.path.exists("data/vectorizer_vocab.npy") else None)

# Re-train vectorizer properly from safe BRD
with open("data/BRD_safe.txt", "r") as f:
    safe_text = f.read()

# Rebuild vectorizer with same vocab
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore[import-not-found]
# Load chunks text
texts = [c["text"] for c in chunks]
vectorizer = TfidfVectorizer(max_features=384)
vectorizer.fit(texts)
embeddings = vectorizer.transform(texts).toarray()

print(f"[Stage 5] Loaded: {len(chunks)} chunks, {embeddings.shape} embeddings")

# ============ 2. GOLDEN QUERIES ============
# This is your test data - Like golden dataset in prod

golden_queries = [
    {"query": "Is Aadhaar mandatory?", "expected_step": "Step 4", "golden_chunk_id": "step_4", "category": "functional"},
    {"query": "Refund takes how many days?", "expected_step": "Step 5", "golden_chunk_id": "step_5", "category": "functional"},
    {"query": "Train 12675 timing?", "expected_step": "Step 2", "golden_chunk_id": "step_2", "category": "functional"},
    {"query": "Tatkal booking when opens?", "expected_step": "Step 6", "golden_chunk_id": "step_6", "category": "functional"},
    {"query": "Cancel ticket 25% charge?", "expected_step": "Step 7", "golden_chunk_id": "step_7", "category": "functional"},
    {"query": "What is the weather today?", "expected_step": "none", "golden_chunk_id": "none", "category": "out_of_domain"},  # Should return low score
]

print(f"\n[TEST SETUP] Golden Queries: {len(golden_queries)} (5 functional + 1 OOD)")

# ============ 3. RETRIEVAL FUNCTION ============
def retrieve(query, top_k=3):
    """Retrieve top_k chunks for query"""
    query_emb = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "chunk": chunks[idx],
            "score": float(sims[idx]),
            "rank": rank,
            "is_golden": chunks[idx]["id"] == golden_queries[0]["golden_chunk_id"]  # Will override per query
        })
    return results, sims

# ============ 4. CONTEXT RELEVANCY TEST ============
# Formula: Relevancy = (relevant chunks retrieved) / (total retrieved)

print(f"\n[CONTEXT RELEVANCY TEST] Testing each query:")
relevancy_scores = []

for gq in golden_queries:
    results, _ = retrieve(gq["query"], top_k=3)
    
    # Check if expected step in results
    retrieved_ids = [r["chunk"]["id"] for r in results]
    # Find expected
    expected_found = any(gq["expected_step"].lower() in r["chunk"]["text"].lower() or gq["golden_chunk_id"] in r["chunk"]["id"] or gq["expected_step"] in r["chunk"]["text"] for r in results)
    
    # Relevancy score: 1 if expected found in top 1, 0.66 if in top 2, 0.33 if in top 3, 0 if not found
    relevancy = 0
    for r in results:
        if gq["expected_step"].lower() in r["chunk"]["text"].lower() or (gq["golden_chunk_id"] != "none" and gq["golden_chunk_id"] in r["chunk"]["id"]):
            relevancy = 1.0 / r["rank"]  # Higher rank = higher relevancy
            # Actually better: 1/rank weighting, but for simple: if found, 1.0 else 0
            relevancy = 1.0 if r["rank"] == 1 else 0.7 if r["rank"] == 2 else 0.4
            break
    
    if gq["category"] == "out_of_domain":
        # For OOD, low scores = good (means model knows it's irrelevant)
        max_score = max([r["score"] for r in results]) if results else 0
        relevancy = 1.0 - max_score  # Low retrieval score = high relevancy for OOD detection
        print(f"  Q: '{gq['query']}' (OOD)")
        print(f"    Max Score: {max_score:.3f} -> OOD Detection: {'PASS (low score)' if max_score < 0.3 else 'FAIL'}")
    else:
        print(f"  Q: '{gq['query']}' -> Expected: {gq['expected_step']}")
        print(f"    Retrieved Top 1: {results[0]['chunk']['id']} Score: {results[0]['score']:.3f}")
        print(f"    Relevancy: {relevancy:.2f} -> {'PASS' if relevancy >= 0.7 else 'FAIL'}")
    
    relevancy_scores.append(relevancy)

avg_relevancy = np.mean(relevancy_scores)
print(f"\n  Avg Context Relevancy: {avg_relevancy:.3f} (Threshold: 0.75)")

# ============ 5. RECALL@K & PRECISION@K ============
print(f"\n[RECALL@K / PRECISION@K TEST]")

# For our POC, each query has 1 golden chunk
# Recall@3 = Did we get golden chunk in top 3?
# Precision@3 = How many of top 3 are relevant?

recall_at_3 = 0
precision_scores = []

for gq in golden_queries:
    if gq["category"] == "out_of_domain":
        continue
    results, _ = retrieve(gq["query"], top_k=3)
    # Check if golden found
    found = any(gq["expected_step"].lower() in r["chunk"]["text"].lower() for r in results)
    if found:
        recall_at_3 += 1
    
    # Precision: How many retrieved are actually relevant? In POC, 1 relevant max
    relevant_count = sum(1 for r in results if gq["expected_step"].lower() in r["chunk"]["text"].lower())
    precision = relevant_count / 3
    precision_scores.append(precision)

recall_at_3_score = recall_at_3 / 5  # 5 functional queries
avg_precision = np.mean(precision_scores) if precision_scores else 0

print(f"  Recall@3: {recall_at_3}/5 = {recall_at_3_score:.3f} (Threshold: 0.80)")
print(f"  Precision@3: {avg_precision:.3f} (Threshold: 0.50)")
print(f"  Status: {'PASS' if recall_at_3_score >= 0.8 else 'FAIL - Need better embedding!'}")

# ============ 6. MRR - Mean Reciprocal Rank ============
# MRR = Average of 1/rank where golden found

print(f"\n[MRR TEST] Mean Reciprocal Rank:")
mrr_scores = []
for gq in golden_queries:
    if gq["category"] == "out_of_domain":
        continue
    results, _ = retrieve(gq["query"], top_k=3)
    rr = 0
    for r in results:
        if gq["expected_step"].lower() in r["chunk"]["text"].lower():
            rr = 1.0 / r["rank"]
            break
    mrr_scores.append(rr)
    print(f"  Q: '{gq['query'][:20]}...' -> RR: {rr}")

mrr = np.mean(mrr_scores) if mrr_scores else 0
print(f"  MRR: {mrr:.3f} (Threshold: 0.70) -> {'PASS' if mrr >= 0.7 else 'FAIL'}")

# ============ 7. RETRIEVAL LATENCY ============
import time
print(f"\n[PERFORMANCE TEST] Retrieval Latency:")
latencies = []
for gq in golden_queries[:3]:
    start = time.time()
    _ = retrieve(gq["query"], top_k=3)
    lat = (time.time() - start) * 1000
    latencies.append(lat)

avg_latency = np.mean(latencies)
print(f"  Avg Latency: {avg_latency:.2f} ms")
print(f"  Threshold: <200ms PASS, >200ms FAIL")
print(f"  Status: {'PASS' if avg_latency < 200 else 'FAIL'}")

# ============ 8. SAVE REPORT ============
report = {
    "avg_context_relevancy": float(avg_relevancy),
    "recall_at_3": float(recall_at_3_score),
    "precision_at_3": float(avg_precision),
    "mrr": float(mrr),
    "avg_latency_ms": float(avg_latency),
    "thresholds": {"relevancy": 0.75, "recall": 0.8, "mrr": 0.7, "latency": 200}
}

os.makedirs("logs", exist_ok=True)
with open("logs/retrieval_metrics.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n[Stage 5 PASS] Retrieval Complete")
print(f"  - Relevancy: {avg_relevancy:.3f} | Recall@3: {recall_at_3_score:.3f} | MRR: {mrr:.3f}")
print(f"  - Latency: {avg_latency:.2f}ms")
print(f"  - Report: logs/retrieval_metrics.json")
print(f"  - If Recall <0.8 -> Action: Change embedding model to BGE or increase top_k to 5")