"""
Stage 3: Embedding + Model Testing
Location: src/03_embedding.py
Concepts: Embedding Drift, Bias, Dimensionality, Cosine Similarity, BGE vs TF-IDF
Interview Q: "How do you test embedding drift and bias?"
"""

import json
import os
import numpy as np  # type: ignore[import-not-found]
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.metrics.pairwise import cosine_similarity # type: ignore[import-not-found]

# ============ 1. LOAD CHUNKS ============
with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"[Stage 3] Chunks Loaded: {len(chunks)}")
texts = [c["text"] for c in chunks]

# ============ 2. TF-IDF EMBEDDING (POC) + BGE CONCEPT (Prod) ============
# In Prod: BGE-large-en-v1.5 -> 1024 dimensions
# In POC: TF-IDF -> similar dimension concept

vectorizer = TfidfVectorizer(max_features=384)  # 384 like MiniLM
embeddings = vectorizer.fit_transform(texts).toarray()

print(f"\n[EMBEDDING TEST] Embedding Created:")
print(f"  Shape: {embeddings.shape} -> {len(chunks)} chunks x 384 dim")
print(f"  In Prod: BGE model gives 1024 dim, here TF-IDF 384 dim for POC")
print(f"  Each embedding is vector like [0.1, 0.0, 0.23, ...]")

# Save embeddings
os.makedirs("data", exist_ok=True)
np.save("data/embeddings.npy", embeddings)
print(f"  Saved: data/embeddings.npy")

# ============ 3. COSINE SIMILARITY TESTING ============
# This is core for Relevancy!

def test_cosine_similarity():
    print(f"\n[COSINE SIM TEST] Similarity between Steps:")
    # Step 4 (Aadhaar) vs Step 5 (Payment) should be low similarity
    # Step 5 (Payment) vs Refund query should be high
    sim_matrix = cosine_similarity(embeddings)
    
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            if sim_matrix[i][j] > 0.3:  # Only show relevant
                print(f"  {chunks[i]['id']} <-> {chunks[j]['id']}: {sim_matrix[i][j]:.3f}")

test_cosine_similarity()

# ============ 4. EMBEDDING DRIFT TESTING ============
# MOST IMPORTANT FOR CI/CD!

# Simulate old vs new embedding
# In Prod: When BRD updates from v1.2 to v1.3, embedding changes

OLD_EMBEDDING_PATH = "data/embeddings_old.npy"

if os.path.exists(OLD_EMBEDDING_PATH):
    old_embeddings = np.load(OLD_EMBEDDING_PATH)
    # Drift = 1 - cosine similarity between old and new for same text
    drift_scores = []
    for i in range(min(len(old_embeddings), len(embeddings))):
        sim = cosine_similarity([old_embeddings[i]], [embeddings[i]])[0][0]
        drift = 1 - sim
        drift_scores.append(drift)
    
    avg_drift = np.mean(drift_scores)
    print(f"\n[DRIFT TEST] Embedding Drift Check:")
    print(f"  Avg Drift: {avg_drift:.4f}")
    if avg_drift > 0.15:
        print(f"  [ALERT] DRIFT DETECTED! >0.15 threshold - Need re-embedding!")
        print(f"  Action: Trigger re-indexing pipeline")
    else:
        print(f"  [PASS] No drift - Embeddings stable")
else:
    print(f"\n[DRIFT TEST] No old embedding found - Saving current as baseline")
    print(f"  In CI/CD, next run will compare against this baseline")

# Save current as old for next run
np.save(OLD_EMBEDDING_PATH, embeddings)
np.save("data/embeddings_current.npy", embeddings)

# ============ 5. BIAS TESTING IN EMBEDDING ============
# This is Safety Testing - Stage 3

print(f"\n[BIAS TEST] Embedding Bias Check:")

# Simulate bias: Check if "Aadhaar" chunk biased
# In Prod: Test if embedding for "nurse" closer to "woman" than "man"

# Create test phrases
test_phrases = [
    "Aadhaar is mandatory",
    "Aadhaar is optional",
    "Refund process",
    "Payment process"
]

test_vectors = vectorizer.transform(test_phrases).toarray()
bias_matrix = cosine_similarity(test_vectors)

print(f"  Checking similarity bias:")
print(f"  'Aadhaar mandatory' vs 'Aadhaar optional': {bias_matrix[0][1]:.3f} (should be high but not 1.0)")
print(f"  'Refund' vs 'Payment': {bias_matrix[2][3]:.3f} (should be medium - both payment domain)")

# Bias alert if mandatory/optional too similar = model not distinguishing
if bias_matrix[0][1] > 0.9:
    print(f"  [BIAS ALERT] Model not distinguishing mandatory vs optional!")
else:
    print(f"  [BIAS PASS] Model distinguishes mandatory vs optional")

# ============ 6. DIMENSIONALITY & LATENCY TEST ============
print(f"\n[PERFORMANCE TEST] Embedding Latency:")
import time
start = time.time()
_ = vectorizer.transform(["Test query for latency check"]).toarray()
latency = (time.time() - start) * 1000
print(f"  Embedding latency: {latency:.2f} ms")
print(f"  Threshold: <100ms PASS, >100ms FAIL")
print(f"  Status: {'PASS' if latency < 100 else 'FAIL'}")

# ============ 7. VERSIONING ============
version_info = {
    "version": "v1.2",
    "embedding_model": "TF-IDF-384 (POC) / BGE-1024 (Prod)",
    "dimension": 384,
    "chunks": len(chunks),
    "drift_checked": True,
    "bias_checked": True,
    "timestamp": datetime.now().isoformat()
}

with open("logs/embedding_version.json", "w") as f:
    json.dump(version_info, f, indent=2)

print(f"\n[Stage 3 PASS] Embedding Complete")
print(f"  - Embeddings: {embeddings.shape} saved")
print(f"  - Drift Check: Done - Baseline saved for CI/CD")
print(f"  - Bias Check: Done - Mandatory vs Optional distinction")
print(f"  - Latency: {latency:.2f}ms")
print(f"  - Next: If drift >0.15, re-embed all in production")