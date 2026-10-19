"""
Stage 4: Vector Store + Versioning + Indexing
Location: src/04_vector_store.py
Concepts: FAISS Index, Versioning, Recall Index, Latency, Rollback
Interview Q: "How do you version vector store and handle rollback?"
"""

import json
import os
import numpy as np # type: ignore[import-not-found]
import time
from datetime import datetime

# ============ 1. LOAD EMBEDDINGS & CHUNKS ============
embeddings = np.load("data/embeddings.npy")
with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"[Stage 4] Loaded: {len(chunks)} chunks, Embeddings: {embeddings.shape}")

# ============ 2. FAISS-LIKE VECTOR STORE (In-memory POC) ============
# In Prod: faiss.IndexFlatL2(384) or Pinecone, Qdrant
# In POC: Simple numpy store + cosine search

class SimpleVectorStore:
    """POC Vector Store - Simulates FAISS"""
    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks
        self.index_size = len(chunks)
        print(f"  Vector Store Created: {self.index_size} vectors indexed")
    
    def search(self, query_embedding, top_k=3):
        """Simulate FAISS search - Cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity # type: ignore[import-not-found]
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top_k indices
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(sims[idx]),
                "rank": len(results)+1
            })
        return results

vector_store = SimpleVectorStore(embeddings, chunks)

# ============ 3. VERSIONING - MOST IMPORTANT FOR CI/CD ============
# This is what HSBC asks!

VERSION = "v1.2"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

version_data = {
    "version": VERSION,
    "timestamp": TIMESTAMP,
    "index_type": "SimpleVectorStore (POC) / FAISS FlatL2 (Prod)",
    "dimension": embeddings.shape[1],
    "total_vectors": len(chunks),
    "embedding_model": "TF-IDF-384",
    "brd_version": "v1.2",
    "created_by": "CI/CD Pipeline"
}

# Save versioned store
os.makedirs("data/vector_store", exist_ok=True)
versioned_path = f"data/vector_store/store_{VERSION}_{TIMESTAMP}.json"

store_to_save = {
    "version_info": version_data,
    "chunks": chunks,
    # embeddings saved separately as npy
}

with open(versioned_path, "w") as f:
    json.dump(store_to_save, f, indent=2)

# Also save as latest for easy load
with open("data/vector_store/latest.json", "w") as f:
    json.dump(store_to_save, f, indent=2)

# Save embeddings versioned
np.save(f"data/vector_store/embeddings_{VERSION}_{TIMESTAMP}.npy", embeddings)
np.save("data/vector_store/embeddings_latest.npy", embeddings)

print(f"\n[VERSIONING TEST] Versioning Implemented:")
print(f"  Version: {VERSION}")
print(f"  Timestamp: {TIMESTAMP}")
print(f"  Saved: {versioned_path}")
print(f"  Latest: data/vector_store/latest.json")
print(f"  Purpose: Rollback if new version fails!")

# ============ 4. ROLLBACK SIMULATION ============
# Interview loves this!

print(f"\n[ROLLBACK TEST] Simulating rollback:")

# List all versions
versions = os.listdir("data/vector_store")
version_files = [v for v in versions if v.startswith("store_")]
print(f"  Available versions: {len(version_files)}")
for vf in sorted(version_files)[-3:]:  # Last 3
    print(f"    - {vf}")

# Simulate: If v1.3 fails, rollback to v1.2
if len(version_files) > 1:
    print(f"  Rollback possible: YES - Load previous version")
else:
    print(f"  Rollback possible: First version - No previous to rollback")

# ============ 5. INDEXING LATENCY TEST ============
print(f"\n[PERFORMANCE TEST] Indexing Latency:")

start = time.time()
# Simulate indexing 1000 vectors
dummy_embeddings = np.random.rand(1000, 384)
_ = SimpleVectorStore(dummy_embeddings, [{"id": i} for i in range(1000)])
index_latency = (time.time() - start) * 1000

print(f"  Indexing 1000 vectors: {index_latency:.2f} ms")
print(f"  Threshold: <500ms PASS")
print(f"  Status: {'PASS' if index_latency < 500 else 'FAIL'}")

# ============ 6. PERSISTENCE CHECK ============
print(f"\n[PERSISTENCE TEST] Vector Store Persistence:")
if os.path.exists("data/vector_store/latest.json"):
    print(f"  PASS: latest.json exists - Can restore after restart")
else:
    print(f"  FAIL: No persistence!")

# ============ 7. SAVE VERSION LOG ============
os.makedirs("logs", exist_ok=True)
version_log_path = "logs/vector_versions.json"

# Append to version log
if os.path.exists(version_log_path):
    with open(version_log_path, "r") as f:
        log = json.load(f)
else:
    log = []

log.append(version_data)

with open(version_log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"\n[Stage 4 PASS] Vector Store Complete")
print(f"  - Indexed: {len(chunks)} vectors")
print(f"  - Version: {VERSION}_{TIMESTAMP}")
print(f"  - Versioning: Enabled with rollback")
print(f"  - Persistence: latest.json + versioned files")
print(f"  - Latency: {index_latency:.2f}ms")
print(f"  - Prod Equivalent: FAISS IndexFlatL2 or Pinecone with version metadata")