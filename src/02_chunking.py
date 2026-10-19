"""
Stage 2: Chunking + Data Quality Testing
Location: src/02_chunking.py
Concepts: Chunk Boundary, Overlap, Metadata Loss, Context Loss, Token Limit
Interview Q: "How do you prevent context loss in chunking?"
"""

import os
import json

# ============ 1. LOAD SAFE BRD ============
BRD_SAFE_PATH = "data/BRD_safe.txt"

with open(BRD_SAFE_PATH, "r") as f:
    brd_text = f.read()

print(f"[Stage 2] BRD Safe Loaded: {len(brd_text)} chars")
print(f"Preview: {brd_text[:150]}...")

# ============ 2. NAIVE CHUNKING (BAD) vs SMART CHUNKING (GOOD) ============
# This is Data Quality Testing

def naive_chunking(text, chunk_size=150):
    """BAD: Breaks Step 4 in middle - Context Loss!"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append({
            "id": f"naive_{i}",
            "text": chunk,
            "method": "naive"
        })
    return chunks

def smart_chunking(text):
    """GOOD: Splits by Step - Preserves context!"""
    chunks = []
    # Split by "Step" keyword - This is Step-aware splitter
    # Like RecursiveCharacterTextSplitter with separators=["Step"]
    steps = text.split("Step")
    
    for idx, step in enumerate(steps):
        if not step.strip():
            continue
        full_step = f"Step{step}".strip()
        chunks.append({
            "id": f"step_{idx}",
            "text": full_step,
            "metadata": {
                "step_number": idx,
                "step_title": full_step[:30],
                "token_count": len(full_step.split()),  # Rough token
                "has_aadhaar": "Aadhaar" in full_step,
                "has_refund": "Refund" in full_step
            },
            "method": "smart"
        })
    return chunks

# Test both
naive_chunks = naive_chunking(brd_text)
smart_chunks = smart_chunking(brd_text)

print(f"\n[DATA QUALITY TEST] Chunking Comparison:")
print(f"  Naive Chunking: {len(naive_chunks)} chunks (BAD - breaks context)")
print(f"    Example: {naive_chunks[1]['text'][:80]}... <- BROKEN!")
print(f"  Smart Chunking: {len(smart_chunks)} chunks (GOOD - Step wise)")
print(f"    Example: {smart_chunks[3]['text'][:80]}... <- COMPLETE!")

# ============ 3. OVERLAP TESTING ============
# Why overlap needed? To prevent boundary loss

def chunk_with_overlap(text, chunk_size=100, overlap=20):
    """With overlap - Prevents loss at boundary"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap
        if start >= len(text):
            break
    return chunks

overlap_chunks = chunk_with_overlap(brd_text, 150, 30)
print(f"\n[TEST] Overlap Chunking: {len(overlap_chunks)} chunks with 30 chars overlap")
print(f"  Purpose: If Step 4 ends at boundary, next chunk still has context")

# ============ 4. TOKEN LIMIT TESTING ============
# This is Performance + Data Validation

def check_token_limit(chunks, max_tokens=100):
    """Check if any chunk exceeds LLM token limit"""
    failed = []
    for c in chunks:
        tokens = c["metadata"]["token_count"]
        if tokens > max_tokens:
            failed.append((c["id"], tokens))
    
    if failed:
        print(f"\n[TOKEN LIMIT TEST] FAIL: {len(failed)} chunks exceed {max_tokens} tokens")
        for fid, tok in failed:
            print(f"    {fid}: {tok} tokens")
    else:
        print(f"\n[TOKEN LIMIT TEST] PASS: All {len(chunks)} chunks under {max_tokens} tokens")
    
    return failed

token_failures = check_token_limit(smart_chunks, max_tokens=50)

# ============ 5. METADATA LOSS CHECK ============
# Most important for interview!

print(f"\n[METADATA TEST] Checking metadata preservation:")
for c in smart_chunks:
    has_meta = "metadata" in c and "step_number" in c["metadata"]
    print(f"  {c['id']}: Metadata={'PASS' if has_meta else 'FAIL - Loss!'} -> {c['metadata']}")

# ============ 6. SAVE FOR NEXT STAGE ============
os.makedirs("data", exist_ok=True)
with open("data/chunks.json", "w") as f:
    json.dump(smart_chunks, f, indent=2)

print(f"\n[Stage 2 PASS] Chunking Complete")
print(f"  - Method: Smart Step-aware (not naive)")
print(f"  - Total Chunks: {len(smart_chunks)}")
print(f"  - Overlap: 30 chars tested")
print(f"  - Token Check: {'FAIL' if token_failures else 'PASS'}")
print(f"  - Metadata: Preserved with step_number, has_aadhaar flag")
print(f"  - Saved: data/chunks.json")
print(f"  - Context Loss: FIXED by Step separator")