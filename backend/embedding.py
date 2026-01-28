import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

# # ---- Paths ----
CHUNKS_PATH = Path("data/chunks/rbi_chunks.json")
EMBED_PATH  = Path("embeddings/rbi_embeddings.npy")
META_PATH   = Path("embeddings/rbi_metadata.json")
INDEX_PATH = Path("embeddings/rbi_faiss.index")


EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Load model ----
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---- Load chunks ----
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

texts = [c["text"] for c in chunks]

# ---- Generate embeddings ----
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # IMPORTANT for cosine similarity
)

embeddings = np.array(embeddings)

# ---- Save ----
np.save(EMBED_PATH, embeddings)
META_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

print(f"Saved {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings.shape}")


#--------faiss index-------------------
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim) # IP because embeddings are normalized
index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index built with {index.ntotal} vectors")


