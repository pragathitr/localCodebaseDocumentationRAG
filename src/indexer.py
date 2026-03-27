"""
Builds FAISS index from children chunks
Stores mapping from child -> parent + metadata
"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os


def build_index(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract all children
    all_children = []
    child_to_parent = {}
    child_metadata = {}  # ← NEW: Store metadata for each child
    parents = {}

    for chunk in chunks:
        parent = chunk['parent']
        parents[parent['id']] = parent

        for child in chunk['children']:
            all_children.append(child['text'])
            child_to_parent[child['id']] = parent['id']
            child_metadata[child['id']] = child.get('metadata', {})  # ← NEW

    print(f"Embedding {len(all_children)} children...")
    child_ids = [c['id'] for chunk in chunks for c in chunk['children']]
    embeddings = model.encode(all_children, show_progress_bar=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    # Save everything
    os.makedirs('data/index', exist_ok=True)
    faiss.write_index(index, 'data/index/faiss.index')

    with open('data/index/child_ids.json', 'w') as f:
        json.dump(child_ids, f)

    with open('data/index/child_to_parent.json', 'w') as f:
        json.dump(child_to_parent, f)

    with open('data/index/child_metadata.json', 'w') as f:  # ← NEW
        json.dump(child_metadata, f, indent=2)

    with open('data/index/parents.json', 'w') as f:
        json.dump(parents, f)

    # Stats with metadata
    total_with_code = sum(
        1 for meta in child_metadata.values()
        if meta.get('has_code', False)
    )

    print(f"\n✓ Indexed {len(all_children)} children")
    print(f"✓ {total_with_code} children contain code blocks")
    print(f"✓ Index saved to data/index/")


if __name__ == "__main__":
    with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    build_index(chunks)