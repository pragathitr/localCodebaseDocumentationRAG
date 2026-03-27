"""
Query the RAG system with metadata support
"""
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Load index and mappings
print("Loading index...")
index = faiss.read_index('data/index/faiss.index')

with open('data/index/child_ids.json', 'r') as f:
    child_ids = json.load(f)

with open('data/index/child_to_parent.json', 'r') as f:
    child_to_parent = json.load(f)

with open('data/index/child_metadata.json', 'r') as f:  # ← NEW
    child_metadata = json.load(f)

with open('data/index/parents.json', 'r') as f:
    parents = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Ready\n")


def query(question, top_k=5):
    # Embed question
    query_embedding = model.encode([question])

    # Search children
    distances, indices = index.search(query_embedding.astype('float32'), top_k)

    # Get matched children with their metadata
    matched_children = []
    for idx, distance in zip(indices[0], distances[0]):
        child_id = child_ids[idx]
        metadata = child_metadata.get(child_id, {})

        matched_children.append({
            'child_id': child_id,
            'distance': float(distance),
            'metadata': metadata
        })

    # Get unique parent chunks
    parent_ids = set()
    sources = []  # ← NEW: Track sources for citations

    for child_info in matched_children:
        child_id = child_info['child_id']
        parent_id = child_to_parent[child_id]

        if parent_id not in parent_ids:
            parent_ids.add(parent_id)
            parent = parents[parent_id]
            metadata = child_info['metadata']

            sources.append({
                'section': metadata.get('section_title', 'Unknown'),
                'url': metadata.get('source_url', ''),
                'has_code': metadata.get('has_code', False)
            })

    # Retrieve parent texts
    context = []
    for parent_id in parent_ids:
        parent = parents[parent_id]
        context.append(f"[{parent['title']}]\n{parent['text']}")

    context_str = "\n\n".join(context)

    # Generate answer with Ollama
    prompt = f"""Based on this context from FastAPI documentation:

{context_str}

Answer this question: {question}

Provide a clear, concise answer with code examples if relevant."""

    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama3.2:3b',
        'prompt': prompt,
        'stream': True
    }, stream=True)

   # answer = response.json()['response']
    # Print tokens as they arrive
    answer = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get('response', '')
            print(token, end='', flush=True)
            answer += token

    print("\n")

    return {
        'question': question,
        'answer': answer,
        'sources': sources,  # ← NEW: Return sources
        'matched_children': matched_children  # ← NEW: Debug info
    }


# Interactive loop
if __name__ == "__main__":
    print("FastAPI Documentation Assistant")
    print("=" * 60)

    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == 'quit':
            break

        print("\nSearching...")
        result = query(question)

        # Display answer
        print(f"\n{'=' * 60}")
        print(f"ANSWER:\n")
        print(result['answer'])
        print(f"\n{'=' * 60}")

        # Display sources
        print(f"\nSOURCES:")
        for i, source in enumerate(result['sources'], 1):
            code_indicator = " [Contains Code]" if source['has_code'] else ""
            print(f"{i}. {source['section']}{code_indicator}")
            print(f"   {source['url']}")

        print(f"{'=' * 60}")