"""
Evaluate the RAG pipeline using retrieval metrics and an LLM judge.

Run from project root:
    python src/evaluate.py
"""
import json
import os
import re
import requests

from query import query, child_to_parent, parents


def compute_keyword_recall(matched_children, expected_keywords):
    """Return (recall_float, n_found) — keywords checked against retrieved parent texts."""
    seen_parents = set()
    combined_text = ""
    for child in matched_children:
        parent_id = child_to_parent[child['child_id']]
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            combined_text += " " + parents[parent_id]['text'].lower()

    found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    recall = found / len(expected_keywords) if expected_keywords else 0.0
    return recall, found


def compute_hit_at_k(matched_children, expected_keywords, k=5):
    """Return 1 if any expected keyword appears in the top-K children's parent texts."""
    for child in matched_children[:k]:
        parent_id = child_to_parent[child['child_id']]
        parent_text = parents[parent_id]['text'].lower()
        for kw in expected_keywords:
            if kw.lower() in parent_text:
                return 1
    return 0


def llm_judge(question, answer):
    """Call Ollama (non-streaming) to score the answer. Returns dict or None on failure."""
    prompt = (
        "You are an evaluation assistant. Rate the following answer on these criteria.\n"
        "Respond ONLY with valid JSON.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Rate each on a scale of 1-5:\n"
        "- relevance: Does the answer address the question?\n"
        "- completeness: Does it cover the topic adequately?\n"
        "- hallucination_free: Does it avoid inventing information? "
        "(5=no hallucination, 1=major hallucination)\n\n"
        'JSON format: {"relevance": int, "completeness": int, '
        '"hallucination_free": int, "reasoning": str}'
    )
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'llama3.2:3b', 'prompt': prompt, 'stream': False},
            timeout=120
        )
        raw = response.json().get('response', '').strip()
        # Strip optional ```json ... ``` fences
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            raw = match.group(1).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [Judge error: {e}]")
        return None


def evaluate():
    with open('evaluation_questions.json', 'r') as f:
        questions = json.load(f)

    results = []

    for item in questions:
        q_id = item['id']
        question = item['question']
        expected_keywords = item['expected_keywords']

        print(f"\nQ{q_id}: \"{question}\"")
        print("  Querying RAG system...")

        result = query(question)
        matched_children = result['matched_children']
        answer = result['answer']

        # Retrieval metrics
        recall, found = compute_keyword_recall(matched_children, expected_keywords)
        hit_k = compute_hit_at_k(matched_children, expected_keywords, k=5)
        avg_distance = (
            sum(c['distance'] for c in matched_children) / len(matched_children)
            if matched_children else 0.0
        )
        code_coverage = any(c['metadata'].get('has_code', False) for c in matched_children)

        # LLM judge
        print("  Running LLM judge...")
        scores = llm_judge(question, answer)

        q_result = {
            'id': q_id,
            'question': question,
            'expected_keywords': expected_keywords,
            'answer': answer,
            'sources': result['sources'],
            'retrieval_metrics': {
                'keyword_recall': round(recall, 4),
                'keywords_found': found,
                'keywords_total': len(expected_keywords),
                'hit_at_5': hit_k,
                'avg_distance': round(avg_distance, 4),
                'code_coverage': code_coverage,
            },
            'llm_judge': scores,
        }
        results.append(q_result)

        # Per-question summary
        print(f"  Keyword Recall:     {recall:.2f}  ({found}/{len(expected_keywords)} found)")
        print(f"  Hit@5:              {hit_k}")
        print(f"  Avg Distance:       {avg_distance:.4f}")
        print(f"  Code Coverage:      {code_coverage}")
        if scores:
            print(f"  Relevance:          {scores.get('relevance', 'N/A')}/5")
            print(f"  Completeness:       {scores.get('completeness', 'N/A')}/5")
            print(f"  Hallucination-free: {scores.get('hallucination_free', 'N/A')}/5")
        else:
            print("  LLM Judge:          failed to parse")

    # Aggregates
    n = len(results)
    mean_recall = sum(r['retrieval_metrics']['keyword_recall'] for r in results) / n
    hit_rate = sum(r['retrieval_metrics']['hit_at_5'] for r in results) / n * 100
    mean_distance = sum(r['retrieval_metrics']['avg_distance'] for r in results) / n

    judge_results = [r['llm_judge'] for r in results if r['llm_judge']]
    nj = len(judge_results)
    mean_relevance = sum(s.get('relevance', 0) for s in judge_results) / nj if nj else None
    mean_completeness = sum(s.get('completeness', 0) for s in judge_results) / nj if nj else None
    mean_hallucination_free = sum(s.get('hallucination_free', 0) for s in judge_results) / nj if nj else None

    print(f"\n{'=' * 60}")
    print(f"=== AGGREGATE ({n} questions) ===")
    print(f"  Mean Keyword Recall:     {mean_recall:.2f}")
    print(f"  Hit@5 Rate:              {hit_rate:.0f}%")
    print(f"  Mean Avg Distance:       {mean_distance:.4f}")
    if mean_relevance is not None:
        print(f"  Mean Relevance:          {mean_relevance:.1f}/5")
        print(f"  Mean Completeness:       {mean_completeness:.1f}/5")
        print(f"  Mean Hallucination-free: {mean_hallucination_free:.1f}/5")
    else:
        print("  LLM Judge:               all failed")
    print(f"{'=' * 60}")

    # Save full results
    os.makedirs('data', exist_ok=True)
    output = {
        'results': results,
        'aggregate': {
            'n_questions': n,
            'mean_keyword_recall': round(mean_recall, 4),
            'hit_at_5_rate': round(hit_rate / 100, 4),
            'mean_avg_distance': round(mean_distance, 4),
            'mean_relevance': round(mean_relevance, 2) if mean_relevance is not None else None,
            'mean_completeness': round(mean_completeness, 2) if mean_completeness is not None else None,
            'mean_hallucination_free': round(mean_hallucination_free, 2) if mean_hallucination_free is not None else None,
            'n_judge_successes': nj,
        }
    }
    with open('data/evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to data/evaluation_results.json")


if __name__ == "__main__":
    evaluate()
