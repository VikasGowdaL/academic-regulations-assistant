"""Evaluate the Academic Regulations Assistant on a verified JSONL question set.

Run from backend/: python -m app.evaluation.evaluate --dataset app/evaluation/questions.jsonl
Requires COHERE_API_KEY and a prebuilt FAISS/Chroma index.
"""
import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from app.rag.chain import generate_answer
from app.rag.multi_query_retriever import retrieve_for_multiple_queries

load_dotenv()
REFUSAL = 'I could not find sufficient information'


def judge(llm, question, expected, answer, contexts):
    prompt = f'''You are evaluating a university-document RAG answer. Evaluate only from the given material.
Return ONLY a JSON object with integer scores correctness and faithfulness, each 0 or 1, and brief reasons.
Correctness=1 only if the answer accurately covers the essential facts in the verified reference answer.
Faithfulness=1 only if every substantive factual claim in the generated answer is supported by retrieved context.
If context is insufficient, faithfulness must be 0 for unsupported substantive claims.
QUESTION:\n{question}\nVERIFIED REFERENCE ANSWER:\n{expected}\nGENERATED ANSWER:\n{answer}\nRETRIEVED CONTEXT:\n{contexts[:16000]}'''
    raw = llm.invoke(prompt).content
    match = re.search(r'\{.*\}', raw, re.S)
    if not match:
        raise ValueError(f'Judge did not return JSON: {raw[:250]}')
    result = json.loads(match.group())
    for field in ('correctness', 'faithfulness'):
        if result.get(field) not in (0, 1):
            raise ValueError(f'Invalid {field} from judge')
    return result


def validate(row, line):
    for field in ('id', 'question', 'out_of_scope', 'reference_answer', 'relevant_sources'):
        if field not in row:
            raise ValueError(f'Line {line}: missing {field}')
    if not isinstance(row['out_of_scope'], bool) or not isinstance(row['relevant_sources'], list):
        raise ValueError(f'Line {line}: invalid out_of_scope or relevant_sources')
    if not row['out_of_scope'] and (not row['reference_answer'] or not row['relevant_sources']):
        raise ValueError(f'Line {line}: in-scope question needs reference_answer and relevant_sources')
    for source in row['relevant_sources']:
        if not isinstance(source, dict) or not source.get('source') or not isinstance(source.get('page'), int):
            raise ValueError(f'Line {line}: source must have source and integer page')


def run(args):
    rows = []
    for i, line in enumerate(Path(args.dataset).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            validate(row, i)
            rows.append(row)
    if len(rows) != 50:
        raise ValueError(f'Expected exactly 50 verified questions, found {len(rows)}')
    llm = ChatCohere(model='command-a-03-2025', temperature=0, cohere_api_key=os.getenv('COHERE_API_KEY'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        question = row['question']
        try:
            # The same retrieval path and threshold used by the application.
            docs = retrieve_for_multiple_queries(question, store_type=args.store, k=args.k)
            retrieved = {(d.metadata.get('source'), d.metadata.get('page_number')) for d in docs}
            expected = {(s['source'], s['page']) for s in row['relevant_sources']}
            recall = len(retrieved & expected) / len(expected) if expected else None
            response = generate_answer(question, store_type=args.store, k=args.k, session_id='eval_' + str(row['id']))
            answer = response['answer']
            refused = REFUSAL.lower() in answer.lower()
            contexts = '\n\n'.join(f"Source: {d.metadata.get('source')} Page: {d.metadata.get('page_number')}\n{d.page_content}" for d in docs)
            scores = None if row['out_of_scope'] or refused else judge(llm, question, row['reference_answer'], answer, contexts)
            result = {'id': row['id'], 'question': question, 'out_of_scope': row['out_of_scope'],
                      'retrieval_recall_at_k': recall, 'refused': refused,
                      'refusal_correct': refused == row['out_of_scope'],
                      'correctness': (0 if refused and not row['out_of_scope'] else (scores['correctness'] if scores else None)),
                      'faithfulness': scores['faithfulness'] if scores else None,
                      'judge_details': scores, 'answer': answer,
                      'retrieved_sources': [{'source': s, 'page': p} for s,p in sorted(retrieved)]}
        except Exception as exc:
            result = {'id': row['id'], 'error': str(exc)}
        results.append(result)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"{len(results)}/{len(rows)} {row['id']} {'ERROR' if 'error' in result else 'OK'}", flush=True)
    valid = [r for r in results if 'error' not in r]
    def average(field):
        values = [r[field] for r in valid if r.get(field) is not None]
        return round(sum(values) / len(values), 4) if values else None
    summary = {'questions_total': len(rows), 'questions_completed': len(valid), 'errors': len(rows)-len(valid),
               'store': args.store, 'k_per_planned_query': args.k,
               'recall_at_k_mean_in_scope': average('retrieval_recall_at_k'),
               'judge_correctness_mean_in_scope_including_refusals_as_zero': average('correctness'),
               'judge_faithfulness_mean_answered_in_scope': average('faithfulness'),
               'refusal_decision_accuracy_all_questions': average('refusal_correct'),
               'out_of_scope_refusal_accuracy': (round(sum(r['refused'] for r in valid if r['out_of_scope']) / sum(r['out_of_scope'] for r in valid), 4) if any(r['out_of_scope'] for r in valid) else None),
               'note': 'LLM-as-judge scores require human spot-checking. Retrieval measured at source-page level, not chunk-level. Retrieval and generation are invoked separately and may yield different chunks if query planning varies.'}
    output.with_name(output.stem + '_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='app/evaluation/questions.jsonl')
    p.add_argument('--store', choices=['faiss', 'chroma'], default='faiss')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--output', default='app/evaluation/results.json')
    run(p.parse_args())
