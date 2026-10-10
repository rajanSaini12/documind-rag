# DocuMind evaluation

- Generated: 2026-10-10T09:05:32+00:00
- Dataset: `qa_dataset.jsonl` (12 questions, 3 unanswerable)
- Documents: `handbook.pdf`
- Answer LLM: fake extractive LLM
- Judge: lexical
- top_k=4, chunk_overlap = chunk_size * 0.125

> Offline run with hashing embeddings, a fake extractive LLM and lexical metrics. Illustrative only: it shows the pipeline works, not real model quality.


| embedding         |   chunk_size |   hit_rate@k |   mrr |   relevance |   faithfulness |   false_refusal_rate |   correct_refusal_rate |   n_questions |
|:------------------|-------------:|-------------:|------:|------------:|---------------:|---------------------:|-----------------------:|--------------:|
| fake:hashing-256  |          256 |        0.889 | 0.889 |       0.662 |          1.000 |                0.222 |                  1.000 |            12 |
| fake:hashing-256  |          512 |        1.000 | 0.917 |       0.774 |          1.000 |                0.111 |                  1.000 |            12 |
| fake:hashing-256  |         1024 |        1.000 | 0.926 |       0.774 |          1.000 |                0.111 |                  1.000 |            12 |
| fake:hashing-1024 |          256 |        1.000 | 0.944 |       0.662 |          1.000 |                0.222 |                  1.000 |            12 |
| fake:hashing-1024 |          512 |        1.000 | 1.000 |       0.774 |          1.000 |                0.111 |                  1.000 |            12 |
| fake:hashing-1024 |         1024 |        1.000 | 1.000 |       0.774 |          1.000 |                0.111 |                  1.000 |            12 |
