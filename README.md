# GraphRAG-Mini

Official implementation of **GraphRAG-Mini: Compact Graph Retrieval-Augmented Generation for Low-Resource Domains** (EMNLP 2025).

GraphRAG-Mini builds a compact entity graph from a small passage collection, prunes
it with a coverage objective, and uses the graph neighbourhood as retrieval context
for a 7B generator. The whole pipeline runs with under 1GB of GPU memory.

## Layout

- `src/graph_index.py` — entity graph construction and pruning
- `src/retriever.py` — neighbourhood retrieval and re-ranking
- `train.py`, `eval.py` — training and evaluation scripts
- `configs/base.yaml`

## Reproduce

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
python eval.py --config configs/base.yaml --index runs/graph.idx
```
