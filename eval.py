"""Evaluation for GraphRAG-Mini retrieval quality."""

import argparse
import json

import torch
import yaml


def recall_at_k(retrieved, gold, k=8):
    picked = retrieved[:k]
    hits = len(set(picked) & set(gold))
    return hits / max(len(gold), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    bundle = torch.load(args.index, map_location="cpu")
    graph = bundle["graph"]
    with open(cfg["eval_set"], "r", encoding="utf-8") as fh:
        cases = json.load(fh)

    scores = []
    for case in cases:
        neigh = graph.neighbours(case["anchors"], hops=1)
        retrieved = [graph.nodes[n] for n in neigh]
        scores.append(recall_at_k(retrieved, case["gold_nodes"]))
    print(json.dumps({"recall@8": sum(scores) / len(scores)}, indent=2))


if __name__ == "__main__":
    main()
