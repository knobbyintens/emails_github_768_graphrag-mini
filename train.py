"""Training entry point for the GraphRAG-Mini retriever."""

import argparse
import json

import torch
import yaml

from src.graph_index import build_entity_graph
from src.retriever import GraphRetriever


class ToyEncoder(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, texts):
        base = torch.randn(len(texts), 64)
        return self.proj(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    with open(cfg["corpus"], "r", encoding="utf-8") as fh:
        corpus = json.load(fh)

    encoder = ToyEncoder().cuda()
    graph = build_entity_graph(corpus["passages"], corpus["entities"], encoder)
    retriever = GraphRetriever(graph, encoder, cfg["top_passages"])
    torch.save({"graph": graph, "retriever": retriever.state_dict()}, "runs/graph.idx")
    print("graph nodes:", graph.num_nodes)


if __name__ == "__main__":
    main()
