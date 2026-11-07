"""Graph neighbourhood retrieval for GraphRAG-Mini."""

import torch
import torch.nn.functional as F

from .graph_index import EntityGraph


class GraphRetriever:
    def __init__(self, graph: EntityGraph, passage_encoder, top_passages: int = 8):
        self.graph = graph
        self.passage_encoder = passage_encoder
        self.top_passages = top_passages

    def _anchor_nodes(self, question, entities, top_nodes: int = 4):
        ids = [self.graph.nodes.index(e) for e in entities if e in self.graph.nodes]
        if not ids:
            return []
        q = self.passage_encoder([question]).detach()
        scores = F.normalize(self.graph.embeddings, dim=-1) @ F.normalize(q, dim=-1).T
        scores = scores.squeeze(-1)
        if ids:
            scores[ids] += 0.5
        return torch.topk(scores, k=min(top_nodes, scores.numel())).indices.tolist()

    def retrieve(self, question, entities, passages, passage_entities):
        anchors = self._anchor_nodes(question, entities)
        neighbourhood = self.graph.neighbours(anchors, hops=1)

        ranked = []
        for idx, (passage, names) in enumerate(zip(passages, passage_entities)):
            overlap = len(set(names) & {self.graph.nodes[n] for n in neighbourhood})
            ranked.append((overlap, idx, passage))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [p for _, _, p in ranked[: self.top_passages]]
