"""Compact entity graph construction for GraphRAG-Mini."""

from collections import defaultdict

import torch
import torch.nn.functional as F


class EntityGraph:
    def __init__(self, nodes, edges, embeddings):
        self.nodes = nodes
        self.edges = edges
        self.embeddings = embeddings

    @property
    def num_nodes(self):
        return len(self.nodes)

    def neighbours(self, node_ids, hops=1):
        frontier = set(node_ids)
        seen = set(frontier)
        for _ in range(hops):
            layer = set()
            for node in frontier:
                layer.update(self.edges.get(node, ()))
            frontier = layer - seen
            seen.update(frontier)
        return sorted(seen)


def build_entity_graph(passages, entity_tokens, encoder, threshold=0.55, max_degree=16):
    """Build a co-occurrence graph and prune low-value edges."""
    nodes, embeddings = [], []
    edges = defaultdict(set)

    anchor_vecs = encoder(entity_tokens).detach()
    for passage in passages:
        names = passage["entities"]
        for name in names:
            if name not in nodes:
                nodes.append(name)
                embeddings.append(anchor_vecs[len(nodes) - 1])

    embeddings = torch.stack(embeddings)
    sims = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T
    sims.fill_diagonal_(0.0)

    for i in range(len(nodes)):
        top = torch.topk(sims[i], k=min(max_degree, len(nodes) - 1))
        for j, score in zip(top.indices.tolist(), top.values.tolist()):
            if score >= threshold:
                edges[i].add(j)
                edges[j].add(i)
    return EntityGraph(nodes, dict(edges), embeddings)
