from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_temporal_report(
    temporal_outputs: Dict[str, torch.Tensor],
    fusion_outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Diagnostics that say whether the M7 branch actually reaches the logits.

    ``temporal_contribution`` is ||W_T z_T|| / ||h_M0|| inside the classifier. A
    branch that stays at its identity initialisation keeps this at exactly zero,
    which is the failure mode this repository has hit before (a module that was
    inert at its stated weight), so it is logged every epoch.
    """
    report: Dict[str, torch.Tensor] = {}
    if not temporal_outputs:
        return report
    for key in ("temporal_feature_abs_mean",):
        if key in temporal_outputs:
            report[key] = temporal_outputs[key]
    for key in ("temporal_contribution", "temporal_head_weight_norm"):
        if key in fusion_outputs:
            report[key] = fusion_outputs[key]
    return report


def sample_event_masked_nodes(
    num_nodes: int,
    root_indices: torch.Tensor,
    graph_batch: torch.Tensor,
    num_graphs: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-event mask of ``ceil(ratio * replies)`` non-source nodes.

    Deliberately the same rule ``GlobalSemanticGAE._sample_batched_mask`` uses for M2':
    the source post is never masked, each event's count is rounded up, and events with a
    single node are skipped. Sharing the rule is what makes the M2a / M2' comparison
    isolate the reconstruction objective rather than a difference in masking.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if ratio <= 0.0 or num_nodes <= 1:
        return mask
    roots = root_indices.to(device=device, dtype=torch.long).reshape(-1)
    owners = graph_batch.to(device=device, dtype=torch.long)
    for graph_index in range(num_graphs):
        graph_nodes = torch.nonzero(owners == graph_index, as_tuple=False).flatten()
        if graph_nodes.numel() <= 1:
            continue
        if graph_index < roots.numel():
            candidates = graph_nodes[graph_nodes != roots[graph_index]]
        else:
            candidates = graph_nodes
        if candidates.numel() == 0:
            continue
        count = max(1, min(int(math.ceil(float(candidates.numel()) * ratio)), int(candidates.numel())))
        chosen = candidates[torch.randperm(candidates.numel(), device=device)[:count]]
        mask[chosen] = True
    return mask


EDGE_FAMILY_NAMES: Dict[str, int] = {"real": 0, "2hop": 1, "semantic": 2}


def parse_edge_families(spec: str) -> Tuple[str, ...]:
    """Parse a --gat-edge-families request such as ``real``, ``real,2hop`` or ``all``.

    Family 0 is the real reply-to-reply propagation graph, family 1 the exact two-hop
    virtual edges and family 2 the event-local semantic Top-k edges. The source-conditioned
    global event node is a *separate* mechanism -- it integrates pooled node states rather
    than edges -- so dropping families never disconnects it.
    """
    text = str(spec or "").strip().lower().replace(" ", "")
    if text in {"", "all", "v0"}:
        return ("real", "2hop", "semantic")
    aliases = {"1hop": "real", "onehop": "real", "twohop": "2hop", "knn": "semantic"}
    names: list[str] = []
    for piece in text.split(","):
        name = aliases.get(piece, piece)
        if name not in EDGE_FAMILY_NAMES:
            raise ValueError(
                f"Unknown edge family {piece!r}; expected any of "
                f"{sorted(EDGE_FAMILY_NAMES)} (or 'all')."
            )
        if name not in names:
            names.append(name)
    if "real" not in names:
        raise ValueError(
            "The real propagation family cannot be dropped: 'real' is the actual reply "
            "tree, and without it the structure branch is no longer a propagation "
            f"encoder. Got {spec!r}."
        )
    return tuple(sorted(names, key=lambda item: EDGE_FAMILY_NAMES[item]))


def disabled_edge_families_for(spec: str) -> Tuple[int, ...]:
    """Indices to disable for a family request, e.g. ``real,2hop`` -> ``(2,)``."""
    keep = {EDGE_FAMILY_NAMES[name] for name in parse_edge_families(spec)}
    return tuple(sorted(set(EDGE_FAMILY_NAMES.values()) - keep))


def graph_mean_pool(node_states: torch.Tensor, graph_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if node_states.numel() == 0:
        return node_states.new_zeros((num_graphs, 0))
    graph_batch = graph_batch.to(device=node_states.device, dtype=torch.long)
    pooled = node_states.new_zeros((num_graphs, node_states.size(-1)))
    pooled.index_add_(0, graph_batch, node_states)
    counts = torch.bincount(graph_batch, minlength=num_graphs).to(
        device=node_states.device,
        dtype=node_states.dtype,
    )
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)


def graph_max_pool(node_states: torch.Tensor, graph_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if node_states.numel() == 0:
        return node_states.new_zeros((num_graphs, node_states.size(-1)))
    graph_batch = graph_batch.to(device=node_states.device, dtype=torch.long)
    index = graph_batch.unsqueeze(-1).expand(-1, node_states.size(-1))
    pooled = torch.full(
        (num_graphs, node_states.size(-1)),
        -torch.inf,
        device=node_states.device,
        dtype=node_states.dtype,
    )
    try:
        pooled.scatter_reduce_(0, index, node_states, reduce="amax", include_self=True)
    except (AttributeError, RuntimeError):
        for graph_index in range(num_graphs):
            mask = graph_batch == graph_index
            if bool(mask.any().item()):
                pooled[graph_index] = node_states[mask].amax(dim=0)
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def event_uniformity_loss(representations: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Encourage graph representations to spread over the unit hypersphere."""
    if temperature <= 0.0:
        raise ValueError("uniformity temperature must be positive.")
    if representations.ndim != 2:
        raise ValueError("uniformity representations must have shape [batch, dim].")
    if representations.size(0) < 2:
        return representations.sum() * 0.0

    normalized = F.normalize(representations, p=2, dim=-1)
    squared_distances = torch.pdist(normalized, p=2).pow(2)
    potentials = torch.exp(-temperature * squared_distances)
    return torch.log(potentials.mean().clamp_min(torch.finfo(potentials.dtype).tiny))

class DenseGraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        num_edge_relations: int,
        use_edge_relations: bool = True,
        use_semantic_edge_attention: bool = True,
        edge_dropout: float = 0.0,
        alpha: float = 0.2,
        attention_type: str = "gat",
    ) -> None:
        super().__init__()
        if not 0.0 <= edge_dropout < 1.0:
            raise ValueError("edge_dropout must be in [0, 1).")
        attention_type = attention_type.strip().lower().replace("_", "")
        if attention_type not in {"gat", "gatv2"}:
            raise ValueError("attention_type must be 'gat' or 'gatv2'.")
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.attention_type = attention_type
        self.use_edge_relations = use_edge_relations
        self.use_semantic_edge_attention = use_semantic_edge_attention
        self.edge_dropout = edge_dropout
        self.linear = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        if self.attention_type == "gat":
            self.attn_src = nn.Parameter(torch.empty(num_heads, out_dim))
            self.attn_dst = nn.Parameter(torch.empty(num_heads, out_dim))
            self.linear_dst = None
            self.attn_dynamic = None
        else:
            self.register_parameter("attn_src", None)
            self.register_parameter("attn_dst", None)
            self.linear_dst = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.attn_dynamic = nn.Parameter(torch.empty(num_heads, out_dim))
        self.rel_bias = nn.Embedding(num_edge_relations, num_heads)
        self.semantic_edge_mlp = (
            nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, 1, bias=False),
            )
            if use_semantic_edge_attention
            else None
        )
        self.bias = nn.Parameter(torch.zeros(num_heads * out_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.attention_type == "gat":
            nn.init.xavier_uniform_(self.attn_src)
            nn.init.xavier_uniform_(self.attn_dst)
        else:
            nn.init.xavier_uniform_(self.linear_dst.weight)
            nn.init.xavier_uniform_(self.attn_dynamic)
        nn.init.xavier_uniform_(self.rel_bias.weight)
        if self.semantic_edge_mlp is not None:
            nn.init.xavier_uniform_(self.semantic_edge_mlp[0].weight)
            nn.init.zeros_(self.semantic_edge_mlp[0].bias)
            # Start from the original GAT exactly; semantic edge evidence is
            # introduced only after the final layer receives gradients.
            nn.init.zeros_(self.semantic_edge_mlp[2].weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def edge_softmax_by_destination(
        edge_logits: torch.Tensor,
        dst_nodes: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if edge_logits.numel() == 0:
            return torch.zeros_like(edge_logits)

        num_heads = edge_logits.size(1)
        try:
            index = dst_nodes.view(-1, 1).expand(-1, num_heads)
            max_per_dst = torch.full(
                (num_nodes, num_heads),
                -float("inf"),
                dtype=edge_logits.dtype,
                device=edge_logits.device,
            )
            max_per_dst.scatter_reduce_(0, index, edge_logits, reduce="amax", include_self=True)
            stabilized = edge_logits - max_per_dst[dst_nodes]
            exp_scores = torch.exp(stabilized.clamp(min=-60.0, max=60.0))
            denom = torch.zeros(
                (num_nodes, num_heads),
                dtype=edge_logits.dtype,
                device=edge_logits.device,
            )
            denom.index_add_(0, dst_nodes, exp_scores)
            return exp_scores / denom[dst_nodes].clamp_min(1e-12)
        except (AttributeError, RuntimeError):
            attention = torch.zeros_like(edge_logits)
            for node in range(num_nodes):
                mask = dst_nodes == node
                if bool(mask.any().item()):
                    attention[mask] = torch.softmax(edge_logits[mask], dim=0)
            return attention

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = node_features.size(0)
        device = node_features.device
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        if num_nodes == 0:
            return node_features.new_zeros((0, self.num_heads * self.out_dim))

        dropped_features = self.dropout(node_features)
        projected = self.linear(dropped_features).view(num_nodes, self.num_heads, self.out_dim)
        if self.attention_type == "gat":
            destination_projected = projected
            src_logits = (projected * self.attn_src.unsqueeze(0)).sum(dim=-1)
            dst_logits = (projected * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        else:
            destination_projected = self.linear_dst(dropped_features).view(
                num_nodes,
                self.num_heads,
                self.out_dim,
            )

        if edge_index.numel() > 0:
            raw_src = edge_index[0].long()
            raw_dst = edge_index[1].long()
            valid_edges = (
                (raw_src >= 0)
                & (raw_src < num_nodes)
                & (raw_dst >= 0)
                & (raw_dst < num_nodes)
                & (raw_src != raw_dst)
            )
            edge_src = raw_src[valid_edges]
            edge_dst = raw_dst[valid_edges]
            if edge_type.numel() == raw_src.numel():
                edge_rel = edge_type.to(device=device, dtype=torch.long)[valid_edges]
            elif edge_type.numel() >= edge_src.numel():
                edge_rel = edge_type.to(device=device, dtype=torch.long)[: edge_src.numel()]
            else:
                edge_rel = torch.zeros(edge_src.size(0), dtype=torch.long, device=device)
            if self.training and self.edge_dropout > 0.0 and edge_src.numel() > 0:
                keep_edges = torch.rand(edge_src.size(0), device=device) >= self.edge_dropout
                edge_src = edge_src[keep_edges]
                edge_dst = edge_dst[keep_edges]
                edge_rel = edge_rel[keep_edges]
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)
            edge_rel = torch.empty(0, dtype=torch.long, device=device)

        loop_nodes = torch.arange(num_nodes, device=device)
        src_nodes = torch.cat([edge_src, loop_nodes], dim=0)
        dst_nodes = torch.cat([edge_dst, loop_nodes], dim=0)
        loop_rel = torch.zeros(num_nodes, dtype=torch.long, device=device)
        edge_rel = torch.cat([edge_rel, loop_rel], dim=0)
        edge_rel = edge_rel.clamp(min=0, max=self.rel_bias.num_embeddings - 1)

        source_states = projected[src_nodes]
        destination_states = destination_projected[dst_nodes]
        if self.attention_type == "gat":
            edge_logits = dst_logits[dst_nodes] + src_logits[src_nodes]
        else:
            dynamic_states = self.leaky_relu(source_states + destination_states)
            edge_logits = (
                dynamic_states * self.attn_dynamic.unsqueeze(0)
            ).sum(dim=-1)
        relation_bias = self.rel_bias(edge_rel) if self.use_edge_relations else 0.0
        semantic_bias: torch.Tensor | float = 0.0
        if self.semantic_edge_mlp is not None:
            edge_change = torch.cat(
                [
                    (source_states - destination_states).abs(),
                    source_states * destination_states,
                ],
                dim=-1,
            )
            semantic_bias = self.semantic_edge_mlp(edge_change).squeeze(-1)
            non_self_loop = (src_nodes != dst_nodes).to(dtype=semantic_bias.dtype)
            semantic_bias = semantic_bias * non_self_loop.unsqueeze(-1)

        if self.attention_type == "gat":
            logits = self.leaky_relu(edge_logits + relation_bias + semantic_bias)
        else:
            logits = edge_logits + relation_bias + semantic_bias
        attention = self.edge_softmax_by_destination(logits, dst_nodes, num_nodes)
        attention = self.dropout(attention)

        updated_heads = torch.zeros_like(projected)
        weighted_messages = attention.unsqueeze(-1) * source_states
        updated_heads.index_add_(0, dst_nodes, weighted_messages)

        updated = updated_heads.reshape(num_nodes, -1) + self.bias
        return self.activation(updated)


class RootConditionedMultiScaleAttentionLayer(nn.Module):
    """Attention over an externally constructed multi-scale, non-self neighborhood.

    The residual update remains the only self-information path.  Queries receive
    the event root state, while keys and values preserve the neighbor state.  This
    intentionally does not reuse TD/BU types, relation biases, or the old
    semantic-change bias: the v1 intervention changes *which* nodes are visible.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        use_relative_time_bias: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.query = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.root_query = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.key = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.value = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim * num_heads))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.use_relative_time_bias = bool(use_relative_time_bias)
        if self.use_relative_time_bias:
            self.time_bias_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.GELU(),
                nn.Linear(16, num_heads),
            )
            # sigmoid(-2.9444) ~= 0.05: time begins as a small correction.
            self.time_gate_logits = nn.Parameter(torch.full((num_heads,), -2.9444))
        else:
            self.time_bias_mlp = None
            self.register_parameter("time_gate_logits", None)
        self.last_mean_abs_time_bias_semantic: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.query, self.root_query, self.key, self.value):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(self.bias)
        if self.time_bias_mlp is not None:
            nn.init.xavier_uniform_(self.time_bias_mlp[0].weight)
            nn.init.zeros_(self.time_bias_mlp[0].bias)
            nn.init.xavier_uniform_(self.time_bias_mlp[2].weight)
            nn.init.zeros_(self.time_bias_mlp[2].bias)

    @staticmethod
    def encode_relative_time(delta_minutes: torch.Tensor) -> torch.Tensor:
        """Signed, compressed time difference sign(dt) * log(1 + abs(dt))."""
        return torch.sign(delta_minutes) * torch.log1p(delta_minutes.abs())

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        node_time: Optional[torch.Tensor] = None,
        edge_source_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = node_features.size(0)
        self.last_mean_abs_time_bias_semantic = node_features.new_zeros(())
        if num_nodes == 0:
            return node_features.new_zeros((0, self.num_heads * self.out_dim))
        graph_batch = graph_batch.to(device=node_features.device, dtype=torch.long)
        roots = root_indices.reshape(-1).to(device=node_features.device, dtype=torch.long)
        if graph_batch.numel() != num_nodes or roots.numel() == 0:
            raise ValueError("Multi-scale GAT requires aligned graph assignments and roots.")
        if bool(((roots < 0) | (roots >= num_nodes)).any().item()):
            raise ValueError("Multi-scale GAT received an out-of-range root.")
        expected_graphs = torch.arange(roots.numel(), device=node_features.device)
        if not torch.equal(graph_batch[roots], expected_graphs):
            raise ValueError("Each multi-scale GAT root must belong to its graph.")

        dropped = self.dropout(node_features)
        root_state = dropped[roots][graph_batch]
        query = (self.query(dropped) + self.root_query(root_state)).view(
            num_nodes, self.num_heads, self.out_dim
        )
        key = self.key(dropped).view(num_nodes, self.num_heads, self.out_dim)
        value = self.value(dropped).view(num_nodes, self.num_heads, self.out_dim)

        if edge_index.numel() == 0:
            return self.activation(value.new_zeros((num_nodes, self.num_heads * self.out_dim)) + self.bias)
        source = edge_index[0].to(device=node_features.device, dtype=torch.long)
        destination = edge_index[1].to(device=node_features.device, dtype=torch.long)
        valid = (
            (source >= 0)
            & (source < num_nodes)
            & (destination >= 0)
            & (destination < num_nodes)
            & (source != destination)
            & (graph_batch[source] == graph_batch[destination])
        )
        source, destination = source[valid], destination[valid]
        if edge_source_type is not None:
            edge_source_type = edge_source_type.to(
                device=node_features.device, dtype=torch.long
            ).reshape(-1)
            if edge_source_type.numel() != edge_index.size(1):
                raise ValueError("edge_source_type must contain one value per edge.")
            edge_source_type = edge_source_type[valid]
        if source.numel() == 0:
            return self.activation(value.new_zeros((num_nodes, self.num_heads * self.out_dim)) + self.bias)
        logits = (query[destination] * key[source]).sum(dim=-1) / math.sqrt(self.out_dim)
        if self.use_relative_time_bias:
            if node_time is None or edge_source_type is None:
                raise ValueError(
                    "Relative-time attention requires aligned node_time and edge_source_type."
                )
            node_time = node_time.to(device=node_features.device, dtype=logits.dtype).reshape(-1)
            if node_time.numel() != num_nodes:
                raise ValueError("node_time must contain one value per node.")
            semantic_mask = edge_source_type == 2
            if bool(semantic_mask.any().item()):
                delta = node_time[source] - node_time[destination]
                encoded_delta = self.encode_relative_time(delta).unsqueeze(-1)
                raw_time_bias = self.time_bias_mlp(encoded_delta)
                gated_time_bias = torch.sigmoid(self.time_gate_logits) * raw_time_bias
                logits = logits + gated_time_bias * semantic_mask.unsqueeze(-1).to(logits.dtype)
                self.last_mean_abs_time_bias_semantic = (
                    gated_time_bias[semantic_mask].abs().mean().detach()
                )
        attention = DenseGraphAttentionLayer.edge_softmax_by_destination(logits, destination, num_nodes)
        attention = self.dropout(attention)
        messages = torch.zeros_like(value)
        messages.index_add_(0, destination, attention.unsqueeze(-1) * value[source])
        return self.activation(messages.reshape(num_nodes, -1) + self.bias)


class SourceConditionedGlobalEventNode(nn.Module):
    """One-way event collector queried by the source/root representation.

    It summarizes the nodes after the first MS-GAT layer.  The resulting event
    vector is read out by the graph encoder; it is never sent back into node
    states, so the global collector cannot overwrite the local propagation view.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.root_query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.root_query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(projection.weight)

    def forward(
        self,
        node_states: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        node_count = int(node_states.size(0))
        roots = root_indices.reshape(-1).to(device=node_states.device, dtype=torch.long)
        graph_batch = graph_batch.to(device=node_states.device, dtype=torch.long)
        if roots.numel() != num_graphs or graph_batch.numel() != node_count:
            raise ValueError("Global event node requires one root and assignment per graph.")
        if bool(((roots < 0) | (roots >= node_count)).any().item()):
            raise ValueError("Global event node received an invalid root index.")
        if not torch.equal(graph_batch[roots], torch.arange(num_graphs, device=node_states.device)):
            raise ValueError("Global event roots must belong to their respective graphs.")

        query = self.root_query(node_states[roots])
        keys = self.key(node_states)
        values = self.value(node_states)
        logits = (query[graph_batch] * keys).sum(dim=-1) / math.sqrt(node_states.size(-1))
        pooled = []
        for graph_index in range(num_graphs):
            indices = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                pooled.append(node_states.new_zeros(node_states.size(-1)))
                continue
            attention = logits[indices].softmax(dim=0)
            pooled.append((attention.unsqueeze(-1) * values[indices]).sum(dim=0))
        return self.output(torch.stack(pooled, dim=0))


class TemporalConsistencyHead(nn.Module):
    """Predict the completed-event representation from an early propagation stage.

    ``future_loss = 1 - cos(P(z_early), sg(z_full))``.  The predictor ``P`` maps
    the early representation into the full-event space and the target carries a
    stop-gradient, so the objective cannot be satisfied by collapsing ``P`` (the
    target is not a function of ``P``).  Nothing here is used at inference.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, early_repr: Tensor, full_repr: Tensor) -> Tensor:
        prediction = self.predictor(early_repr)
        target = full_repr.detach()
        cosine = F.cosine_similarity(prediction, target, dim=-1)
        return (1.0 - cosine).mean()


class NodeRepresentationFusion(nn.Module):
    """Combine independent and source-conditioned node views into one node vector.

    ``independent`` is the reply/source embedding produced without seeing the
    claim; ``relation`` is the pair encoding ``[CLS] source [SEP] reply [SEP]``.
    Modes:

    * ``independent``  — ``h = c`` (plain V1G representation, no fusion module)
    * ``relation``     — ``h = r`` (source-conditioned nodes alone)
    * ``gate``         — ``h = g ⊙ c + (1 - g) ⊙ r``, ``g = σ(W_g [c ; r])``
    * ``residual-norm``— ``h = LayerNorm(W_c c + W_r r)``

    The gate starts at 0.5 (bias 0) rather than independent-dominant: in the
    earlier relation-branch experiment a gate initialised near 1.0 never
    opened, which would silently turn this ablation back into the baseline.
    """

    MODES = ("independent", "relation", "gate", "residual-norm")

    def __init__(self, dim: int, mode: str, dropout: float = 0.2) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"node representation mode must be one of {self.MODES}.")
        self.mode = mode
        if mode == "gate":
            self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)
        elif mode == "residual-norm":
            self.independent_projection = nn.Linear(dim, dim)
            self.relation_projection = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

    def forward(self, independent: Tensor, relation: Tensor) -> Tensor:
        if self.mode == "independent":
            return independent
        if self.mode == "relation":
            return relation
        if self.mode == "gate":
            gate = self.gate(torch.cat([independent, relation], dim=-1))
            return gate * independent + (1.0 - gate) * relation
        fused = self.independent_projection(independent) + self.relation_projection(relation)
        return self.norm(self.dropout(fused))


class SourceConditionedEvidencePooling(nn.Module):
    """Second global view: source-as-query attention over replies only.

    ``h_E = softmax(q_s K^T / sqrt(d)) V`` where the query comes from the
    source state and keys/values come from reply states (i = 1..N).  The
    source is never used as a value, so the module answers "which replies are
    evidence for this source?" instead of "what does the whole event look
    like?".  Events without replies produce a zero evidence vector.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(projection.weight)

    def forward(
        self,
        node_states: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        node_count = int(node_states.size(0))
        roots = root_indices.reshape(-1).to(device=node_states.device, dtype=torch.long)
        graph_batch = graph_batch.to(device=node_states.device, dtype=torch.long)
        if roots.numel() != num_graphs or graph_batch.numel() != node_count:
            raise ValueError("Evidence pooling requires one root and assignment per graph.")
        if bool(((roots < 0) | (roots >= node_count)).any().item()):
            raise ValueError("Evidence pooling received an invalid root index.")
        if not torch.equal(graph_batch[roots], torch.arange(num_graphs, device=node_states.device)):
            raise ValueError("Evidence pooling roots must belong to their respective graphs.")

        query = self.query(node_states[roots])
        keys = self.key(node_states)
        values = self.value(node_states)
        logits = (query[graph_batch] * keys).sum(dim=-1) / math.sqrt(node_states.size(-1))
        pooled = []
        for graph_index in range(num_graphs):
            replies = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            replies = replies[replies != roots[graph_index]]
            if replies.numel() == 0:
                pooled.append(node_states.new_zeros(node_states.size(-1)))
                continue
            attention = logits[replies].softmax(dim=0)
            pooled.append((attention.unsqueeze(-1) * values[replies]).sum(dim=0))
        return torch.stack(pooled, dim=0)


class EvidenceReadout(nn.Module):
    """Keep the root separate and attend only over real comments in each graph."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.score = nn.Linear(dim, 1, bias=False)
        self.output = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                    nn.LayerNorm(dim), nn.Dropout(dropout))

    def forward(self, nodes, roots, graph_batch=None, num_graphs=1):
        roots = roots.reshape(-1).to(device=nodes.device, dtype=torch.long)
        single = graph_batch is None
        if single:
            graph_batch = torch.zeros(len(nodes), dtype=torch.long, device=nodes.device)
        else:
            graph_batch = graph_batch.to(device=nodes.device, dtype=torch.long)
        if roots.numel() != num_graphs or graph_batch.numel() != len(nodes):
            raise ValueError("Evidence readout requires one root per graph and one assignment per node")
        if bool(((roots < 0) | (roots >= len(nodes))).any()):
            raise ValueError("Invalid root index")
        if not torch.equal(graph_batch[roots], torch.arange(num_graphs, device=nodes.device)):
            raise ValueError("Root does not belong to its graph")
        pooled, weights = [], torch.zeros(len(nodes), device=nodes.device, dtype=nodes.dtype)
        for g in range(num_graphs):
            indices = torch.nonzero(graph_batch == g, as_tuple=False).flatten()
            indices = indices[indices != roots[g]]
            root = nodes[roots[g]]
            if indices.numel():
                comments = nodes[indices]
                logits = self.score(torch.tanh(self.key(comments) + self.query(root))).squeeze(-1)
                alpha = logits.softmax(0)
                weights = weights.index_copy(0, indices, alpha)
                summary = (alpha.unsqueeze(-1) * comments).sum(0)
            else:
                summary = torch.zeros_like(root)
            pooled.append(self.output(torch.cat([root, summary], dim=-1)))
        result = torch.stack(pooled)
        return (result[0] if single else result), weights


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_edge_relations: int,
        use_edge_relations: bool = True,
        use_semantic_edge_attention: bool = True,
        edge_dropout: float = 0.0,
        readout: str = "mean",
        attention_type: str = "gat",
        residual_mode: str = "fixed",
        residual_alpha: float = 1.0,
        neighborhood: str = "local",
        semantic_k: int = 3,
        global_event_node: bool = False,
        source_evidence_pooling: bool = False,
        depth_encoding: bool = False,
        relative_time_bias: bool = False,
        topology: str = "tree",
        adaptive_channels: str | tuple[str, ...] | None = None,
        sibling_sets: Optional[nn.Module] = None,
        sibling_event: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("graph hidden_dim must be divisible by num_heads.")
        readout = readout.strip().lower().replace("_", "-")
        if readout not in {"mean", "propagation-change", "evidence"}:
            raise ValueError("GAT readout must be 'mean', 'propagation-change', or 'evidence'.")
        attention_type = attention_type.strip().lower().replace("_", "")
        if attention_type not in {"gat", "gatv2"}:
            raise ValueError("GAT attention_type must be 'gat' or 'gatv2'.")
        residual_mode = residual_mode.strip().lower().replace("_", "-")
        if residual_mode not in {"fixed", "learnable"}:
            raise ValueError("GAT residual_mode must be 'fixed' or 'learnable'.")
        if not 0.0 < residual_alpha <= 1.0:
            raise ValueError("GAT residual_alpha must be in (0, 1].")
        neighborhood = neighborhood.strip().lower().replace("_", "-")
        if neighborhood not in {"local", "ms-gat-v1"}:
            raise ValueError("GAT neighborhood must be 'local' or 'ms-gat-v1'.")
        if semantic_k <= 0:
            raise ValueError("GAT semantic_k must be positive.")
        topology = topology.strip().lower().replace("_", "-")
        if topology not in {"tree", "star", "set"}:
            raise ValueError("GAT topology must be 'tree', 'star', or 'set'.")

        head_dim = hidden_dim // num_heads
        self.readout = readout
        if readout == "evidence":
            self.evidence_readout = EvidenceReadout(hidden_dim, dropout)
        self.attention_type = attention_type
        self.residual_mode = residual_mode
        self.residual_alpha = float(residual_alpha)
        self.neighborhood = neighborhood
        self.semantic_k = int(semantic_k)
        self.topology = topology
        self.global_event_node_enabled = bool(global_event_node)
        self.source_evidence_pooling_enabled = bool(source_evidence_pooling)
        # Diagnostic-only edge-family ablation (0=one-hop, 1=two-hop, 2=semantic-KNN).
        # Empty by default, so training and inference are unaffected.
        self.disabled_edge_families: tuple[int, ...] = ()
        self.depth_encoding_enabled = bool(depth_encoding)
        self.relative_time_bias_enabled = bool(relative_time_bias)
        if self.global_event_node_enabled and self.neighborhood != "ms-gat-v1":
            raise ValueError("global_event_node requires neighborhood='ms-gat-v1'.")
        if self.source_evidence_pooling_enabled and not self.global_event_node_enabled:
            raise ValueError(
                "source_evidence_pooling extends the global event node and requires global_event_node=True."
            )
        if self.depth_encoding_enabled and self.neighborhood != "ms-gat-v1":
            raise ValueError("depth_encoding requires neighborhood='ms-gat-v1'.")
        if self.relative_time_bias_enabled and self.neighborhood != "ms-gat-v1":
            raise ValueError("relative_time_bias requires neighborhood='ms-gat-v1'.")

        def _build_layer() -> nn.Module:
            if self.neighborhood == "ms-gat-v1":
                return RootConditionedMultiScaleAttentionLayer(
                    hidden_dim,
                    head_dim,
                    num_heads,
                    dropout,
                    use_relative_time_bias=self.relative_time_bias_enabled,
                )
            return DenseGraphAttentionLayer(
                hidden_dim,
                head_dim,
                num_heads,
                dropout,
                num_edge_relations,
                use_edge_relations=use_edge_relations,
                use_semantic_edge_attention=use_semantic_edge_attention,
                edge_dropout=edge_dropout,
                attention_type=self.attention_type,
            )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.depth_encoding_enabled:
            # Five stable source-distance buckets: 0, 1, 2, 3, and >=4.
            self.depth_embedding = nn.Embedding(5, hidden_dim)
            nn.init.normal_(self.depth_embedding.weight, mean=0.0, std=0.02)
        else:
            self.depth_embedding = None
        # ACM mode replaces the attention stack with adaptive channel mixing
        # (identity / low-pass / high-pass / exact two-hop), so no GAT layer is
        # instantiated at all.
        # The sibling-set interaction runs on the *projected* node features
        # (hidden_dim), which is the H of the design diagram: the graph branch
        # gets H + gamma_sib * delta while GA-GRU keeps the untouched H.
        self.sibling_sets = sibling_sets
        # Event-level sibling vector (concatenative fusion).  Only one of the two
        # sibling pathways is installed at a time.
        self.sibling_event = sibling_event
        self.adaptive_channels: tuple[str, ...] | None = (
            None if adaptive_channels is None else parse_acm_channels(adaptive_channels)
        )
        if self.adaptive_channels is not None:
            if self.neighborhood != "ms-gat-v1":
                raise ValueError(
                    "adaptive_channels requires neighborhood='ms-gat-v1'; it reads the "
                    "one-hop / exact two-hop / semantic edge families."
                )
            if self.relative_time_bias_enabled:
                raise ValueError(
                    "adaptive_channels has no attention logits, so gat_relative_time_bias "
                    "cannot be combined with it."
                )
            self.layers = nn.ModuleList(
                [
                    AdaptiveChannelLayer(hidden_dim, dropout, self.adaptive_channels),
                    AdaptiveChannelLayer(hidden_dim, dropout, self.adaptive_channels),
                ]
            )
            self.acm_diagnostics = ACMDiagnostics(self.adaptive_channels, len(self.layers))
            if "semantic" not in self.adaptive_channels:
                # The semantic-KNN family is still built by the MS-GAT edge builder;
                # drop it so unused edges cost no aggregation work.
                self.disabled_edge_families = tuple(
                    sorted(set(self.disabled_edge_families) | {2})
                )
        else:
            self.acm_diagnostics = None
            self.layers = nn.ModuleList([_build_layer(), _build_layer()])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.layers))])
        if self.global_event_node_enabled:
            self.global_event_node = SourceConditionedGlobalEventNode(hidden_dim)
            self.global_event_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.global_event_gate = nn.Parameter(torch.tensor(0.1))
            if self.source_evidence_pooling_enabled:
                # Dual global view: h_G' = h_G + MLP([h_G || h_E]).  The
                # zero-initialized final layer keeps the model exactly equal to
                # the plain V1G global event node at initialization, so the
                # validated h_G pathway is never destroyed while h_E grows in.
                self.source_evidence_pooling = SourceConditionedEvidencePooling(hidden_dim)
                self.evidence_fusion = nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(self.evidence_fusion[-1].weight)
                nn.init.zeros_(self.evidence_fusion[-1].bias)
            else:
                self.source_evidence_pooling = None
                self.evidence_fusion = None
        else:
            self.global_event_node = None
            self.global_event_projection = None
            self.source_evidence_pooling = None
            self.evidence_fusion = None
            self.register_parameter("global_event_gate", None)
        if self.residual_mode == "learnable":
            initial_alpha = min(max(self.residual_alpha, 1e-4), 1.0 - 1e-4)
            initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
            self.residual_logits = nn.Parameter(
                torch.full((len(self.layers),), initial_logit)
            )
        else:
            self.register_parameter("residual_logits", None)
        self.dropout = nn.Dropout(dropout)
        if self.readout == "propagation-change":
            self.change_readout = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

    def _run_stack(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        layers: nn.ModuleList,
        norms: nn.ModuleList,
        root_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        num_graphs: int = 1,
        node_time: Optional[torch.Tensor] = None,
        edge_source_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden = node_features
        event_state: Optional[torch.Tensor] = None
        evidence_state: Optional[torch.Tensor] = None
        for layer_index, (layer, norm) in enumerate(zip(layers, norms)):
            if self.neighborhood == "ms-gat-v1":
                if root_index is None or graph_batch is None:
                    raise ValueError("MS-GAT requires root indices and graph assignments.")
                updated = layer(
                    hidden,
                    edge_index,
                    root_index,
                    graph_batch,
                    node_time=node_time,
                    edge_source_type=edge_source_type,
                )
            else:
                updated = layer(hidden, edge_index, edge_type)
            updated = self.dropout(updated)
            if self.residual_mode == "learnable":
                alpha = torch.sigmoid(self.residual_logits[layer_index]).to(
                    dtype=hidden.dtype
                )
                hidden = norm((1.0 - alpha) * hidden + alpha * updated)
            else:
                hidden = norm(hidden + self.residual_alpha * updated)
            if (
                layer_index == 0
                and self.global_event_node_enabled
                and self.global_event_node is not None
                and root_index is not None
                and graph_batch is not None
            ):
                event_state = self.global_event_node(
                    hidden,
                    root_index,
                    graph_batch,
                    num_graphs,
                )
                if self.source_evidence_pooling is not None:
                    evidence_state = self.source_evidence_pooling(
                        hidden,
                        root_index,
                        graph_batch,
                        num_graphs,
                    )
                    event_state = event_state + self.evidence_fusion(
                        torch.cat([event_state, evidence_state], dim=-1)
                    )
        return hidden, event_state, evidence_state

    @staticmethod
    def _tree_depths(
        edge_index_td: torch.Tensor,
        num_nodes: int,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute source distance on the original TD tree before virtual edges exist."""
        children: list[list[int]] = [[] for _ in range(num_nodes)]
        owners = graph_batch.detach().cpu().tolist()
        if edge_index_td.numel():
            sources = edge_index_td[0].detach().cpu().tolist()
            destinations = edge_index_td[1].detach().cpu().tolist()
            for source, destination in zip(sources, destinations):
                source, destination = int(source), int(destination)
                if (
                    source == destination
                    or source < 0
                    or destination < 0
                    or source >= num_nodes
                    or destination >= num_nodes
                    or owners[source] != owners[destination]
                ):
                    continue
                children[source].append(destination)

        depths = [-1] * num_nodes
        for root in root_indices.reshape(-1).detach().cpu().tolist():
            root = int(root)
            if root < 0 or root >= num_nodes:
                continue
            depths[root] = 0
            queue = [root]
            cursor = 0
            while cursor < len(queue):
                parent = queue[cursor]
                cursor += 1
                for child in children[parent]:
                    candidate_depth = depths[parent] + 1
                    if depths[child] < 0 or candidate_depth < depths[child]:
                        depths[child] = candidate_depth
                        queue.append(child)

        # Valid conversation trees are connected. Treat malformed or isolated nodes
        # as source-level nodes rather than inventing a virtual propagation distance.
        depths = [0 if depth < 0 else min(depth, 4) for depth in depths]
        return torch.tensor(depths, dtype=torch.long, device=device)

    def _build_ms_gat_edges(
        self,
        node_features: torch.Tensor,
        one_hop_edges: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
        return_source_types: bool = False,
    ) -> tuple:
        """Construct the fixed multi-scale neighborhood for one forward pass.

        The returned directed edges contain the undirected one-hop propagation graph,
        exact two-hop virtual edges, and event-local semantic Top-k edges. The topology
        is detached from autograd and remains unchanged between training and evaluation.
        """
        device = node_features.device
        num_nodes = int(node_features.size(0))
        graph_batch = graph_batch.to(device=device, dtype=torch.long)
        if graph_batch.numel() != num_nodes:
            raise ValueError("graph_batch must contain one assignment per node.")
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")

        one_by_graph: list[set[tuple[int, int]]] = [set() for _ in range(num_graphs)]
        if one_hop_edges.numel():
            source = one_hop_edges[0].to(device=device, dtype=torch.long)
            destination = one_hop_edges[1].to(device=device, dtype=torch.long)
            valid = (
                (source >= 0)
                & (source < num_nodes)
                & (destination >= 0)
                & (destination < num_nodes)
                & (source != destination)
                & (graph_batch[source] == graph_batch[destination])
            )
            source_cpu = source[valid].detach().cpu().tolist()
            destination_cpu = destination[valid].detach().cpu().tolist()
            graph_cpu = graph_batch.detach().cpu().tolist()
            for src, dst in zip(source_cpu, destination_cpu):
                graph_index = int(graph_cpu[src])
                if 0 <= graph_index < num_graphs:
                    one_by_graph[graph_index].add((int(src), int(dst)))

        normalized = F.normalize(node_features.detach(), dim=-1)
        result_edges: list[tuple[int, int]] = []
        result_edge_source_types: list[int] = []
        edge_counts = torch.zeros((num_graphs, 3), dtype=torch.long, device=device)
        node_graph = graph_batch.detach().cpu().tolist()

        for graph_index in range(num_graphs):
            nodes = [node for node, owner in enumerate(node_graph) if owner == graph_index]
            one = one_by_graph[graph_index]
            adjacency: dict[int, set[int]] = {node: set() for node in nodes}
            for src, dst in one:
                adjacency[dst].add(src)

            two: set[tuple[int, int]] = set()
            if self.topology == "tree":
                for destination in nodes:
                    for middle in adjacency[destination]:
                        for source in adjacency.get(middle, ()):
                            pair = (source, destination)
                            if source != destination and pair not in one:
                                two.add(pair)

            semantic: set[tuple[int, int]] = set()
            blocked_by_destination: dict[int, set[int]] = {node: set() for node in nodes}
            for source, destination in one | two:
                blocked_by_destination[destination].add(source)
            for destination in nodes:
                candidates = [
                    source for source in nodes
                    if source != destination
                    and source not in blocked_by_destination[destination]
                ]
                if not candidates:
                    continue
                candidate_index = torch.tensor(candidates, dtype=torch.long, device=device)
                scores = normalized[candidate_index] @ normalized[destination]
                keep = min(self.semantic_k, len(candidates))
                chosen = torch.topk(scores, k=keep, largest=True, sorted=True).indices
                semantic.update((candidates[int(index.item())], destination) for index in chosen)

            result_edges.extend(one)
            result_edge_source_types.extend([0] * len(one))
            result_edges.extend(two)
            result_edge_source_types.extend([1] * len(two))
            result_edges.extend(semantic)
            result_edge_source_types.extend([2] * len(semantic))
            edge_counts[graph_index] = torch.tensor(
                [len(one), len(two), len(semantic)], dtype=torch.long, device=device
            )

        if result_edges:
            edge_index = torch.tensor(result_edges, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_source_type = torch.tensor(
            result_edge_source_types, dtype=torch.long, device=device
        )
        if return_source_types:
            return edge_index, edge_counts, edge_source_type
        return edge_index, edge_counts

    def _pool_propagation_change(
        self,
        initial: torch.Tensor,
        hidden: torch.Tensor,
        graph_batch: Optional[torch.Tensor],
        num_graphs: int,
    ) -> torch.Tensor:
        delta = hidden - initial
        if graph_batch is None:
            if delta.size(0) == 0:
                statistics = delta.new_zeros(delta.size(-1) * 3)
            else:
                mean = delta.mean(dim=0)
                variance = (delta - mean).square().mean(dim=0)
                safe_std = torch.sqrt(variance.clamp_min(1e-12))
                std = torch.where(variance > 0.0, safe_std, torch.zeros_like(safe_std))
                max_abs = delta.abs().amax(dim=0)
                statistics = torch.cat([mean, std, max_abs], dim=-1)
            return self.change_readout(statistics)

        graph_batch = graph_batch.to(device=delta.device, dtype=torch.long)
        mean = graph_mean_pool(delta, graph_batch, num_graphs)
        centered = delta - mean[graph_batch]
        variance = graph_mean_pool(centered.square(), graph_batch, num_graphs)
        safe_std = torch.sqrt(variance.clamp_min(1e-12))
        std = torch.where(variance > 0.0, safe_std, torch.zeros_like(safe_std))
        max_abs = graph_max_pool(delta.abs(), graph_batch, num_graphs)
        statistics = torch.cat([mean, std, max_abs], dim=-1)
        return self.change_readout(statistics)

    def set_disabled_edge_families(self, families: tuple[int, ...] = ()) -> None:
        """Diagnostic switch: drop whole MS-GAT edge families before message passing."""
        self.disabled_edge_families = tuple(int(family) for family in families)

    def set_acm_gate_mode(self, mode) -> None:
        """Diagnostic switch on ACM channel gates (no-op for GAT encoders)."""
        for layer in self.layers:
            if hasattr(layer, "set_gate_mode"):
                layer.set_gate_mode(mode)

    @staticmethod
    def _undirected_edges(
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one UNDIRECTED edge set for the GAT — propagation direction is ignored.

        Every conversation edge is added in both directions and gets the SAME structural
        relation type for both directions (root / internal / terminal role only), so the GAT
        treats the propagation graph as an undirected graph and does not distinguish top-down
        from bottom-up. The direction-specific bottom-up edge types are intentionally unused.
        """
        del edge_index_bu, edge_type_bu  # direction ignored; reverse edges reuse the TD role types
        if edge_index_td.numel() == 0:
            return edge_index_td, edge_type_td
        undirected_edges = torch.cat([edge_index_td, edge_index_td.flip(0)], dim=1)
        undirected_types = torch.cat([edge_type_td, edge_type_td], dim=0)
        return undirected_edges, undirected_types

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
        root_index: torch.Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        node_feature_mask: Optional[torch.Tensor] = None,
        mask_token: Optional[torch.Tensor] = None,
        node_time: Optional[torch.Tensor] = None,
        keep_edge_families: Optional[tuple[int, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """``keep_edge_families`` restricts the MS-GAT families for this call only.

        Used by the masked-reconstruction pass so it can be limited to the real
        propagation tree: a virtual semantic-KNN edge is built from node features,
        so using it to reconstruct a masked node would let the neighbourhood
        itself carry information about that node.  ``None`` keeps every family,
        which is what classification always uses.
        """
        if node_feature_mask is not None:
            if mask_token is None:
                raise ValueError("mask_token is required when node_feature_mask is provided.")
            node_feature_mask = node_feature_mask.to(device=node_features.device, dtype=torch.bool)
            if node_feature_mask.numel() != node_features.size(0):
                raise ValueError("node_feature_mask must have one entry per graph node.")
            if bool(node_feature_mask.any().item()):
                node_features = node_features.clone()
                node_features[node_feature_mask] = mask_token.to(
                    device=node_features.device,
                    dtype=node_features.dtype,
                )

        initial = self.input_proj(node_features)
        edge_index, edge_type = self._undirected_edges(
            edge_index_td,
            edge_index_bu,
            edge_type_td,
            edge_type_bu,
        )
        if graph_batch is None:
            resolved_num_graphs = 1
            resolved_graph_batch = torch.zeros(
                initial.size(0), dtype=torch.long, device=initial.device
            )
        else:
            resolved_graph_batch = graph_batch.to(device=initial.device, dtype=torch.long)
            if num_graphs is None:
                num_graphs = (
                    int(resolved_graph_batch.max().detach().cpu().item()) + 1
                    if resolved_graph_batch.numel()
                    else 0
                )
            resolved_num_graphs = int(num_graphs)

        sibling_relative_norm: Optional[torch.Tensor] = None
        if self.sibling_sets is not None:
            sibling_delta = self.sibling_sets(initial, edge_index_td, resolved_graph_batch)
            scaled_delta = self.sibling_sets.gate() * sibling_delta
            # R_sib: how much of the node representation the sibling path actually
            # moves. A value near zero means the module cannot matter downstream.
            sibling_relative_norm = scaled_delta.norm(dim=-1).mean() / (
                initial.norm(dim=-1).mean() + 1e-8
            )
            initial = initial + scaled_delta

        sibling_event_repr: Optional[torch.Tensor] = None
        if self.sibling_event is not None:
            sibling_event_repr, _ = self.sibling_event(
                initial, edge_index_td, resolved_graph_batch, resolved_num_graphs
            )
            sibling_event_repr = self.sibling_event.gate() * sibling_event_repr

        if self.depth_encoding_enabled and self.depth_embedding is not None:
            node_depths = self._tree_depths(
                edge_index_td,
                int(initial.size(0)),
                root_index,
                resolved_graph_batch,
                initial.device,
            )
            initial = initial + self.depth_embedding(node_depths)
        else:
            node_depths = torch.zeros(initial.size(0), dtype=torch.long, device=initial.device)

        if self.neighborhood == "ms-gat-v1":
            edge_index, neighborhood_edge_counts, edge_source_type = self._build_ms_gat_edges(
                node_features,
                edge_index,
                resolved_graph_batch,
                resolved_num_graphs,
                return_source_types=True,
            )
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=initial.device)
            disabled_families = self.disabled_edge_families
            if keep_edge_families is not None:
                allowed = {int(family) for family in keep_edge_families}
                known = {int(family) for family in torch.unique(edge_source_type).tolist()}
                disabled_families = tuple(sorted(known - allowed))
            if disabled_families:
                keep = torch.ones(
                    edge_source_type.numel(), dtype=torch.bool, device=edge_source_type.device
                )
                for family in disabled_families:
                    keep &= edge_source_type != int(family)
                edge_index = edge_index[:, keep]
                edge_source_type = edge_source_type[keep]
                edge_type = edge_type[keep]
                # The counts are built before this filter; report what is actually used so
                # a disabled family reads exactly zero in the logs and diagnostics.
                for family in disabled_families:
                    if 0 <= int(family) < neighborhood_edge_counts.size(-1):
                        neighborhood_edge_counts[:, int(family)] = 0
        else:
            edge_source_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=initial.device)
            neighborhood_edge_counts = torch.zeros(
                (resolved_num_graphs, 3), dtype=torch.long, device=initial.device
            )

        hidden, event_state, evidence_state = self._run_stack(
            initial,
            edge_index,
            edge_type,
            self.layers,
            self.norms,
            root_index=root_index,
            graph_batch=resolved_graph_batch,
            num_graphs=resolved_num_graphs,
            node_time=node_time,
            edge_source_type=edge_source_type,
        )

        if self.readout == "evidence":
            pooled, _ = self.evidence_readout(hidden, root_index, graph_batch, resolved_num_graphs)
        elif self.readout == "propagation-change":
            pooled = self._pool_propagation_change(
                initial=initial,
                hidden=hidden,
                graph_batch=graph_batch,
                num_graphs=resolved_num_graphs,
            )
        elif graph_batch is None:
            pooled = hidden.mean(dim=0)
        else:
            pooled = graph_mean_pool(hidden, graph_batch, resolved_num_graphs)

        if event_state is not None and self.global_event_projection is not None:
            event_update = self.global_event_projection(event_state)
            if graph_batch is None:
                event_update = event_update.squeeze(0)
            gate = self.global_event_gate.to(dtype=pooled.dtype)
            pooled = pooled + gate * event_update

        result: Dict[str, torch.Tensor] = {
            "node_states": hidden,
            "pooled": pooled,
            "propagation_delta": hidden - initial,
            "neighborhood_edge_counts": neighborhood_edge_counts,
            "node_depths": node_depths,
        }
        if self.acm_diagnostics is not None:
            result.update(self.acm_diagnostics.summarize(self.layers))
        if self.sibling_sets is not None:
            result.update(self.sibling_sets.diagnostics())
        if self.sibling_event is not None:
            result.update(self.sibling_event.diagnostics())
            result["sibling_event_repr"] = sibling_event_repr
            result["sibling_event_norm"] = sibling_event_repr.norm(dim=-1).mean()
        if sibling_relative_norm is not None:
            result["sibling_relative_norm"] = sibling_relative_norm
        if event_state is not None:
            result["global_event_state"] = event_state
            result["global_event_gate"] = self.global_event_gate
        if evidence_state is not None:
            result["source_evidence_state"] = evidence_state
        if self.relative_time_bias_enabled:
            time_layers = [
                layer for layer in self.layers
                if isinstance(layer, RootConditionedMultiScaleAttentionLayer)
            ]
            time_gates = torch.stack(
                [torch.sigmoid(layer.time_gate_logits) for layer in time_layers], dim=0
            ).mean(dim=0)
            bias_values = [
                layer.last_mean_abs_time_bias_semantic
                for layer in time_layers
                if layer.last_mean_abs_time_bias_semantic is not None
            ]
            result["time_gate_per_head"] = time_gates
            result["time_gate_mean"] = time_gates.mean()
            result["mean_abs_time_bias_semantic"] = (
                torch.stack(bias_values).mean()
                if bias_values else initial.new_zeros(())
            )
        return result


class GlobalSemanticGAE(nn.Module):
    """Masked global-semantic prediction used by the GAT structure branch.

    One GAT encoder serves two roles in a single module:
      * the *unmasked* forward produces the graph-level structure vector used for
        classification with graph mean pooling, and
      * during training a *masked* forward feeds a GAT attention decoder that predicts
        deterministic unmasked GAT node states (self-distillation), raw node features,
        or parent-to-child semantic changes.

    This replaces the previous standalone "GAT branch" plus a separate reconstructor:
    encode, mask sampling, graph decode and masked-node reconstruction now live here.
    The default latent target is produced by the same online GAT with dropout and edge
    dropout disabled, then immediately detached. This keeps the target stable without
    adding a second trainable encoder. Raw input-feature reconstruction remains available
    as an ablation mode for backward compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_edge_relations: int,
        use_edge_relations: bool = True,
        use_semantic_edge_attention: bool = True,
        edge_dropout: float = 0.0,
        mask_ratio: float = 0.25,
        sce_gamma: float = 2.0,
        target_mode: str = "latent",
        variance_normalized_mse: bool = True,
        feature_std_floor: float = 0.05,
        readout: str = "mean",
        attention_type: str = "gat",
        residual_mode: str = "fixed",
        residual_alpha: float = 1.0,
        neighborhood: str = "local",
        semantic_k: int = 3,
        global_event_node: bool = False,
        source_evidence_pooling: bool = False,
        depth_encoding: bool = False,
        relative_time_bias: bool = False,
        topology: str = "tree",
        adaptive_channels: str | tuple[str, ...] | None = None,
        sibling_sets: Optional[nn.Module] = None,
        sibling_event: Optional[nn.Module] = None,
        reconstruction_real_edges_only: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        if target_mode not in {"latent", "raw"}:
            raise ValueError("target_mode must be 'latent' or 'raw'.")
        if feature_std_floor <= 0.0:
            raise ValueError("feature_std_floor must be positive.")
        self.mask_ratio = mask_ratio
        self.output_dim = hidden_dim  # encoder pooled / node-state dimension
        self.sce_gamma = sce_gamma  # kept only for backward-compatible configs
        self.target_mode = target_mode
        self.variance_normalized_mse = variance_normalized_mse
        self.feature_std_floor = feature_std_floor
        self.reconstruction_enabled = True
        # Family 0 is the undirected real propagation tree; families 1 and 2 are
        # the virtual two-hop and semantic-KNN edges.
        self.reconstruction_real_edges_only = bool(reconstruction_real_edges_only)
        self.register_buffer("reconstruction_feature_std", torch.ones(input_dim))

        # The same two-layer GAT encoder is shared by classification and reconstruction.
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_edge_relations=num_edge_relations,
            use_edge_relations=use_edge_relations,
            use_semantic_edge_attention=use_semantic_edge_attention,
            edge_dropout=edge_dropout,
            readout=readout,
            attention_type=attention_type,
            residual_mode=residual_mode,
            residual_alpha=residual_alpha,
            neighborhood=neighborhood,
            semantic_k=semantic_k,
            global_event_node=global_event_node,
            source_evidence_pooling=source_evidence_pooling,
            depth_encoding=depth_encoding,
            relative_time_bias=relative_time_bias,
            topology=topology,
            adaptive_channels=adaptive_channels,
            sibling_sets=sibling_sets,
            sibling_event=sibling_event,
        )

        # GARD global prediction head (training only): [MASK] -> GAT encoder -> GAT decoder -> MSE.
        self.mask_token = nn.Parameter(torch.zeros(input_dim))  # input-space [MASK]
        self.decoder = DenseGraphAttentionLayer(
            self.output_dim,
            self.output_dim // num_heads,
            num_heads,
            dropout,
            num_edge_relations,
            use_edge_relations=use_edge_relations,
            use_semantic_edge_attention=use_semantic_edge_attention,
            edge_dropout=0.0,
            attention_type=attention_type,
        )
        self.decoder_norm = nn.LayerNorm(self.output_dim)
        if self.target_mode == "latent":
            # The GAT decoder itself is the predictor. Avoid another MLP so the
            # auxiliary task directly supervises graph message passing.
            self.output_head = nn.Identity()
            self.prediction_dim = self.output_dim
        else:
            self.output_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.output_dim, input_dim),
            )
            self.prediction_dim = input_dim

    def set_reconstruction_enabled(self, enabled: bool) -> None:
        self.reconstruction_enabled = bool(enabled)

    @torch.no_grad()
    def set_reconstruction_feature_std(self, feature_std: torch.Tensor) -> None:
        if self.target_mode != "raw":
            raise RuntimeError("Feature standard deviations are used only with raw targets.")
        feature_std = torch.as_tensor(feature_std).flatten()
        if feature_std.shape != self.reconstruction_feature_std.shape:
            raise ValueError(
                "feature_std must have shape "
                f"{tuple(self.reconstruction_feature_std.shape)}, got {tuple(feature_std.shape)}."
            )
        if not bool(torch.isfinite(feature_std).all().item()):
            raise ValueError("feature_std must contain only finite values.")
        self.reconstruction_feature_std.copy_(
            feature_std.to(
                device=self.reconstruction_feature_std.device,
                dtype=self.reconstruction_feature_std.dtype,
            ).clamp_min(self.feature_std_floor)
        )

    @torch.no_grad()
    def initialize_reconstruction_output(self, feature_mean: torch.Tensor) -> None:
        if self.target_mode != "raw":
            raise RuntimeError("Feature-mean output initialization is used only with raw targets.")
        feature_mean = torch.as_tensor(feature_mean).flatten()
        output_layer = self.output_head[-1]
        if feature_mean.shape != output_layer.bias.shape:
            raise ValueError(
                f"feature_mean must have shape {tuple(output_layer.bias.shape)}, "
                f"got {tuple(feature_mean.shape)}."
            )
        if not bool(torch.isfinite(feature_mean).all().item()):
            raise ValueError("feature_mean must contain only finite values.")
        nn.init.zeros_(output_layer.weight)
        output_layer.bias.copy_(
            feature_mean.to(device=output_layer.bias.device, dtype=output_layer.bias.dtype)
        )

    def _sample_mask(
        self,
        num_nodes: int,
        root_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if not self.training or self.mask_ratio <= 0.0 or num_nodes <= 1:
            return mask
        candidate_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        roots = root_index.to(device=device, dtype=torch.long).reshape(-1)
        valid_roots = roots[(roots >= 0) & (roots < num_nodes)]
        candidate_mask[valid_roots] = False
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return mask
        mask_count = int(math.ceil(float(candidate_indices.numel()) * self.mask_ratio))
        mask_count = max(1, min(mask_count, int(candidate_indices.numel())))
        selected = candidate_indices[torch.randperm(candidate_indices.numel(), device=device)[:mask_count]]
        mask[selected] = True
        return mask

    def _sample_batched_mask(
        self,
        num_nodes: int,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if not self.training or self.mask_ratio <= 0.0 or num_nodes <= 1:
            return mask
        root_indices = root_indices.to(device=device, dtype=torch.long).reshape(-1)
        graph_batch = graph_batch.to(device=device, dtype=torch.long)
        for graph_index in range(num_graphs):
            graph_nodes = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
            if graph_nodes.numel() <= 1:
                continue
            if graph_index < root_indices.numel():
                candidate_indices = graph_nodes[graph_nodes != root_indices[graph_index]]
            else:
                candidate_indices = graph_nodes
            if candidate_indices.numel() == 0:
                continue
            mask_count = int(math.ceil(float(candidate_indices.numel()) * self.mask_ratio))
            mask_count = max(1, min(mask_count, int(candidate_indices.numel())))
            selected = candidate_indices[torch.randperm(candidate_indices.numel(), device=device)[:mask_count]]
            mask[selected] = True
        return mask

    @staticmethod
    def _undirected_edges(
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Undirected edge set for the re-mask decoder (direction ignored, symmetric types)."""
        del edge_index_bu, edge_type_bu  # direction ignored; reverse edges reuse the TD role types
        if edge_index_td.numel() == 0:
            return edge_index_td, edge_type_td
        undirected_edges = torch.cat([edge_index_td, edge_index_td.flip(0)], dim=1)
        undirected_types = torch.cat([edge_type_td, edge_type_td], dim=0)
        return undirected_edges, undirected_types

    def _reconstruct(
        self,
        masked_node_states: torch.Tensor,
        target_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = target_states.detach()
        decoded = self.decoder(masked_node_states, edge_index, edge_type)
        # Do not bypass the graph decoder with an encoder-state residual. The
        # masked-node prediction must be produced by attention message passing.
        decoded = self.decoder_norm(decoded)
        predictions = self.output_head(decoded)
        if self.target_mode == "raw" and self.variance_normalized_mse:
            feature_std = self.reconstruction_feature_std.to(
                device=predictions.device,
                dtype=predictions.dtype,
            ).clamp_min(self.feature_std_floor)
            normalized_error = (predictions[mask] - targets[mask]) / feature_std
            loss = normalized_error.square().mean()
        else:
            loss = F.mse_loss(predictions[mask], targets[mask])
        return loss, predictions, targets

    def _deterministic_latent_targets(
        self,
        node_features: torch.Tensor,
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
        root_index: torch.Tensor,
        graph_batch: Optional[torch.Tensor],
        num_graphs: Optional[int],
        node_time: Optional[torch.Tensor],
    ) -> torch.Tensor:
        encoder_was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                target_outputs = self.encoder(
                    node_features=node_features,
                    edge_index_td=edge_index_td,
                    edge_index_bu=edge_index_bu,
                    edge_type_td=edge_type_td,
                    edge_type_bu=edge_type_bu,
                    root_index=root_index,
                    graph_batch=graph_batch,
                    num_graphs=num_graphs,
                    node_time=node_time,
                )
        finally:
            self.encoder.train(encoder_was_training)
        return target_outputs["node_states"].detach()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
        root_index: torch.Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        node_time: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(
            node_features=node_features,
            edge_index_td=edge_index_td,
            edge_index_bu=edge_index_bu,
            edge_type_td=edge_type_td,
            edge_type_bu=edge_type_bu,
            root_index=root_index,
            graph_batch=graph_batch,
            num_graphs=num_graphs,
            node_time=node_time,
        )
        outputs: Dict[str, torch.Tensor] = {
            "pooled": encoded["pooled"],
            # Exposed for claim-guided evidence attention.
            "node_states": encoded["node_states"],
            "neighborhood_edge_counts": encoded["neighborhood_edge_counts"],
        }
        for key in (
            "sibling_gate",
            "sibling_attention_entropy",
            "sibling_group_count",
            "sibling_gated_node_share",
            "sibling_attention_self_weight",
            "sibling_event_repr",
            "sibling_event_norm",
            "sibling_relative_norm",
        ):
            if key in encoded:
                outputs[key] = encoded[key]
        if "acm_channel_names" in encoded:
            outputs["acm_channel_names"] = encoded["acm_channel_names"]
            outputs["acm_gate_per_layer"] = encoded["acm_gate_per_layer"]
            outputs["acm_zero_channel_gate_mass"] = encoded["acm_zero_channel_gate_mass"]
            for name in encoded["acm_channel_names"]:
                outputs[f"acm_gate_{name}"] = encoded[f"acm_gate_{name}"]
        for diagnostic in (
            "time_gate_mean",
            "time_gate_per_head",
            "mean_abs_time_bias_semantic",
        ):
            if diagnostic in encoded:
                outputs[diagnostic] = encoded[diagnostic]
        if not self.reconstruction_enabled:
            return outputs

        if graph_batch is None:
            mask = self._sample_mask(
                node_features.size(0),
                root_index,
                node_features.device,
            )
        else:
            if num_graphs is None:
                num_graphs = int(root_index.numel())
            mask = self._sample_batched_mask(
                node_features.size(0),
                root_index,
                graph_batch,
                int(num_graphs),
                node_features.device,
            )
        if bool(mask.any().item()):
            if self.target_mode == "latent":
                reconstruction_targets = self._deterministic_latent_targets(
                    node_features=node_features,
                    edge_index_td=edge_index_td,
                    edge_index_bu=edge_index_bu,
                    edge_type_td=edge_type_td,
                    edge_type_bu=edge_type_bu,
                    root_index=root_index,
                    graph_batch=graph_batch,
                    num_graphs=num_graphs,
                    node_time=node_time,
                )
            else:
                reconstruction_targets = node_features
            masked = self.encoder(
                node_features=node_features,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_type_td=edge_type_td,
                edge_type_bu=edge_type_bu,
                root_index=root_index,
                graph_batch=graph_batch,
                num_graphs=num_graphs,
                node_feature_mask=mask,
                mask_token=self.mask_token,
                node_time=node_time,
                keep_edge_families=(
                    (0,) if self.reconstruction_real_edges_only else None
                ),
            )
            edge_index_ud, edge_type_ud = self._undirected_edges(
                edge_index_td, edge_index_bu, edge_type_td, edge_type_bu
            )
            loss, predictions, targets = self._reconstruct(
                masked["node_states"], reconstruction_targets, edge_index_ud, edge_type_ud, mask
            )
            outputs["reconstruction_loss"] = loss
            outputs["masked_count"] = mask.to(dtype=node_features.dtype).sum()
            reconstruction_outputs = {
                "global_semantic_predictions": predictions,
                "global_semantic_targets": targets,
                "global_semantic_mask": mask,
            }
            outputs["reconstruction"] = reconstruction_outputs
        return outputs


class LatentSetTransformerBlock(nn.Module):
    """Refine latent graph tokens with packed node sets and latent self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_key_value_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,
        node_sets: torch.Tensor,
        node_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_latents = self.cross_query_norm(latents)
        normalized_nodes = self.cross_key_value_norm(node_sets)
        cross_output, _ = self.cross_attention(
            normalized_latents,
            normalized_nodes,
            normalized_nodes,
            key_padding_mask=node_padding_mask,
            need_weights=False,
        )
        latents = latents + self.dropout(cross_output)

        normalized_latents = self.self_attention_norm(latents)
        self_output, _ = self.self_attention(
            normalized_latents,
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        latents = latents + self.dropout(self_output)
        latents = latents + self.dropout(
            self.feed_forward(self.feed_forward_norm(latents))
        )
        return latents


class GlobalSetTransformerEncoder(nn.Module):
    """Encode all posts in an event without relying on edges, order, or timestamps.

    A small bank of learned latent tokens cross-attends to every node in each graph. This
    captures non-local interactions between replies in different branches with O(N * L)
    attention, where L is the fixed number of latent tokens. GA-GRU remains responsible for
    parent-child propagation; this encoder deliberately supplies a complementary global view.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_latents: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("Set Transformer hidden_dim must be positive.")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("Set Transformer hidden_dim must be divisible by num_heads.")
        if num_latents <= 0:
            raise ValueError("Set Transformer num_latents must be positive.")
        if num_layers <= 0:
            raise ValueError("Set Transformer num_layers must be positive.")

        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.root_marker = nn.Parameter(torch.zeros(hidden_dim))
        self.root_condition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.latent_tokens = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                LatentSetTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.root_marker, std=0.02)
        nn.init.normal_(self.latent_tokens, std=0.02)

    @staticmethod
    def _pack_node_sets(
        node_states: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_states.ndim != 2:
            raise ValueError("Set Transformer node states must have shape [nodes, hidden].")
        if graph_batch.ndim != 1 or graph_batch.numel() != node_states.size(0):
            raise ValueError("graph_batch must contain one graph index per node.")
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")
        if node_states.size(0) == 0:
            raise ValueError("Set Transformer cannot encode an empty graph batch.")

        graph_batch = graph_batch.to(device=node_states.device, dtype=torch.long)
        if int(graph_batch.min().item()) < 0 or int(graph_batch.max().item()) >= num_graphs:
            raise ValueError("graph_batch contains an out-of-range graph index.")
        if graph_batch.numel() > 1 and bool((graph_batch[1:] < graph_batch[:-1]).any().item()):
            raise ValueError("Set Transformer expects graph nodes to be packed contiguously.")

        counts = torch.bincount(graph_batch, minlength=num_graphs)
        if bool((counts == 0).any().item()):
            raise ValueError("Every graph must contain at least one node.")
        offsets = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(node_states.size(0), device=node_states.device) - offsets[graph_batch]
        max_nodes = int(counts.max().item())
        flat_indices = graph_batch * max_nodes + positions

        padded = node_states.new_zeros((num_graphs * max_nodes, node_states.size(-1)))
        padded = padded.index_copy(0, flat_indices, node_states)
        padded = padded.view(num_graphs, max_nodes, node_states.size(-1))

        valid = torch.zeros(num_graphs * max_nodes, dtype=torch.bool, device=node_states.device)
        valid.index_fill_(0, flat_indices, True)
        padding_mask = ~valid.view(num_graphs, max_nodes)
        return padded, padding_mask

    def forward(
        self,
        node_features: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        root_indices = root_indices.to(device=node_features.device, dtype=torch.long).view(-1)
        graph_batch = graph_batch.to(device=node_features.device, dtype=torch.long)
        if root_indices.numel() != num_graphs:
            raise ValueError("Set Transformer requires one root index per graph.")
        if bool(((root_indices < 0) | (root_indices >= node_features.size(0))).any().item()):
            raise ValueError("Set Transformer root index is out of range.")
        expected_graphs = torch.arange(num_graphs, device=node_features.device)
        if not torch.equal(graph_batch[root_indices], expected_graphs):
            raise ValueError("Each root index must belong to its corresponding packed graph.")

        node_states = self.input_projection(node_features)
        root_offsets = torch.zeros_like(node_states)
        root_offsets.index_add_(
            0,
            root_indices,
            self.root_marker.unsqueeze(0).expand(num_graphs, -1),
        )
        node_states = node_states + root_offsets
        node_sets, node_padding_mask = self._pack_node_sets(
            node_states,
            graph_batch,
            num_graphs,
        )

        root_states = node_states[root_indices]
        latents = self.latent_tokens.expand(num_graphs, -1, -1)
        latents = latents + self.root_condition(root_states).unsqueeze(1)
        for block in self.blocks:
            latents = block(latents, node_sets, node_padding_mask)

        normalized_latents = self.output_norm(latents)
        return {
            "pooled": normalized_latents.mean(dim=1),
            "latents": normalized_latents,
        }


class PropagationSemanticGRU(nn.Module):
    """Graph-aware GRU predicting semantic drift along propagation edges (recursive scheme).

    Each direction runs one pass over the tree in dependency order. For a node v:
      * attention aggregates the *already-updated* states of v's direct parents
        (top-down) or direct children (bottom-up). The query can be globally learned or
        conditioned on v, and an optional mean residual limits noisy attention weights;
      * a GRU cell then updates v's state from its own projected features with the context
        as the initial hidden state, so the state recursively summarises v plus its side of
        the tree and carries multi-hop information to the next level.

    Semantic supervision is edge-wise. For every propagation edge p -> c, the recursive
    top-down state of p predicts c and the recursive bottom-up state of c predicts p. The
    target endpoint's own state is not used by its predictor. The optional directional
    readout averages TD states while retaining the BU root, where recursive child
    information converges.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        semantic_dim: int | None = None,
        semantic_loss_mse_weight: float = 0.1,
        semantic_target: str = "online",
        semantic_prediction_mode: str = "absolute",
        semantic_loss_criterion: str = "auto",
        semantic_sce_gamma: float = 1.0,
        direction: str = "bidirectional",
        aggregation: str = "attention",
        attention_query_mode: str = "global",
        attention_score_mode: str = "dot",
        attention_residual_alpha: float = 1.0,
        attention_residual_mode: str = "fixed",
        attention_gate_init: float = 0.1,
        semantic_direction_reduction: str = "sum",
        direction_fusion: str = "joint",
        bu_residual_init: float = 0.1,
        bu_shared_gradient_scale: float = 1.0,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        semantic_dim = semantic_dim or hidden_dim
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive.")
        if semantic_loss_mse_weight < 0.0:
            raise ValueError("semantic_loss_mse_weight must be non-negative.")
        if semantic_target not in ("online", "fixed", "raw"):
            raise ValueError("semantic_target must be 'online', 'fixed', or 'raw'.")
        if semantic_prediction_mode not in ("absolute", "residual"):
            raise ValueError(
                "semantic_prediction_mode must be 'absolute' or 'residual'."
            )
        if semantic_loss_criterion not in ("auto", "mse", "cosine", "sce"):
            raise ValueError(
                "semantic_loss_criterion must be 'auto', 'mse', 'cosine', or 'sce'."
            )
        if semantic_sce_gamma < 1.0:
            raise ValueError("semantic_sce_gamma must be at least 1.0.")
        if direction not in ("bidirectional", "topdown", "bottomup"):
            raise ValueError(
                "direction must be 'bidirectional', 'topdown', or 'bottomup'."
            )
        if aggregation not in ("attention", "mean"):
            raise ValueError("aggregation must be 'attention' or 'mean'.")
        if attention_query_mode not in ("global", "node"):
            raise ValueError("attention_query_mode must be 'global' or 'node'.")
        if attention_score_mode not in ("dot", "additive"):
            raise ValueError("attention_score_mode must be 'dot' or 'additive'.")
        if not 0.0 <= attention_residual_alpha <= 1.0:
            raise ValueError("attention_residual_alpha must be in [0, 1].")
        if attention_residual_mode not in ("fixed", "adaptive"):
            raise ValueError("attention_residual_mode must be 'fixed' or 'adaptive'.")
        if not 0.0 < attention_gate_init < 1.0:
            raise ValueError("attention_gate_init must be in (0, 1).")
        if semantic_direction_reduction not in ("sum", "mean"):
            raise ValueError(
                "semantic_direction_reduction must be 'sum' or 'mean'."
            )
        if direction_fusion not in ("joint", "td-residual"):
            raise ValueError("direction_fusion must be 'joint' or 'td-residual'.")
        if not 0.0 < bu_residual_init < 1.0:
            raise ValueError("bu_residual_init must be in (0, 1).")
        if not 0.0 <= bu_shared_gradient_scale <= 1.0:
            raise ValueError("bu_shared_gradient_scale must be in [0, 1].")
        if readout not in ("mean", "td-mean-bu-root", "evidence"):
            raise ValueError("readout must be 'mean', 'td-mean-bu-root', or 'evidence'.")
        self.hidden_dim = hidden_dim
        self.semantic_dim = semantic_dim
        self.prediction_dim = input_dim if semantic_target == "raw" else semantic_dim
        self.semantic_loss_mse_weight = semantic_loss_mse_weight
        self.semantic_target = semantic_target
        self.semantic_prediction_mode = semantic_prediction_mode
        self.semantic_loss_criterion = semantic_loss_criterion
        self.semantic_sce_gamma = semantic_sce_gamma
        self.direction = direction
        self.aggregation = aggregation
        self.attention_query_mode = attention_query_mode
        self.attention_score_mode = attention_score_mode
        self.attention_residual_alpha = float(attention_residual_alpha)
        self.attention_residual_mode = attention_residual_mode
        self.attention_gate_init = float(attention_gate_init)
        self.semantic_direction_reduction = semantic_direction_reduction
        self.direction_fusion = direction_fusion
        self.bu_residual_init = float(bu_residual_init)
        self.bu_shared_gradient_scale = float(bu_shared_gradient_scale)
        self.readout = readout
        self.reconstruction_enabled = True
        if readout == "evidence":
            self.evidence_readout = EvidenceReadout(2 * hidden_dim, dropout)
        self.use_topdown = direction in ("bidirectional", "topdown")
        self.use_bottomup = direction in ("bidirectional", "bottomup")
        self.semantic_proj = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(semantic_dim),
        )
        # Stationary self-prediction target. `semantic_proj` is optimised by the
        # classification loss, so using it as the prediction target gives the predictors a
        # target that drifts every step and the auxiliary loss cannot converge. With
        # `semantic_target="fixed"` the predictors instead regress a frozen low-dimensional
        # projection of the raw node features, which is stationary and meaningful (a random
        # linear projection preserves pairwise structure), so the loss is well-posed.
        if semantic_target == "fixed":
            fixed_target_proj = nn.Linear(input_dim, semantic_dim, bias=False)
            nn.init.xavier_uniform_(fixed_target_proj.weight)
            fixed_target_proj.weight.requires_grad_(False)
            self.fixed_target_proj = fixed_target_proj
        self.state_input_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.down_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.up_gru = nn.GRUCell(hidden_dim, hidden_dim)
        # Learnable attention queries for aggregating direct parents / children.
        self.down_context_start = nn.Parameter(torch.zeros(hidden_dim))
        self.up_context_start = nn.Parameter(torch.zeros(hidden_dim))

        self.down_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if self.attention_score_mode == "additive":
            self.down_score = nn.Linear(hidden_dim, 1, bias=False)
            self.up_score = nn.Linear(hidden_dim, 1, bias=False)
        if self.attention_residual_mode == "adaptive":
            self.down_attention_gate = self._make_attention_gate(
                hidden_dim,
                self.attention_gate_init,
            )
            self.up_attention_gate = self._make_attention_gate(
                hidden_dim,
                self.attention_gate_init,
            )
        self.child_semantic_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.prediction_dim),
        )
        self.parent_semantic_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.prediction_dim),
        )
        if self.semantic_prediction_mode == "residual":
            for predictor in (
                self.child_semantic_predictor,
                self.parent_semantic_predictor,
            ):
                nn.init.zeros_(predictor[-1].weight)
                nn.init.zeros_(predictor[-1].bias)

        self.state_norm = nn.LayerNorm(hidden_dim * 2)
        if self.direction_fusion == "td-residual":
            self.td_state_norm = nn.LayerNorm(hidden_dim * 2)
            self.bu_state_norm = nn.LayerNorm(hidden_dim * 2)
            self.bu_residual_logit = nn.Parameter(
                torch.tensor(math.log(bu_residual_init / (1.0 - bu_residual_init)))
            )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_attention_gate(hidden_dim: int, initial_gate: float) -> nn.Sequential:
        gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, 1),
        )
        nn.init.zeros_(gate[-1].weight)
        nn.init.constant_(gate[-1].bias, math.log(initial_gate / (1.0 - initial_gate)))
        return gate

    def _blend_attention_context(
        self,
        query_inputs: torch.Tensor,
        mean_context: torch.Tensor,
        attention_context: torch.Tensor,
        gate: Optional[nn.Module],
    ) -> torch.Tensor:
        residual = attention_context - mean_context
        if self.attention_residual_mode == "fixed":
            return mean_context + self.attention_residual_alpha * residual
        if gate is None:
            raise RuntimeError("Adaptive attention requires a direction-specific gate.")
        gate_inputs = torch.cat([query_inputs, mean_context, residual], dim=-1)
        adaptive_weight = torch.sigmoid(gate(gate_inputs))
        return mean_context + self.attention_residual_alpha * adaptive_weight * residual

    @staticmethod
    def _neighbors(edge_index: torch.Tensor, num_nodes: int) -> tuple[List[List[int]], List[List[int]]]:
        parents: List[List[int]] = [[] for _ in range(num_nodes)]
        children: List[List[int]] = [[] for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return parents, children

        src_nodes = edge_index[0].detach().cpu().tolist()
        dst_nodes = edge_index[1].detach().cpu().tolist()
        for src, dst in zip(src_nodes, dst_nodes):
            src = int(src)
            dst = int(dst)
            if src == dst or src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                continue
            parents[dst].append(src)
            children[src].append(dst)
        return parents, children

    @staticmethod
    def _depth_order_and_depths(children: List[List[int]], root_index: int) -> tuple[List[int], List[int]]:
        num_nodes = len(children)
        depth = [-1 for _ in range(num_nodes)]
        if 0 <= root_index < num_nodes:
            depth[root_index] = 0
            queue = [root_index]
            cursor = 0
            while cursor < len(queue):
                node = queue[cursor]
                cursor += 1
                for child in children[node]:
                    if depth[child] >= 0:
                        continue
                    depth[child] = depth[node] + 1
                    queue.append(child)

        for node in range(num_nodes):
            if depth[node] < 0:
                depth[node] = 0
        order = sorted(range(num_nodes), key=lambda node: depth[node])
        return order, depth

    def _attend(
        self,
        query_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        query_proj: nn.Linear,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
        score_proj: Optional[nn.Linear] = None,
        gate: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        values = value_proj(neighbor_states)
        mean_context = values.mean(dim=0)
        if self.aggregation == "mean" or self.attention_residual_alpha == 0.0:
            return self.dropout(mean_context)

        query = query_proj(query_state).unsqueeze(0)
        keys = key_proj(neighbor_states)
        if self.attention_score_mode == "additive":
            if score_proj is None:
                raise RuntimeError("Additive attention requires a score projection.")
            scores = score_proj(torch.tanh(keys + query)).squeeze(-1)
        else:
            scores = (keys * query).sum(dim=-1) / (self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=0)
        attention_context = torch.sum(weights.unsqueeze(-1) * values, dim=0)
        context = self._blend_attention_context(
            query_state,
            mean_context,
            attention_context,
            gate,
        )
        return self.dropout(context)

    @staticmethod
    def _semantic_vector_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mse_weight: float,
        raw_mse: bool = False,
        criterion: str = "auto",
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Vector regression loss for the parent->child / child->parent predictions.

        `auto` keeps the historical behaviour exactly as it was (plain MSE for the raw BERT
        target, otherwise cosine plus a small MSE term). The explicit criteria exist because MSE
        on 768-d BERT coordinates optimises euclidean distance, which is not the same thing as
        semantic distance: it spends capacity on vector norm and on individual dimensions. The
        cosine criteria drop the magnitude and keep only direction, and `sce` additionally
        down-weights whatever is already well reconstructed (GraphMAE's scaled cosine error).
        """
        if predictions.numel() == 0:
            return targets.sum() * 0.0
        prediction_vectors = predictions
        target_vectors = targets.detach()
        if criterion == "auto":
            if raw_mse:
                return F.mse_loss(prediction_vectors, target_vectors)
            cosine_loss = (
                1.0 - F.cosine_similarity(prediction_vectors, target_vectors, dim=-1)
            ).mean()
            mse_loss = F.mse_loss(prediction_vectors, target_vectors)
            return cosine_loss + mse_weight * mse_loss
        if criterion == "mse":
            return F.mse_loss(prediction_vectors, target_vectors)
        # cosine similarity is unbounded below for degenerate vectors, so clamp the per-edge
        # term into [0, 2] before the power; otherwise a single sign-flipped prediction with
        # gamma > 1 can dominate the batch.
        cosine_gap = torch.clamp(
            1.0 - F.cosine_similarity(prediction_vectors, target_vectors, dim=-1),
            min=0.0,
            max=2.0,
        )
        if criterion == "cosine":
            return cosine_gap.mean()
        return cosine_gap.pow(gamma).mean()

    def _edgewise_semantic_predictions(
        self,
        down_states: torch.Tensor,
        up_states: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        prediction_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = prediction_target.size(0)
        device = prediction_target.device
        edge_src = edge_src.to(device=device, dtype=torch.long)
        edge_dst = edge_dst.to(device=device, dtype=torch.long)

        if edge_src.numel() == 0 or not self.reconstruction_enabled:
            empty_predictions = prediction_target.new_empty((0, self.prediction_dim))
            empty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            node_predictions = prediction_target.new_zeros((num_nodes, self.prediction_dim))
            zero_loss = prediction_target.sum() * 0.0
            return {
                "child_semantic_predictions": node_predictions,
                "parent_semantic_predictions": node_predictions.clone(),
                "child_edge_semantic_predictions": empty_predictions,
                "parent_edge_semantic_predictions": empty_predictions.clone(),
                "semantic_edge_index": torch.empty((2, 0), dtype=torch.long, device=device),
                "child_semantic_loss": zero_loss,
                "parent_semantic_loss": zero_loss,
                "child_semantic_cosine_gap": zero_loss,
                "parent_semantic_cosine_gap": zero_loss,
                "semantic_loss": zero_loss,
                "child_prediction_mask": empty_mask,
                "parent_prediction_mask": empty_mask.clone(),
            }

        # For p -> c, p's recursive top-down state predicts c, while c's
        # recursive bottom-up state predicts p.
        zero_loss = prediction_target.sum() * 0.0
        if self.use_topdown:
            child_edge_residuals = self.child_semantic_predictor(down_states[edge_src])
            child_edge_predictions = (
                prediction_target[edge_src].detach() + child_edge_residuals
                if self.semantic_prediction_mode == "residual"
                else child_edge_residuals
            )
            child_semantic_loss = self._semantic_vector_loss(
                child_edge_predictions,
                prediction_target[edge_dst],
                self.semantic_loss_mse_weight,
                raw_mse=self.semantic_target == "raw",
                criterion=self.semantic_loss_criterion,
                gamma=self.semantic_sce_gamma,
            )
        else:
            child_edge_predictions = prediction_target.new_zeros(
                (edge_src.numel(), self.prediction_dim)
            )
            child_semantic_loss = zero_loss

        if self.use_bottomup:
            parent_edge_residuals = self.parent_semantic_predictor(up_states[edge_dst])
            parent_edge_predictions = (
                prediction_target[edge_dst].detach() + parent_edge_residuals
                if self.semantic_prediction_mode == "residual"
                else parent_edge_residuals
            )
            parent_semantic_loss = self._semantic_vector_loss(
                parent_edge_predictions,
                prediction_target[edge_src],
                self.semantic_loss_mse_weight,
                raw_mse=self.semantic_target == "raw",
                criterion=self.semantic_loss_criterion,
                gamma=self.semantic_sce_gamma,
            )
        else:
            parent_edge_predictions = prediction_target.new_zeros(
                (edge_dst.numel(), self.prediction_dim)
            )
            parent_semantic_loss = zero_loss

        child_counts = prediction_target.new_zeros(num_nodes)
        child_counts.index_add_(0, edge_dst, prediction_target.new_ones(edge_dst.numel()))
        parent_counts = prediction_target.new_zeros(num_nodes)
        parent_counts.index_add_(0, edge_src, prediction_target.new_ones(edge_src.numel()))

        child_semantic_predictions = prediction_target.new_zeros(
            (num_nodes, self.prediction_dim)
        )
        if self.use_topdown:
            child_semantic_predictions = child_semantic_predictions.index_add(
                0, edge_dst, child_edge_predictions
            )
            child_semantic_predictions = child_semantic_predictions / child_counts.clamp_min(
                1.0
            ).unsqueeze(-1)
        parent_semantic_predictions = prediction_target.new_zeros(
            (num_nodes, self.prediction_dim)
        )
        if self.use_bottomup:
            parent_semantic_predictions = parent_semantic_predictions.index_add(
                0, edge_src, parent_edge_predictions
            )
            parent_semantic_predictions = parent_semantic_predictions / parent_counts.clamp_min(
                1.0
            ).unsqueeze(-1)

        semantic_loss = child_semantic_loss + parent_semantic_loss
        if self.semantic_direction_reduction == "mean":
            active_directions = int(self.use_topdown) + int(self.use_bottomup)
            semantic_loss = semantic_loss / max(active_directions, 1)

        # Criterion-independent semantic evolution discrepancy d = 1 - cos(prediction, target),
        # reported even when the objective is MSE so the two criteria can be compared on the
        # same interpretable quantity rather than on losses with different scales.
        def _cosine_gap(edge_predictions: torch.Tensor, edge_targets: torch.Tensor) -> torch.Tensor:
            if edge_predictions.numel() == 0:
                return semantic_loss.new_zeros(())
            return (
                1.0
                - F.cosine_similarity(edge_predictions, edge_targets.detach(), dim=-1)
            ).mean()

        return {
            "child_semantic_predictions": child_semantic_predictions,
            "parent_semantic_predictions": parent_semantic_predictions,
            "child_edge_semantic_predictions": child_edge_predictions,
            "parent_edge_semantic_predictions": parent_edge_predictions,
            "semantic_edge_index": torch.stack([edge_src, edge_dst], dim=0),
            "child_semantic_loss": child_semantic_loss,
            "parent_semantic_loss": parent_semantic_loss,
            "child_semantic_cosine_gap": _cosine_gap(child_edge_predictions, prediction_target[edge_dst]),
            "parent_semantic_cosine_gap": _cosine_gap(parent_edge_predictions, prediction_target[edge_src]),
            "semantic_loss": semantic_loss,
            "child_prediction_mask": (child_counts > 0) & self.use_topdown,
            "parent_prediction_mask": (parent_counts > 0) & self.use_bottomup,
        }

    def _prediction_target(
        self,
        node_features: torch.Tensor,
        projected_semantics: torch.Tensor,
        prediction_target_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_features = (
            node_features
            if prediction_target_features is None
            else prediction_target_features
        )
        if target_features.shape != node_features.shape:
            raise ValueError(
                "GA-GRU prediction target features must match the online node-feature shape."
            )
        if self.semantic_target == "raw":
            return target_features.detach()
        if self.semantic_target == "fixed":
            with torch.no_grad():
                return F.layer_norm(
                    self.fixed_target_proj(target_features),
                    (self.semantic_dim,),
                )
        return projected_semantics

    @staticmethod
    def _batched_depths(
        edge_index: torch.Tensor,
        num_nodes: int,
        root_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        children: List[List[int]] = [[] for _ in range(num_nodes)]
        if edge_index.numel() > 0:
            src_nodes = edge_index[0].detach().cpu().tolist()
            dst_nodes = edge_index[1].detach().cpu().tolist()
            for src, dst in zip(src_nodes, dst_nodes):
                src = int(src)
                dst = int(dst)
                if src == dst or src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                    continue
                children[src].append(dst)

        depth = [-1 for _ in range(num_nodes)]
        for root in root_indices.detach().cpu().tolist():
            root = int(root)
            if root < 0 or root >= num_nodes:
                continue
            if depth[root] < 0:
                depth[root] = 0
            queue = [root]
            cursor = 0
            while cursor < len(queue):
                node = queue[cursor]
                cursor += 1
                for child in children[node]:
                    if depth[child] >= 0:
                        continue
                    depth[child] = depth[node] + 1
                    queue.append(child)

        for node in range(num_nodes):
            if depth[node] < 0:
                depth[node] = 0
        return torch.tensor(depth, dtype=torch.long, device=device)

    def _edge_context(
        self,
        neighbor_states: torch.Tensor,
        owner_nodes: torch.Tensor,
        num_nodes: int,
        query_state: torch.Tensor,
        query_proj: nn.Linear,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
        owner_query_states: Optional[torch.Tensor] = None,
        score_proj: Optional[nn.Linear] = None,
        gate: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        contexts = neighbor_states.new_zeros((num_nodes, self.hidden_dim))
        if owner_nodes.numel() == 0:
            return contexts
        values = value_proj(neighbor_states)
        counts = values.new_zeros(num_nodes)
        counts.index_add_(0, owner_nodes, values.new_ones(owner_nodes.numel()))
        mean_weights = counts[owner_nodes].clamp_min(1.0).reciprocal()
        contexts.index_add_(0, owner_nodes, mean_weights.unsqueeze(-1) * values)
        if self.aggregation == "mean" or self.attention_residual_alpha == 0.0:
            return self.dropout(contexts)

        if self.attention_query_mode == "node":
            if owner_query_states is None:
                raise ValueError(
                    "owner_query_states is required for node-conditioned attention."
                )
            if owner_query_states.size(0) != num_nodes:
                raise ValueError(
                    "owner_query_states must contain one query state per node."
                )
            owner_query_inputs = owner_query_states + query_state
            query = query_proj(owner_query_inputs[owner_nodes])
        else:
            owner_query_inputs = query_state.unsqueeze(0).expand(num_nodes, -1)
            query = query_proj(query_state).unsqueeze(0)
        keys = key_proj(neighbor_states)
        if self.attention_score_mode == "additive":
            if score_proj is None:
                raise RuntimeError("Additive attention requires a score projection.")
            scores = score_proj(torch.tanh(keys + query)).squeeze(-1)
        else:
            scores = (keys * query).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attention_weights = DenseGraphAttentionLayer.edge_softmax_by_destination(
            scores.unsqueeze(-1), owner_nodes, num_nodes
        ).squeeze(-1)
        attention_contexts = torch.zeros_like(contexts)
        attention_contexts.index_add_(
            0,
            owner_nodes,
            attention_weights.unsqueeze(-1) * values,
        )
        contexts = self._blend_attention_context(
            owner_query_inputs,
            contexts,
            attention_contexts,
            gate,
        )
        return self.dropout(contexts)

    def _combine_directional_states(
        self,
        down_states: torch.Tensor,
        up_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.direction_fusion == "td-residual":
            zero_states = torch.zeros_like(down_states)
            td_states = self.td_state_norm(
                torch.cat([down_states, zero_states], dim=-1)
            )
            td_states = torch.cat(
                [td_states[..., : self.hidden_dim], zero_states],
                dim=-1,
            )
            bu_states = self.bu_state_norm(
                torch.cat([zero_states, up_states], dim=-1)
            )
            bu_states = torch.cat(
                [zero_states, bu_states[..., self.hidden_dim :]],
                dim=-1,
            )
            if self.use_topdown and self.use_bottomup:
                bu_weight = torch.sigmoid(self.bu_residual_logit)
                return td_states + bu_weight * bu_states
            if self.use_topdown:
                return td_states
            return bu_states

        down_output = down_states if self.use_topdown else torch.zeros_like(down_states)
        up_output = up_states if self.use_bottomup else torch.zeros_like(up_states)
        node_states = self.state_norm(torch.cat([down_output, up_output], dim=-1))
        if not self.use_topdown:
            node_states = torch.cat(
                [torch.zeros_like(node_states[..., : self.hidden_dim]), node_states[..., self.hidden_dim :]],
                dim=-1,
            )
        if not self.use_bottomup:
            node_states = torch.cat(
                [node_states[..., : self.hidden_dim], torch.zeros_like(node_states[..., self.hidden_dim :])],
                dim=-1,
            )
        return node_states

    def _bottomup_state_inputs(self, state_inputs: torch.Tensor) -> torch.Tensor:
        """Retain BU forward features while controlling gradients into the shared TD projection."""
        if self.bu_shared_gradient_scale >= 1.0:
            return state_inputs
        detached_inputs = state_inputs.detach()
        return detached_inputs + self.bu_shared_gradient_scale * (
            state_inputs - detached_inputs
        )

    def _graph_readout(
        self,
        node_states: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.readout == "evidence":
            return self.evidence_readout(node_states, root_indices, graph_batch, num_graphs)[0]
        mean_pooled = graph_mean_pool(node_states, graph_batch, num_graphs)
        if self.readout == "mean":
            return mean_pooled
        if root_indices.numel() != num_graphs:
            raise ValueError("root_indices must contain exactly one root per graph.")
        if bool(((root_indices < 0) | (root_indices >= node_states.size(0))).any().item()):
            raise ValueError("root_indices contains an out-of-range node index.")

        # TD information is distributed across descendants, whereas recursive BU
        # information converges at each graph's root. Averaging every BU state would
        # dilute that summary by roughly the graph size on shallow, star-shaped trees.
        td_pooled = mean_pooled[:, : self.hidden_dim]
        bu_root = node_states[root_indices, self.hidden_dim :]
        return torch.cat([td_pooled, bu_root], dim=-1)

    def forward_batched(
        self,
        node_features: torch.Tensor,
        edge_index_td: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
        prediction_target_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = node_features.size(0)
        device = node_features.device
        graph_batch = graph_batch.to(device=device, dtype=torch.long)
        root_indices = root_indices.to(device=device, dtype=torch.long)
        target_semantics = self.semantic_proj(node_features)
        state_inputs = self.state_input_proj(target_semantics)
        prediction_target = self._prediction_target(
            node_features,
            target_semantics,
            prediction_target_features,
        )

        depths = self._batched_depths(edge_index_td, num_nodes, root_indices, device)
        if edge_index_td.numel() > 0:
            edge_src = edge_index_td[0].to(device=device, dtype=torch.long)
            edge_dst = edge_index_td[1].to(device=device, dtype=torch.long)
            valid_edges = (
                (edge_src >= 0)
                & (edge_src < num_nodes)
                & (edge_dst >= 0)
                & (edge_dst < num_nodes)
                & (edge_src != edge_dst)
            )
            edge_src = edge_src[valid_edges]
            edge_dst = edge_dst[valid_edges]
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)

        max_depth = int(depths.max().detach().cpu().item()) if num_nodes > 0 else 0
        down_states = state_inputs
        if self.use_topdown:
            for depth_value in range(max_depth + 1):
                edge_mask = depths[edge_dst] == depth_value if edge_dst.numel() > 0 else edge_dst.bool()
                if not bool(edge_mask.any().item()):
                    continue
                active_nodes = torch.unique(edge_dst[edge_mask], sorted=True)
                contexts = self._edge_context(
                    down_states[edge_src[edge_mask]],
                    edge_dst[edge_mask],
                    num_nodes,
                    self.down_context_start,
                    self.down_query,
                    self.down_key,
                    self.down_value,
                    owner_query_states=state_inputs,
                    score_proj=getattr(self, "down_score", None),
                    gate=getattr(self, "down_attention_gate", None),
                )[active_nodes]
                down_states = down_states.index_copy(
                    0, active_nodes, self.down_gru(state_inputs[active_nodes], contexts)
                )

        up_state_inputs = self._bottomup_state_inputs(state_inputs)
        up_states = up_state_inputs
        if self.use_bottomup:
            for depth_value in range(max_depth, -1, -1):
                edge_mask = depths[edge_src] == depth_value if edge_src.numel() > 0 else edge_src.bool()
                if not bool(edge_mask.any().item()):
                    continue
                active_nodes = torch.unique(edge_src[edge_mask], sorted=True)
                contexts = self._edge_context(
                    up_states[edge_dst[edge_mask]],
                    edge_src[edge_mask],
                    num_nodes,
                    self.up_context_start,
                    self.up_query,
                    self.up_key,
                    self.up_value,
                    owner_query_states=up_state_inputs,
                    score_proj=getattr(self, "up_score", None),
                    gate=getattr(self, "up_attention_gate", None),
                )[active_nodes]
                up_states = up_states.index_copy(
                    0, active_nodes, self.up_gru(up_state_inputs[active_nodes], contexts)
                )

        node_states = self._combine_directional_states(down_states, up_states)
        pooled = self._graph_readout(
            node_states,
            root_indices,
            graph_batch,
            int(num_graphs),
        )
        prediction_outputs = self._edgewise_semantic_predictions(
            down_states,
            up_states,
            edge_src,
            edge_dst,
            prediction_target,
        )

        return {
            "pooled": pooled,
            # Per-node recursive states; needed by claim-guided evidence attention.
            "node_states": node_states,
            "target_semantics": prediction_target,
            **prediction_outputs,
        }

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index_td: torch.Tensor,
        root_index: torch.Tensor,
        prediction_target_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = node_features.size(0)
        device = node_features.device
        target_semantics = self.semantic_proj(node_features)
        state_inputs = self.state_input_proj(target_semantics)
        prediction_target = self._prediction_target(
            node_features,
            target_semantics,
            prediction_target_features,
        )
        parents, children = self._neighbors(edge_index_td, num_nodes)
        order, depths = self._depth_order_and_depths(children, int(root_index.item()))

        # Recursive one-hop scheme: nodes are visited parents-first (top-down) or
        # children-first (bottom-up), so a direct neighbour's state already summarises its
        # whole side of the tree — attending over direct neighbours therefore carries
        # multi-hop information without walking full paths (O(N) instead of O(N*depth)).
        # The target node's own features never enter its prediction context.
        down_states: List[torch.Tensor] = [state_inputs[node] for node in range(num_nodes)]
        if self.use_topdown:
            for node in order:
                node_parents = parents[node]
                if not node_parents:
                    continue
                parent_states = torch.stack([down_states[parent] for parent in node_parents], dim=0)
                query_state = self.down_context_start
                if self.attention_query_mode == "node":
                    query_state = query_state + state_inputs[node]
                context = self._attend(
                    query_state,
                    parent_states,
                    self.down_query,
                    self.down_key,
                    self.down_value,
                    getattr(self, "down_score", None),
                    getattr(self, "down_attention_gate", None),
                )
                down_states[node] = self.down_gru(
                    state_inputs[node].unsqueeze(0), context.unsqueeze(0)
                ).squeeze(0)

        up_state_inputs = self._bottomup_state_inputs(state_inputs)
        up_states: List[torch.Tensor] = [
            up_state_inputs[node] for node in range(num_nodes)
        ]
        if self.use_bottomup:
            for node in reversed(order):
                node_children = children[node]
                if not node_children:
                    continue
                child_states = torch.stack([up_states[child] for child in node_children], dim=0)
                query_state = self.up_context_start
                if self.attention_query_mode == "node":
                    query_state = query_state + up_state_inputs[node]
                context = self._attend(
                    query_state,
                    child_states,
                    self.up_query,
                    self.up_key,
                    self.up_value,
                    getattr(self, "up_score", None),
                    getattr(self, "up_attention_gate", None),
                )
                up_states[node] = self.up_gru(
                    up_state_inputs[node].unsqueeze(0), context.unsqueeze(0)
                ).squeeze(0)

        down_tensor = torch.stack(down_states, dim=0)
        up_tensor = torch.stack(up_states, dim=0)

        node_states = self._combine_directional_states(down_tensor, up_tensor)
        pooled = self._graph_readout(
            node_states,
            root_index.reshape(1).to(device=device, dtype=torch.long),
            torch.zeros(num_nodes, dtype=torch.long, device=device),
            1,
        )[0]

        if edge_index_td.numel() > 0:
            edge_src = edge_index_td[0].to(device=device, dtype=torch.long)
            edge_dst = edge_index_td[1].to(device=device, dtype=torch.long)
            valid_edges = (
                (edge_src >= 0)
                & (edge_src < num_nodes)
                & (edge_dst >= 0)
                & (edge_dst < num_nodes)
                & (edge_src != edge_dst)
            )
            edge_src = edge_src[valid_edges]
            edge_dst = edge_dst[valid_edges]
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)
        prediction_outputs = self._edgewise_semantic_predictions(
            down_tensor,
            up_tensor,
            edge_src,
            edge_dst,
            prediction_target,
        )

        return {
            "pooled": pooled,
            # Per-node recursive states; needed by claim-guided evidence attention.
            "node_states": node_states,
            "target_semantics": prediction_target,
            **prediction_outputs,
        }


class SourceResidualClassifier(nn.Module):
    """Source-anchored residual evidence head (V1G-R1).

    ``l_s = SourceHead(h_s)`` is the source-only initial decision over the shared
    GA-GRU semantic projection of the root, while the fused propagation
    representation may only predict a logit correction ``Δl``.  Final logits are
    ``l_s + γΔl`` with a learnable sigmoid gate initialized to ``gate_init``, so
    propagation evidence starts as a weak revision instead of a second
    full-event classifier.
    """

    def __init__(
        self,
        source_dim: int,
        propagation_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        gate_init: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("source residual gate_init must be in (0, 1).")
        self.gate_init = float(gate_init)
        self.source_head = nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Dropout(dropout),
            nn.Linear(source_dim, num_classes),
        )
        # No output bias: a revision must be explained by event evidence rather
        # than a global class offset learned on the propagation branch.
        self.correction_head = nn.Sequential(
            nn.LayerNorm(propagation_dim),
            nn.Linear(propagation_dim, num_classes, bias=False),
        )
        self.correction_gate_logit = nn.Parameter(
            torch.tensor(math.log(self.gate_init / (1.0 - self.gate_init)))
        )

    def forward(
        self,
        source_repr: torch.Tensor,
        propagation_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source_logits = self.source_head(source_repr)
        delta_logits = self.correction_head(propagation_repr)
        gamma = torch.sigmoid(self.correction_gate_logit)
        return {
            "logits": source_logits + gamma * delta_logits,
            "source_logits": source_logits,
            "delta_logits": delta_logits,
            "gamma": gamma,
        }


class GATGAGRUFusionClassifier(nn.Module):
    """Fuse a global structure branch and propagation-semantic GA-GRU by concatenation.

    Historical GAT names remain as compatibility aliases. The first branch may also be a
    Set Transformer.
    """

    def __init__(
        self,
        gat_dim: int,
        gagru_dim: int,
        fusion_dim: int,
        dropout: float,
        num_classes: int = 2,
        use_gat: bool = True,
        uniformity_temperature: float = 2.0,
        temporal_dim: int = 0,
        temporal_identity_init: bool = True,
        root_feature_dim: int = 0,
        root_projection_dim: int = 0,
    ) -> None:
        super().__init__()
        self.use_gat = use_gat
        self.uniformity_temperature = uniformity_temperature
        self.temporal_dim = int(temporal_dim)
        if self.temporal_dim < 0:
            raise ValueError("temporal_dim cannot be negative.")
        self.gat_proj = nn.Linear(gat_dim, fusion_dim)
        self.gagru_proj = nn.Linear(gagru_dim, fusion_dim)
        self.gat_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        classifier_input_dim = fusion_dim * 2
        # Root Feature Enhancement (Bi-GCN): concatenate a projection of the source post's own
        # representation into the classifier input so the claim cannot be diluted by the reply
        # pool. The published ablation shows every branch (UD/TD/BU/Bi-GCN) improves with it.
        # Default 0 keeps the concatenation, and therefore every existing run, bit-identical.
        self.root_feature_dim = int(root_feature_dim)
        self.root_projection_dim = int(root_projection_dim)
        if self.root_feature_dim > 0:
            if self.root_projection_dim <= 0:
                raise ValueError(
                    "root_projection_dim must be positive when root_feature_dim is enabled."
                )
            self.root_projection = nn.Linear(self.root_feature_dim, self.root_projection_dim)
            classifier_input_dim += self.root_projection_dim
        else:
            if self.root_projection_dim != 0:
                raise ValueError(
                    "root_projection_dim is meaningless while root_feature_dim is 0."
                )
            self.root_projection = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        # The temporal block enters through an additive head: h = h_M0 + W_T z_T.
        # That is the same function family as concatenating z_T into the first
        # Linear, but M0's layer shapes stay untouched, so with a zeroed head the
        # initial logits are bit-identical to M0. Widening the Linear instead would
        # change the reduction order and perturb the logits by ~1e-7.
        if self.temporal_dim > 0:
            self.temporal_head = nn.Linear(self.temporal_dim, fusion_dim)
            if temporal_identity_init:
                with torch.no_grad():
                    self.temporal_head.weight.zero_()
                    self.temporal_head.bias.zero_()
        else:
            self.temporal_head = None

    def forward(
        self,
        gat_repr: torch.Tensor,
        gagru_repr: torch.Tensor,
        temporal_repr: Optional[torch.Tensor] = None,
        root_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Plain concatenation: branch projections retain their learned relative
        # scales. The classifier's first LayerNorm operates on the concatenated
        # vector instead of forcing the weaker branch to equal norm beforehand.
        z_gagru = self.gagru_proj(gagru_repr)
        if self.use_gat:
            z_gat = self.gat_proj(gat_repr)
        else:
            z_gat = torch.zeros_like(z_gagru)
        fused = torch.cat([z_gat, z_gagru], dim=-1)
        root_report: Dict[str, torch.Tensor] = {}
        if self.root_projection is not None:
            if root_features is None:
                raise ValueError(
                    "This fusion classifier carries root feature enhancement, but no root "
                    "features were supplied."
                )
            if root_features.shape[-1] != self.root_feature_dim:
                raise ValueError(
                    "Root feature width mismatch: expected "
                    f"{self.root_feature_dim}, got {root_features.shape[-1]}."
                )
            z_root = self.root_projection(root_features)
            fused = torch.cat([fused, z_root], dim=-1)
            # A direct concatenation cannot be inert the way a zero-initialised residual can,
            # but ||W_r|| still shows whether the new block is actually receiving gradient.
            root_report = {
                "root_projection_weight_norm": self.root_projection.weight.norm(),
                "root_representation_norm": z_root.norm(dim=-1).mean(),
            }
        hidden = self.classifier[2](self.classifier[1](self.classifier[0](fused)))
        temporal_report: Dict[str, torch.Tensor] = {}
        if self.temporal_dim > 0:
            if temporal_repr is None:
                raise ValueError(
                    "This fusion classifier carries a temporal branch, but no temporal "
                    "representation was supplied."
                )
            if temporal_repr.shape[-1] != self.temporal_dim:
                raise ValueError(
                    "Temporal representation width mismatch: "
                    f"expected {self.temporal_dim}, got {temporal_repr.shape[-1]}."
                )
            delta = self.temporal_head(temporal_repr)
            # ||W_T z_T|| / ||h_M0|| is the only informative inertness signal: a
            # LayerNorm'd z_T always has norm sqrt(dim), so its own norm says
            # nothing about whether the branch reaches the logits.
            temporal_report = {
                "temporal_hidden_delta": delta,
                "temporal_contribution": delta.norm(dim=-1).mean()
                / (hidden.norm(dim=-1).mean() + 1e-8),
                "temporal_head_weight_norm": self.temporal_head.weight.norm(),
            }
            hidden = hidden + delta
        elif temporal_repr is not None:
            raise ValueError(
                "A temporal representation was supplied to a classifier that was built "
                "without temporal_dim; construct the model with temporal dynamics enabled."
            )
        for layer in self.classifier[3:]:
            hidden = layer(hidden)
        logits = hidden
        outputs = {
            "logits": logits,
            "gat_proj": z_gat,
            "structure_proj": z_gat,
            "gagru_proj": z_gagru,
            "fused_repr": fused,
            "uniformity_loss": event_uniformity_loss(fused, self.uniformity_temperature),
            **temporal_report,
            **root_report,
        }
        if self.use_gat:
            outputs["branch_abs_cosine"] = F.cosine_similarity(
                z_gat,
                z_gagru,
                dim=-1,
            ).abs().mean()
            structure_logits = self.gat_classifier(z_gat)
            outputs["structure_logits"] = structure_logits
            outputs["gat_logits"] = structure_logits
        return outputs

    def classify(
        self,
        fused_repr: torch.Tensor,
        temporal_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the classifier stack to an already computed fused representation.

        Callers that reuse ``fused_repr`` (snapshot classification, branch
        contribution measurement) must go through this method rather than calling
        ``classifier`` directly, otherwise a temporal branch would be skipped.
        """
        hidden = self.classifier[2](self.classifier[1](self.classifier[0](fused_repr)))
        if self.temporal_dim > 0:
            if temporal_repr is None:
                raise ValueError(
                    "This fusion classifier carries a temporal branch, so classify() needs "
                    "the temporal representation of the same events."
                )
            hidden = hidden + self.temporal_head(temporal_repr)
        elif temporal_repr is not None:
            raise ValueError(
                "A temporal representation was supplied to a classifier without a "
                "temporal branch."
            )
        for layer in self.classifier[3:]:
            hidden = layer(hidden)
        return hidden


class DualBranchRumorDetector(nn.Module):
    def __init__(
        self,
        graph_input_dim: int,
        graph_hidden_dim: int = 128,
        fusion_dim: int = 256,
        graph_heads: int = 4,
        dropout: float = 0.2,
        input_bottleneck_dim: int = 0,
        num_edge_relations: int = 7,
        use_edge_relations: bool = True,
        use_semantic_edge_attention: bool = True,
        edge_dropout: float = 0.0,
        semantic_hidden_dim: int | None = None,
        semantic_projection_dim: int | None = None,
        semantic_loss_mse_weight: float = 0.1,
        semantic_target: str = "online",
        semantic_prediction_mode: str = "absolute",
        semantic_loss_criterion: str = "auto",
        semantic_sce_gamma: float = 1.0,
        node_representation: str = "independent",
        temporal_consistency: bool = False,
        root_feature_dim: int = 0,
        root_projection_dim: int = 0,
        gagru_direction: str = "bidirectional",
        gagru_aggregation: str = "attention",
        gagru_attention_query: str = "global",
        gagru_attention_score_mode: str = "dot",
        gagru_attention_residual_alpha: float = 1.0,
        gagru_attention_residual_mode: str = "fixed",
        gagru_attention_gate_init: float = 0.1,
        gagru_semantic_direction_reduction: str = "sum",
        gagru_direction_fusion: str = "joint",
        gagru_bu_residual_init: float = 0.1,
        gagru_bu_shared_gradient_scale: float = 1.0,
        gagru_readout: str = "mean",
        use_gat: bool = True,
        structure_encoder: str | None = None,
        gat_input_mode: str = "raw",
        gat_readout: str = "mean",
        gat_attention_type: str = "gat",
        gat_residual_mode: str = "fixed",
        gat_residual_alpha: float = 1.0,
        gat_neighborhood: str = "local",
        gat_semantic_k: int = 3,
        gat_global_event_node: bool = False,
        gat_source_evidence_pooling: bool = False,
        gat_depth_encoding: bool = False,
        gat_relative_time_bias: bool = False,
        acm_channels: str | tuple[str, ...] | None = None,
        sibling_interaction: bool = False,
        sibling_fusion: str = "node-residual",
        sibling_mode: str = "true",
        sibling_heads: int = 4,
        sibling_gate_init: float = 0.1,
        sibling_gate_mode: str = "learned",
        sibling_dim: int = 128,
        source_residual: bool = False,
        source_residual_gate_init: float = 0.2,
        contrastive_projection_dim: int = 0,
        gat_topology: str = "tree",
        set_transformer_num_latents: int = 8,
        set_transformer_num_layers: int = 2,
        global_semantic_mask_ratio: float = 0.25,
        global_semantic_sce_gamma: float = 2.0,
        global_semantic_target: str = "latent",
        global_semantic_variance_normalized_mse: bool = True,
        global_semantic_feature_std_floor: float = 0.05,
        global_recon_real_edges_only: bool = False,
        claim_guided_attention: bool = False,
        claim_attention_dim: int = 128,
        propagation_adapter: bool = False,
        propagation_adapter_bottleneck: int = 192,
        propagation_adapter_finetune: bool = False,
        masked_input_ratio: float = 0.0,
        temporal_dynamics: str = "off",
        temporal_evolution: str = "off",
        temporal_evolution_stages: int = 4,
        temporal_evolution_hidden: int = 128,
        temporal_evolution_dropout: float = 0.0,
        temporal_evolution_decay_init: float = -2.3,
        temporal_evolution_residual_zero_init: bool = True,
        temporal_evolution_final_stage_aux_only: bool = True,
        temporal_dim: int = 64,
        temporal_hidden: int = 64,
        temporal_channels: int = 32,
        temporal_kernel_size: int = 3,
        temporal_dropout: float = 0.0,
        temporal_normalize: bool = True,
        temporal_identity_init: bool = True,
        contrastive_views: bool = False,
        contrastive_temperature: float = 0.2,
        contrastive_node_drop_probability: float = 0.1,
        contrastive_attribute_mask_probability: float = 0.1,
        contrastive_max_probability: float = 0.5,
        uniformity_temperature: float = 2.0,
        joint_lora_model_name: Optional[str] = None,
        joint_lora_rank: int = 8,
        joint_lora_alpha: float = 16.0,
        joint_lora_dropout: float = 0.05,
        joint_lora_last_layers: int = 4,
        joint_lora_pooling: str = "mean",
        joint_lora_scope: str = "all",
        joint_lora_text_batch_size: int = 16,
        joint_lora_local_files_only: bool = True,
        joint_lora_adapter_checkpoint: Optional[str] = None,
        joint_lora_gradient_checkpointing: bool = False,
        joint_lora_use_amp: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if not 0.0 <= global_semantic_mask_ratio <= 1.0:
            raise ValueError("global_semantic_mask_ratio must be in [0, 1].")
        if input_bottleneck_dim < 0:
            raise ValueError("input_bottleneck_dim must be zero or positive.")
        semantic_hidden_dim = semantic_hidden_dim or graph_hidden_dim
        semantic_projection_dim = semantic_projection_dim or semantic_hidden_dim
        if structure_encoder is None:
            structure_encoder = "gat" if use_gat else "none"
        structure_encoder = structure_encoder.strip().lower().replace("_", "-")
        if structure_encoder not in {"gat", "acm-gcn", "set-transformer", "none"}:
            raise ValueError(
                "structure_encoder must be 'gat', 'acm-gcn', 'set-transformer', or 'none'."
            )
        # ACM replaces the attention stack inside the same structure branch, so it
        # shares every downstream plumbing decision with the GAT branch.
        self.adaptive_channels = (
            None
            if structure_encoder != "acm-gcn"
            else tuple(parse_acm_channels(acm_channels or DEFAULT_ACM_CHANNELS))
        )
        gat_input_mode = gat_input_mode.strip().lower().replace("_", "-")
        if gat_input_mode not in {"raw", "semantic-residual", "neighbor-residual"}:
            raise ValueError(
                "gat_input_mode must be 'raw', 'semantic-residual', or 'neighbor-residual'."
            )
        if gat_input_mode != "raw" and structure_encoder not in {"gat", "acm-gcn"}:
            raise ValueError(
                f"gat_input_mode='{gat_input_mode}' requires structure_encoder='gat' or 'acm-gcn'."
            )
        if gat_input_mode == "semantic-residual" and semantic_target != "raw":
            raise ValueError("gat_input_mode='semantic-residual' requires semantic_target='raw'.")
        if gat_input_mode != "raw" and global_semantic_target != "latent":
            raise ValueError(
                f"gat_input_mode='{gat_input_mode}' requires global_semantic_target='latent'."
            )
        gat_readout = gat_readout.strip().lower().replace("_", "-")
        if gat_readout not in {"mean", "propagation-change", "evidence"}:
            raise ValueError("gat_readout must be 'mean', 'propagation-change', or 'evidence'.")
        if gat_readout != "mean" and structure_encoder not in {"gat", "acm-gcn"}:
            raise ValueError(
                f"gat_readout='{gat_readout}' requires structure_encoder='gat' or 'acm-gcn'."
            )
        gat_attention_type = gat_attention_type.strip().lower().replace("_", "")
        if gat_attention_type not in {"gat", "gatv2"}:
            raise ValueError("gat_attention_type must be 'gat' or 'gatv2'.")
        gat_residual_mode = gat_residual_mode.strip().lower().replace("_", "-")
        if gat_residual_mode not in {"fixed", "learnable"}:
            raise ValueError("gat_residual_mode must be 'fixed' or 'learnable'.")
        if not 0.0 < gat_residual_alpha <= 1.0:
            raise ValueError("gat_residual_alpha must be in (0, 1].")
        gat_neighborhood = gat_neighborhood.strip().lower().replace("_", "-")
        if gat_neighborhood not in {"local", "ms-gat-v1"}:
            raise ValueError("gat_neighborhood must be 'local' or 'ms-gat-v1'.")
        if gat_semantic_k <= 0:
            raise ValueError("gat_semantic_k must be positive.")
        gat_topology = gat_topology.strip().lower().replace("_", "-")
        if gat_topology not in {"tree", "star", "set"}:
            raise ValueError("gat_topology must be 'tree', 'star', or 'set'.")
        if structure_encoder not in {"gat", "acm-gcn"}:
            if gat_attention_type != "gat":
                raise ValueError(
                    f"gat_attention_type='{gat_attention_type}' requires structure_encoder='gat'."
                )
            if gat_residual_mode != "fixed" or gat_residual_alpha != 1.0:
                raise ValueError(
                    "Non-default GAT residual settings require structure_encoder='gat'."
                )
        if structure_encoder == "acm-gcn" and (
            gat_attention_type != "gat"
            or gat_residual_mode != "fixed"
            or gat_residual_alpha != 1.0
        ):
            # ACM has no attention logits and uses the fixed residual path, so
            # accepting these settings would silently ignore them.
            raise ValueError(
                "ACM-GCN does not use gat_attention_type / gat_residual_mode; leave them "
                "at their defaults."
            )
        self.structure_encoder_type = structure_encoder
        self.use_gat = structure_encoder in {"gat", "acm-gcn"}
        self.use_structure_encoder = structure_encoder != "none"
        self.gat_input_mode = gat_input_mode
        self.gat_readout = gat_readout
        self.gat_attention_type = gat_attention_type
        self.gat_residual_mode = gat_residual_mode
        self.gat_residual_alpha = float(gat_residual_alpha)
        self.gat_neighborhood = gat_neighborhood
        self.gat_semantic_k = int(gat_semantic_k)
        self.gat_topology = gat_topology
        self.global_recon_real_edges_only = bool(global_recon_real_edges_only)
        self.gat_global_event_node = bool(gat_global_event_node)
        self.gat_source_evidence_pooling = bool(gat_source_evidence_pooling)
        self.gat_depth_encoding = bool(gat_depth_encoding)
        if self.gat_source_evidence_pooling and not self.gat_global_event_node:
            raise ValueError(
                "gat_source_evidence_pooling extends the global event node and requires gat_global_event_node=True."
            )
        self.gat_relative_time_bias = bool(gat_relative_time_bias)
        # V1G-R1: the fusion output becomes a logit correction on top of the
        # source-only decision instead of an independent full-event classifier.
        self.source_residual_enabled = bool(source_residual)
        # Cross-event SupCon: a train-only projection head on the fused event
        # representation. Inference never touches it, so deployment cost is zero.
        self.contrastive_projection_dim = int(contrastive_projection_dim)
        if self.contrastive_projection_dim > 0:
            self.contrastive_head = nn.Sequential(
                nn.Linear(2 * fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, self.contrastive_projection_dim),
            )
        if self.gat_global_event_node and self.gat_neighborhood != "ms-gat-v1":
            raise ValueError("gat_global_event_node requires gat_neighborhood='ms-gat-v1'.")
        if self.gat_depth_encoding and self.gat_neighborhood != "ms-gat-v1":
            raise ValueError("gat_depth_encoding requires gat_neighborhood='ms-gat-v1'.")
        if self.gat_relative_time_bias and self.gat_neighborhood != "ms-gat-v1":
            raise ValueError("gat_relative_time_bias requires gat_neighborhood='ms-gat-v1'.")
        if self.gat_topology == "set" and self.structure_encoder_type != "set-transformer":
            raise ValueError("gat_topology='set' requires structure_encoder='set-transformer'.")
        self.use_joint_lora = joint_lora_model_name is not None
        joint_lora_scope = joint_lora_scope.strip().lower()
        if joint_lora_scope not in {"all", "gat"}:
            raise ValueError("joint_lora_scope must be 'all' or 'gat'.")
        if self.use_joint_lora and joint_lora_scope == "gat" and not self.use_gat:
            raise ValueError("joint_lora_scope='gat' requires the GAT structure encoder.")
        self.joint_lora_scope = joint_lora_scope
        self.input_bottleneck_dim = int(input_bottleneck_dim)
        # Source-conditioned node representation: how the two text views of a
        # node (independent vs source/reply pair) become the vector that every
        # downstream branch consumes.
        node_representation = node_representation.strip().lower().replace("_", "-")
        if node_representation not in NodeRepresentationFusion.MODES:
            raise ValueError(
                f"node_representation must be one of {NodeRepresentationFusion.MODES}."
            )
        self.node_representation = node_representation
        self.node_fusion = (
            None
            if node_representation == "independent"
            else NodeRepresentationFusion(
                int(graph_input_dim), node_representation, dropout
            )
        )
        # Future-aware self-supervision: an early propagation stage predicts the
        # completed event's representation. Training-only; unused at inference.
        self.temporal_consistency_enabled = bool(temporal_consistency)
        self.temporal_consistency = (
            TemporalConsistencyHead(2 * fusion_dim, dropout)
            if self.temporal_consistency_enabled
            else None
        )
        self.branch_input_dim = (
            self.input_bottleneck_dim
            if self.input_bottleneck_dim > 0
            else graph_input_dim
        )
        if self.input_bottleneck_dim > 0:
            # A shared trainable bottleneck prevents either branch from reading the
            # high-dimensional frozen node representation directly.
            self.input_bottleneck = nn.Sequential(
                nn.Linear(graph_input_dim, self.input_bottleneck_dim),
                nn.LayerNorm(self.input_bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.input_bottleneck = None
        self.graph_output_dim = graph_hidden_dim
        self.semantic_output_dim = semantic_hidden_dim * 2
        if self.use_joint_lora:
            self.joint_text_encoder = JointLoRATextEncoder(
                model_name=str(joint_lora_model_name),
                rank=joint_lora_rank,
                alpha=joint_lora_alpha,
                dropout=joint_lora_dropout,
                last_layers=joint_lora_last_layers,
                pooling=joint_lora_pooling,
                text_batch_size=joint_lora_text_batch_size,
                local_files_only=joint_lora_local_files_only,
                adapter_checkpoint=joint_lora_adapter_checkpoint,
                gradient_checkpointing=joint_lora_gradient_checkpointing,
                use_amp=joint_lora_use_amp,
            )
            if self.joint_text_encoder.output_dim != graph_input_dim:
                raise ValueError(
                    "Online LoRA hidden size must match the stored graph feature dimension: "
                    f"online={self.joint_text_encoder.output_dim}, graph={graph_input_dim}."
                )
        # Sibling-set interaction: a residual on the GRAPH branch only.  The
        # GA-GRU branch keeps the unmodified node features, so the two branches
        # stay complementary; gamma_sib = 0 restores the exact V1G baseline.
        self.sibling_mode = str(sibling_mode)
        if sibling_interaction and self.structure_encoder_type not in {"gat", "acm-gcn"}:
            raise ValueError(
                "sibling_interaction requires a graph structure branch "
                "(structure_encoder 'gat' or 'acm-gcn')."
            )
        self.sibling_interaction_enabled = bool(sibling_interaction)
        self.sibling_fusion = str(sibling_fusion).strip().lower().replace("_", "-")
        if self.sibling_fusion not in {"node-residual", "event-concat"}:
            raise ValueError(
                "sibling_fusion must be 'node-residual' or 'event-concat'."
            )
        if self.sibling_interaction_enabled and self.sibling_fusion == "node-residual":
            self.sibling_sets = SiblingSetInteraction(
                dim=int(graph_hidden_dim),
                num_heads=int(sibling_heads),
                dropout=dropout,
                gate_init=float(sibling_gate_init),
                mode=self.sibling_mode,
                gate_mode=str(sibling_gate_mode),
            )
            self.sibling_event = None
        elif self.sibling_interaction_enabled:
            self.sibling_sets = None
            self.sibling_event = SiblingEventBranch(
                dim=int(graph_hidden_dim),
                num_heads=int(sibling_heads),
                dropout=dropout,
                output_dim=int(sibling_dim),
                mode=self.sibling_mode,
                gate_init=float(sibling_gate_init),
                gate_mode=str(sibling_gate_mode),
            )
        else:
            self.sibling_sets = None
            self.sibling_event = None
        if self.sibling_event is not None:
            # The sibling block is part of the classifier input, so it cannot be
            # bypassed by the optimiser the way a 0.1 node-level residual was.
            classifier_input_dim = 2 * fusion_dim + int(sibling_dim)
            self.sibling_classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Dropout(dropout),
                nn.Linear(classifier_input_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.sibling_classifier = None
        # ClaHi-style claim-guided evidence attention: z = [z_B || z_E], concatenated
        # rather than folded into the pooled representation (the earlier SCEP module
        # added its evidence view as a residual into the pooled state and failed).
        # M6 propagation-infused semantic adapter: x -> x' before either branch, so
        # the graph model receives propagation-aware semantics instead of raw BERT
        # vectors. Zero-initialized at the output layer, so x' = x at init and the
        # model starts bit-identical to the baseline.
        self.propagation_adapter_enabled = bool(propagation_adapter)
        self.propagation_adapter_finetune = bool(propagation_adapter_finetune)
        if self.propagation_adapter_enabled:
            self.propagation_adapter = PropagationSemanticAdapter(
                int(graph_input_dim), int(propagation_adapter_bottleneck)
            )
            if not self.propagation_adapter_finetune:
                for parameter in self.propagation_adapter.parameters():
                    parameter.requires_grad_(False)
        else:
            self.propagation_adapter = None
        # RAGCL-style same-event graph contrastive regularisation.  Training only:
        # the augmented views exist to shape the representation, and inference runs
        # the untouched baseline forward.
        self.contrastive_views_enabled = bool(contrastive_views)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_node_drop_probability = float(contrastive_node_drop_probability)
        self.contrastive_attribute_mask_probability = float(contrastive_attribute_mask_probability)
        self.contrastive_max_probability = float(contrastive_max_probability)
        if self.contrastive_views_enabled:
            if self.contrastive_temperature <= 0.0:
                raise ValueError("contrastive_temperature must be positive.")
            if not 0.0 <= self.contrastive_node_drop_probability <= 1.0:
                raise ValueError("contrastive node drop probability must be in [0, 1].")
            if not 0.0 <= self.contrastive_attribute_mask_probability <= 1.0:
                raise ValueError("contrastive attribute mask probability must be in [0, 1].")
        self.claim_attention_enabled = bool(claim_guided_attention)
        self.claim_attention_dim = int(claim_attention_dim)
        if self.claim_attention_enabled:
            if not self.use_gat:
                raise ValueError(
                    "claim_guided_attention requires a graph structure branch "
                    "(structure_encoder 'gat' or 'acm-gcn')."
                )
            if self.sibling_event is not None:
                raise ValueError(
                    "claim_guided_attention and sibling event-concat fusion both define "
                    "the classifier input; enable at most one."
                )
            self.claim_attention = ClaimGuidedEventAttention(
                graph_dim=self.graph_output_dim,
                semantic_dim=self.semantic_output_dim,
                hidden_dim=self.claim_attention_dim,
                dropout=dropout,
            )
            evidence_input_dim = 2 * fusion_dim + self.claim_attention_dim
            self.evidence_classifier = nn.Sequential(
                nn.LayerNorm(evidence_input_dim),
                nn.Dropout(dropout),
                nn.Linear(evidence_input_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.claim_attention = None
            self.evidence_classifier = None
        if self.use_gat:
            if self.gat_input_mode != "raw":
                self.gat_residual_norm = nn.LayerNorm(self.branch_input_dim)
            self.global_semantic = GlobalSemanticGAE(
                input_dim=self.branch_input_dim,
                hidden_dim=graph_hidden_dim,
                num_heads=graph_heads,
                dropout=dropout,
                num_edge_relations=num_edge_relations,
                use_edge_relations=use_edge_relations,
                use_semantic_edge_attention=use_semantic_edge_attention,
                edge_dropout=edge_dropout,
                mask_ratio=global_semantic_mask_ratio,
                sce_gamma=global_semantic_sce_gamma,
                target_mode=global_semantic_target,
                variance_normalized_mse=global_semantic_variance_normalized_mse,
                feature_std_floor=global_semantic_feature_std_floor,
                readout=self.gat_readout,
                attention_type=self.gat_attention_type,
                residual_mode=self.gat_residual_mode,
                residual_alpha=self.gat_residual_alpha,
                neighborhood=self.gat_neighborhood,
                semantic_k=self.gat_semantic_k,
                global_event_node=self.gat_global_event_node,
                source_evidence_pooling=self.gat_source_evidence_pooling,
                depth_encoding=self.gat_depth_encoding,
                relative_time_bias=self.gat_relative_time_bias,
                topology=self.gat_topology,
                adaptive_channels=self.adaptive_channels,
                sibling_sets=self.sibling_sets,
                sibling_event=self.sibling_event,
                reconstruction_real_edges_only=self.global_recon_real_edges_only,
            )
        elif self.structure_encoder_type == "set-transformer":
            self.global_set_transformer = GlobalSetTransformerEncoder(
                input_dim=self.branch_input_dim,
                hidden_dim=graph_hidden_dim,
                num_heads=graph_heads,
                num_latents=set_transformer_num_latents,
                num_layers=set_transformer_num_layers,
                dropout=dropout,
            )
        self.semantic_encoder = PropagationSemanticGRU(
            input_dim=self.branch_input_dim,
            hidden_dim=semantic_hidden_dim,
            dropout=dropout,
            semantic_dim=semantic_projection_dim,
            semantic_loss_mse_weight=semantic_loss_mse_weight,
            semantic_target=semantic_target,
            semantic_prediction_mode=semantic_prediction_mode,
            semantic_loss_criterion=semantic_loss_criterion,
            semantic_sce_gamma=semantic_sce_gamma,
            direction=gagru_direction,
            aggregation=gagru_aggregation,
            attention_query_mode=gagru_attention_query,
            attention_score_mode=gagru_attention_score_mode,
            attention_residual_alpha=gagru_attention_residual_alpha,
            attention_residual_mode=gagru_attention_residual_mode,
            attention_gate_init=gagru_attention_gate_init,
            semantic_direction_reduction=gagru_semantic_direction_reduction,
            direction_fusion=gagru_direction_fusion,
            bu_residual_init=gagru_bu_residual_init,
            bu_shared_gradient_scale=gagru_bu_shared_gradient_scale,
            readout=gagru_readout,
        )
        self.masked_input_ratio = float(masked_input_ratio)
        if not 0.0 <= self.masked_input_ratio < 1.0:
            raise ValueError("masked_input_ratio must be in [0, 1).")

        # M7 event-level temporal dynamics.  This is a new information source
        # (how the event grew over time), not an edge-level time bias: the failed
        # gat_relative_time_bias only modulated alpha_ij, whereas this branch reads
        # the whole response profile of the event.
        self.temporal_dynamics = str(temporal_dynamics).strip().lower()
        if self.temporal_dynamics in {"", "none", "off", "false"}:
            self.temporal_dynamics = "off"
        if self.temporal_dynamics != "off":
            raise ValueError(
                "The compact GA-GRU + GAT release does not include the discarded "
                "temporal-dynamics experiments. Use temporal_dynamics='off'."
            )
        if self.temporal_dynamics != "off" and not self.use_gat:
            raise ValueError(
                "The temporal dynamics branch expects the structure branch to be present."
            )
        if self.temporal_dynamics != "off":
            # Each of these installs its own readout on top of fused_repr, so the
            # temporal block would silently never reach the logits.
            for name, active in (
                ("claim_guided_attention", self.claim_attention is not None),
                ("sibling_interaction (event branch)", self.sibling_event is not None),
                ("source_residual", self.source_residual_enabled),
                ("temporal_consistency", self.temporal_consistency is not None),
            ):
                if active:
                    raise ValueError(
                        f"temporal_dynamics cannot be combined with {name}: both write to "
                        "the shared fusion classifier and one would be ignored."
                    )
            self.temporal_branch = TemporalDynamicsBranch(
                mode=self.temporal_dynamics,
                dim=int(temporal_dim),
                hidden=int(temporal_hidden),
                channels=int(temporal_channels),
                kernel_size=int(temporal_kernel_size),
                dropout=float(temporal_dropout),
                normalize=bool(temporal_normalize),
            )
        else:
            self.temporal_branch = None
        self.temporal_dim = int(temporal_dim) if self.temporal_branch is not None else 0
        self.temporal_identity_init = bool(temporal_identity_init)
        self.root_feature_dim = int(root_feature_dim)
        self.root_projection_dim = int(root_projection_dim)

        # TSE-V1G: the event becomes a sequence of cumulative propagation states,
        # all encoded by the *same* V1G + GA-GRU, and only a time-decayed residual
        # correction is allowed to change the full-stage prediction.
        self.temporal_evolution_mode = str(temporal_evolution).strip().lower()
        if self.temporal_evolution_mode in {"", "none", "off", "false"}:
            self.temporal_evolution_mode = "off"
        if self.temporal_evolution_mode not in ("off", "on"):
            raise ValueError(
                "temporal_evolution must be 'off' or 'on', got "
                f"{temporal_evolution!r}."
            )
        self.temporal_evolution_stages = int(temporal_evolution_stages)
        self.temporal_evolution_final_stage_aux_only = bool(
            temporal_evolution_final_stage_aux_only
        )
        self.temporal_evolution_enabled = self.temporal_evolution_mode == "on"
        if self.temporal_evolution_enabled:
            if self.temporal_branch is not None:
                raise ValueError(
                    "temporal_evolution and temporal_dynamics are two different answers to "
                    "the same question; enable only one of them."
                )
            if not self.use_gat:
                raise ValueError(
                    "temporal_evolution re-encodes each propagation stage with the "
                    "structure branch, so it requires the structure encoder."
                )
            if self.use_joint_lora:
                raise ValueError(
                    "temporal_evolution is not supported with joint LoRA: stage snapshots "
                    "would need per-stage token sidecars."
                )
            for name, active in (
                ("claim_guided_attention", self.claim_attention is not None),
                ("sibling_interaction (event branch)", self.sibling_event is not None),
                ("source_residual", self.source_residual_enabled),
                ("temporal_consistency", self.temporal_consistency is not None),
            ):
                if active:
                    raise ValueError(
                        f"temporal_evolution cannot be combined with {name}: both write to "
                        "the shared fusion/classifier and one would be ignored."
                    )
            self.temporal_evolution = TemporalStateEvolution(
                state_dim=2 * fusion_dim,
                hidden=int(temporal_evolution_hidden),
                num_stages=self.temporal_evolution_stages,
                dropout=float(temporal_evolution_dropout),
                decay_init=float(temporal_evolution_decay_init),
                residual_zero_init=bool(temporal_evolution_residual_zero_init),
            )
        else:
            self.temporal_evolution = None

        self.fusion = GATGAGRUFusionClassifier(
            gat_dim=self.graph_output_dim,
            gagru_dim=self.semantic_output_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            num_classes=num_classes,
            use_gat=self.use_structure_encoder,
            uniformity_temperature=uniformity_temperature,
            temporal_dim=self.temporal_dim,
            temporal_identity_init=self.temporal_identity_init,
            root_feature_dim=self.root_feature_dim,
            root_projection_dim=self.root_projection_dim,
        )
        if self.source_residual_enabled:
            self.source_residual_classifier = SourceResidualClassifier(
                source_dim=self.semantic_encoder.semantic_dim,
                propagation_dim=2 * fusion_dim,
                num_classes=num_classes,
                dropout=dropout,
                gate_init=source_residual_gate_init,
            )

    def _mask_classification_input(
        self,
        batched_graph: Dict[str, object],
        device: torch.device,
    ) -> tuple[Dict[str, object], float]:
        """M2a: hide part of the node semantics from the classifier itself.

        M2' masks a *separate* view that only feeds the reconstruction head, so its
        classification path never sees a masked node. This ablation masks the tensor the
        encoders consume, which is what separates "masking acts as a regulariser" from
        "the reconstruction objective is what helps".

        Returns ``(graph, realised_share)``. The graph comes back unchanged when the
        ablation is off or the model is evaluating, and as a shallow copy otherwise.
        """
        ratio = float(self.masked_input_ratio)
        if ratio <= 0.0 or not self.training:
            return batched_graph, 0.0
        features = batched_graph.get("node_features")
        if not torch.is_tensor(features) or features.numel() == 0:
            return batched_graph, 0.0
        mask = sample_event_masked_nodes(
            int(features.size(0)),
            batched_graph["root_indices"],
            batched_graph["graph_batch"],
            int(batched_graph["num_graphs"]),
            ratio,
            device,
        )
        if not bool(mask.any()):
            return batched_graph, 0.0
        masked_features = features.clone()
        masked_features[mask.to(features.device)] = 0.0
        masked_graph = dict(batched_graph)
        masked_graph["node_features"] = masked_features
        return masked_graph, float(mask.sum()) / float(features.size(0))

    def masked_input_view(
        self,
        batched_graph: Dict[str, object],
        ratio: float,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, object]]:
        """A copy of the graph whose classification input carries a semantic mask.

        Unlike ``_mask_classification_input`` this is deliberately not gated on training
        mode: the clean/mask consistency objective needs a masked view *and* the unmasked
        one for the same event. Returns ``None`` when nothing can be masked (only
        source-only events), so callers can skip the extra forward.
        """
        features = batched_graph.get("node_features")
        if not torch.is_tensor(features) or features.numel() == 0:
            return None
        if device is None:
            device = features.device
        mask = sample_event_masked_nodes(
            int(features.size(0)),
            batched_graph["root_indices"],
            batched_graph["graph_batch"],
            int(batched_graph["num_graphs"]),
            float(ratio),
            device,
        )
        if not bool(mask.any()):
            return None
        masked_features = features.clone()
        masked_features[mask.to(features.device)] = 0.0
        view = dict(batched_graph)
        view["node_features"] = masked_features
        return view

    def temporal_features(
        self,
        batched_graph: Dict[str, object],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Event-level temporal features for one batched graph.

        Reads only elapsed minutes and the real reply tree, so no label or fitted
        quantity enters here. Batch dictionaries carry tensors on the CPU, so every
        input is moved explicitly; otherwise the features would be built on the CPU
        and clash with the model's CUDA buffers during standardisation.
        """
        if self.temporal_branch is None:
            raise RuntimeError("Temporal dynamics is disabled on this model.")
        node_time = batched_graph.get("node_time")
        if node_time is None:
            raise ValueError(
                "The temporal dynamics branch needs node_time (elapsed minutes since the "
                "source post) inside the batch, but this graph does not provide it."
            )
        if device is None:
            device = next(self.parameters()).device
        return compute_temporal_features(
            node_time.to(device=device, dtype=torch.float32),
            batched_graph["graph_batch"].to(device=device, dtype=torch.long),
            batched_graph["edge_index_td"].to(device=device, dtype=torch.long),
            batched_graph["root_indices"].to(device=device, dtype=torch.long),
            int(batched_graph["num_graphs"]),
        )

    def temporal_features_from_batch(
        self,
        batch: Dict[str, object],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Temporal features for a training batch, used to fit train-split statistics.

        The statistics pass must stay on the CPU: it accumulates float64 sums over
        thousands of events and never touches the model, so it should not allocate
        GPU memory. Pass ``device=torch.device("cpu")`` there.
        """
        if "batched_graph" not in batch:
            raise ValueError("temporal_features_from_batch expects a batched_graph batch.")
        return self.temporal_features(batch["batched_graph"], device=device)

    def set_disabled_edge_families(self, families: Tuple[int, ...] = ()) -> None:
        """Drop edge families from the MS-GAT neighbourhood (0=real, 1=2-hop, 2=semantic).

        This is the *removal* direction: the real reply tree and the source-conditioned
        global event node stay untouched, and everything else (GA-GRU, fusion, classifier)
        is unaffected, so a run with families disabled isolates their contribution.
        """
        if not self.use_gat:
            if families:
                raise ValueError(
                    "Edge families only exist for the MS-GAT structure encoder, which is "
                    "disabled on this model."
                )
            return
        encoder = getattr(self.global_semantic, "encoder", None)
        if encoder is None or not hasattr(encoder, "set_disabled_edge_families"):
            raise RuntimeError("The structure encoder does not expose edge families.")
        encoder.set_disabled_edge_families(tuple(int(family) for family in families))

    @torch.no_grad()
    def set_temporal_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install the temporal feature mean/std fitted on the training split only."""
        if self.temporal_branch is None:
            raise RuntimeError("Cannot set temporal statistics when the branch is disabled.")
        self.temporal_branch.set_statistics(mean, std)

    @torch.no_grad()
    def set_global_semantic_feature_std(self, feature_std: torch.Tensor) -> None:
        if not self.use_gat:
            raise RuntimeError("Cannot set GAT reconstruction statistics when the GAT branch is disabled.")
        self.global_semantic.set_reconstruction_feature_std(feature_std)

    @torch.no_grad()
    def initialize_global_semantic_output(self, feature_mean: torch.Tensor) -> None:
        if not self.use_gat:
            raise RuntimeError("Cannot initialize GAT reconstruction when the GAT branch is disabled.")
        self.global_semantic.initialize_reconstruction_output(feature_mean)

    def set_semantic_reconstruction_enabled(self, enabled: bool) -> None:
        # semantic-residual GAT inputs consume predictions even without an auxiliary loss.
        self.semantic_encoder.reconstruction_enabled = bool(enabled or self.gat_input_mode == "semantic-residual")

    def set_global_semantic_reconstruction_enabled(self, enabled: bool) -> None:
        if self.use_gat:
            self.global_semantic.set_reconstruction_enabled(enabled)

    def _apply_propagation_adapter(
        self,
        structure_features: torch.Tensor,
        semantic_features: torch.Tensor,
        prediction_target_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Replace x with x' before either branch, consistently for all views.

        Applied to the structure stream, the semantic stream and the prediction
        target together, so the whole model sees one node space.
        """
        if self.propagation_adapter is None:
            return structure_features, semantic_features, prediction_target_features
        same_semantic = semantic_features.data_ptr() == structure_features.data_ptr()
        adapted_structure = self.propagation_adapter(structure_features)
        adapted_semantic = (
            adapted_structure
            if same_semantic
            else self.propagation_adapter(semantic_features)
        )
        adapted_target = None
        if prediction_target_features is not None:
            adapted_target = self.propagation_adapter(prediction_target_features)
            if prediction_target_features.requires_grad is False:
                adapted_target = adapted_target.detach()
        return adapted_structure, adapted_semantic, adapted_target

    def set_propagation_adapter_finetune(self, enabled: bool) -> None:
        """Toggle whether the adapter keeps training in stage 2."""
        if self.propagation_adapter is None:
            return
        self.propagation_adapter_finetune = bool(enabled)
        for parameter in self.propagation_adapter.parameters():
            parameter.requires_grad_(bool(enabled))

    def _constrain_node_features(
        self,
        structure_features: torch.Tensor,
        semantic_features: torch.Tensor,
        prediction_target_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.input_bottleneck is None:
            return structure_features, semantic_features, prediction_target_features
        same_structure_semantic = semantic_features.data_ptr() == structure_features.data_ptr()
        same_semantic_target = (
            prediction_target_features is not None
            and prediction_target_features.data_ptr() == semantic_features.data_ptr()
        )
        structure_features = self.input_bottleneck(structure_features)
        semantic_features = (
            structure_features
            if same_structure_semantic
            else self.input_bottleneck(semantic_features)
        )
        if prediction_target_features is not None:
            prediction_target_features = (
                semantic_features.detach()
                if same_semantic_target
                else self.input_bottleneck(prediction_target_features)
            )
        return structure_features, semantic_features, prediction_target_features

    def _resolve_node_features(
        self,
        stored_node_features: torch.Tensor,
        graph: Dict[str, object],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        stored_node_features = stored_node_features.to(device)
        if not self.use_joint_lora:
            semantic_node_features = graph.get("semantic_node_features")
            if semantic_node_features is None:
                structure_view, semantic_view, target_view = self._constrain_node_features(
                    stored_node_features,
                    stored_node_features,
                    None,
                )
                return self._apply_propagation_adapter(
                    structure_view, semantic_view, target_view
                )
            semantic_node_features = semantic_node_features.to(device)
            if semantic_node_features.shape != stored_node_features.shape:
                raise ValueError(
                    "Offline semantic node features must align with structure node features: "
                    f"structure={tuple(stored_node_features.shape)}, "
                    f"semantic={tuple(semantic_node_features.shape)}."
                )
            if self.node_fusion is not None:
                # Cross-encoder experiment: fuse both text views into ONE node
                # representation consumed by every branch, instead of feeding the
                # structure branch with one view and GA-GRU with the other.
                independent_view, relation_view, _ = self._constrain_node_features(
                    stored_node_features,
                    semantic_node_features,
                    None,
                )
                fused = self.node_fusion(independent_view, relation_view)
                return self._apply_propagation_adapter(fused, fused, fused.detach())
            structure_view, semantic_view, target_view = self._constrain_node_features(
                stored_node_features,
                semantic_node_features,
                semantic_node_features.detach(),
            )
            return self._apply_propagation_adapter(
                structure_view, semantic_view, target_view
            )
        if "node_input_ids" not in graph or "node_attention_mask" not in graph:
            raise KeyError(
                "Joint LoRA requires node_input_ids and node_attention_mask token sidecars."
            )
        input_ids = graph["node_input_ids"].to(device=device, dtype=torch.long)
        attention_mask = graph["node_attention_mask"].to(device=device, dtype=torch.long)
        online_features = self.joint_text_encoder(input_ids, attention_mask)
        if online_features.shape != stored_node_features.shape:
            raise ValueError(
                "Online LoRA node features do not align with the stored graph features: "
                f"online={tuple(online_features.shape)}, "
                f"stored={tuple(stored_node_features.shape)}."
            )
        # In GAT-only mode the semantic branch remains a stable frozen-BERT view.
        # This prevents GA-GRU prediction supervision from updating LoRA and makes
        # the two branches complementary instead of feeding both the same online view.
        semantic_features = (
            stored_node_features
            if self.joint_lora_scope == "gat"
            else online_features
        )
        structure_view, semantic_view, target_view = self._constrain_node_features(
            online_features,
            semantic_features,
            stored_node_features.detach(),
        )
        return self._apply_propagation_adapter(
            structure_view, semantic_view, target_view
        )

    def set_claim_evidence_enabled(self, enabled: bool) -> None:
        """Diagnostic switch: zero the claim-guided evidence block at inference.

        Same classifier weights, evidence removed, which isolates the contribution
        of the evidence view from the effect of adding a classifier head.
        """
        if self.claim_attention is not None:
            self.claim_attention.set_evidence_enabled(enabled)

    def set_sibling_gate(self, value) -> None:
        """Diagnostic switch: force gamma_sib (0.0 switches the branch off).

        Both sibling pathways must be covered: missing the event-concat one made
        the zero-sibling control silently do nothing, which would invalidate the
        Full-vs-Zero comparison.
        """
        for module in (self.sibling_sets, self.sibling_event):
            if module is not None:
                module.set_gate(value)

    def _raw_root_features(
        self,
        batched_graph: Dict[str, object],
        root_indices: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """The source post's own input feature, gathered per event, or None when disabled.

        Deliberately reads the *stored* features rather than a branch view: the root enhancement
        is meant to carry the claim's own representation, and the masking rule never touches the
        root node, so this is the unmasked source feature. `root_indices` is one entry per graph,
        so the result is (num_graphs, feature_dim) in graph order, exactly like the branch
        readouts it is concatenated with.
        """
        if self.root_feature_dim <= 0:
            return None
        features = batched_graph["node_features"]
        if not isinstance(features, torch.Tensor):  # pragma: no cover - defensive
            raise TypeError("batched_graph['node_features'] must be a tensor.")
        return features.to(device)[root_indices.to(device=device, dtype=torch.long)]

    def _source_root_representation(
        self,
        semantic_node_features: torch.Tensor,
        root_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project source posts through the existing GA-GRU semantic encoder.

        ``h_s = semantic_proj(x_0)`` reuses the same projection the semantic
        branch applies to every node, so the source head shares the established
        semantic space instead of growing a separate encoder.
        """
        roots = root_indices.to(
            device=semantic_node_features.device, dtype=torch.long
        ).reshape(-1)
        return self.semantic_encoder.semantic_proj(semantic_node_features[roots])

    def _gat_node_features(
        self,
        structure_node_features: torch.Tensor,
        semantic_outputs: Dict[str, torch.Tensor],
        edge_index_td: torch.Tensor,
    ) -> torch.Tensor:
        if self.gat_input_mode == "raw":
            return structure_node_features
        if self.gat_input_mode == "neighbor-residual":
            num_nodes = structure_node_features.size(0)
            edge_index_td = edge_index_td.to(
                device=structure_node_features.device,
                dtype=torch.long,
            )
            neighbor_sum = torch.zeros_like(structure_node_features)
            neighbor_count = structure_node_features.new_zeros((num_nodes, 1))
            if edge_index_td.numel() > 0:
                edge_src = edge_index_td[0]
                edge_dst = edge_index_td[1]
                valid_edges = (
                    (edge_src >= 0)
                    & (edge_src < num_nodes)
                    & (edge_dst >= 0)
                    & (edge_dst < num_nodes)
                    & (edge_src != edge_dst)
                )
                edge_src = edge_src[valid_edges]
                edge_dst = edge_dst[valid_edges]
                if edge_src.numel() > 0:
                    neighbor_sum.index_add_(0, edge_dst, structure_node_features[edge_src])
                    neighbor_sum.index_add_(0, edge_src, structure_node_features[edge_dst])
                    ones = structure_node_features.new_ones((edge_src.numel(), 1))
                    neighbor_count.index_add_(0, edge_dst, ones)
                    neighbor_count.index_add_(0, edge_src, ones)

            neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0)
            residual_features = structure_node_features - neighbor_mean
            residual_features = residual_features * (neighbor_count > 0).to(
                dtype=structure_node_features.dtype
            )
            return self.gat_residual_norm(residual_features)

        targets = semantic_outputs["target_semantics"].detach()
        if targets.shape != structure_node_features.shape:
            raise ValueError(
                "Semantic-residual GAT input requires raw GA-GRU targets aligned with "
                "the structure node features."
            )

        residual_sum = torch.zeros_like(targets)
        residual_count = targets.new_zeros((targets.size(0), 1))
        for prediction_key, mask_key in (
            ("child_semantic_predictions", "child_prediction_mask"),
            ("parent_semantic_predictions", "parent_prediction_mask"),
        ):
            predictions = semantic_outputs[prediction_key].detach()
            mask = semantic_outputs[mask_key].to(device=targets.device, dtype=torch.bool)
            if predictions.shape != targets.shape or mask.shape != targets.shape[:1]:
                raise ValueError("GA-GRU prediction outputs do not align with raw node targets.")
            mask_column = mask.unsqueeze(-1).to(dtype=targets.dtype)
            residual_sum = residual_sum + (targets - predictions) * mask_column
            residual_count = residual_count + mask_column

        residual_features = residual_sum / residual_count.clamp_min(1.0)
        return self.gat_residual_norm(residual_features)

    def _resolve_topology(
        self,
        edge_index_td: torch.Tensor,
        edge_index_bu: torch.Tensor,
        edge_type_td: torch.Tensor,
        edge_type_bu: torch.Tensor,
        root_indices: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tree edges or a source-centered star for every event.

        The star graph deliberately removes every original reply-to-reply parent
        relation.  It is shared by GAT and GA-GRU so the no-parent experiment
        cannot retain hierarchy through the semantic branch.
        """
        if self.gat_topology == "tree":
            return edge_index_td, edge_index_bu, edge_type_td, edge_type_bu

        if self.gat_topology == "set":
            device = graph_batch.device
            empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_types = torch.empty((0,), dtype=torch.long, device=device)
            return empty_edges, empty_edges.clone(), empty_types, empty_types.clone()

        device = graph_batch.device
        roots = root_indices.to(device=device, dtype=torch.long).reshape(-1)
        owners = graph_batch.to(device=device, dtype=torch.long)
        if roots.numel() != num_graphs:
            raise ValueError("Star topology requires one source root per graph.")

        td_parts: list[torch.Tensor] = []
        bu_parts: list[torch.Tensor] = []
        for graph_index, root in enumerate(roots.tolist()):
            if root < 0 or root >= owners.numel() or int(owners[root].item()) != graph_index:
                raise ValueError("Each star-graph root must belong to its graph.")
            replies = torch.nonzero(owners == graph_index, as_tuple=False).flatten()
            replies = replies[replies != root]
            if replies.numel() == 0:
                continue
            source = torch.full_like(replies, root)
            td_parts.append(torch.stack([source, replies], dim=0))
            bu_parts.append(torch.stack([replies, source], dim=0))

        if td_parts:
            star_td = torch.cat(td_parts, dim=1)
            star_bu = torch.cat(bu_parts, dim=1)
        else:
            star_td = torch.empty((2, 0), dtype=torch.long, device=device)
            star_bu = torch.empty((2, 0), dtype=torch.long, device=device)
        # Relation labels carry no parent/child information in the star setting.
        star_type_td = torch.zeros(star_td.size(1), dtype=torch.long, device=device)
        star_type_bu = torch.zeros(star_bu.size(1), dtype=torch.long, device=device)
        return star_td, star_bu, star_type_td, star_type_bu

    def _forward_batched(self, batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
        batched_graph = batch["batched_graph"]
        batched_graph, masked_input_share = self._mask_classification_input(
            batched_graph, device
        )
        structure_node_features, semantic_node_features, prediction_target_features = self._resolve_node_features(
            batched_graph["node_features"],
            batched_graph,
            device,
        )
        edge_index_td = batched_graph["edge_index_td"].to(device)
        edge_index_bu = batched_graph["edge_index_bu"].to(device)
        edge_type_td = batched_graph["edge_type_td"].to(device)
        edge_type_bu = batched_graph["edge_type_bu"].to(device)
        root_indices = batched_graph["root_indices"].to(device)
        graph_batch = batched_graph["graph_batch"].to(device)
        node_time = batched_graph.get("node_time")
        if node_time is not None:
            node_time = node_time.to(device=device, dtype=structure_node_features.dtype)
        num_graphs = int(batched_graph["num_graphs"])
        # M7 temporal dynamics reads the real reply tree and elapsed minutes
        # directly from the batch, i.e. before any topology override, because the
        # temporal profile describes the actual propagation rather than the graph
        # variant the encoder happens to use.
        temporal_outputs: Dict[str, torch.Tensor] = {}
        temporal_repr: Optional[torch.Tensor] = None
        if self.temporal_branch is not None:
            temporal_outputs = self.temporal_branch(
                self.temporal_features(batched_graph, device)
            )
            temporal_repr = temporal_outputs["temporal_repr"]
        edge_index_td, edge_index_bu, edge_type_td, edge_type_bu = self._resolve_topology(
            edge_index_td,
            edge_index_bu,
            edge_type_td,
            edge_type_bu,
            root_indices,
            graph_batch,
            num_graphs,
        )

        semantic_outputs = self.semantic_encoder.forward_batched(
            node_features=semantic_node_features,
            edge_index_td=edge_index_td,
            root_indices=root_indices,
            graph_batch=graph_batch,
            num_graphs=num_graphs,
            prediction_target_features=prediction_target_features,
        )
        semantic_repr = semantic_outputs["pooled"]

        if self.use_gat:
            gat_node_features = self._gat_node_features(
                structure_node_features,
                semantic_outputs,
                edge_index_td,
            )
            global_outputs = self.global_semantic(
                gat_node_features,
                edge_index_td,
                edge_index_bu,
                edge_type_td,
                edge_type_bu,
                root_indices,
                graph_batch=graph_batch,
                num_graphs=num_graphs,
                node_time=node_time,
            )
            structure_only_repr = global_outputs["pooled"]
        elif self.structure_encoder_type == "set-transformer":
            global_outputs = {}
            set_outputs = self.global_set_transformer(
                node_features=structure_node_features,
                root_indices=root_indices,
                graph_batch=graph_batch,
                num_graphs=num_graphs,
            )
            structure_only_repr = set_outputs["pooled"]
        else:
            global_outputs = {}
            structure_only_repr = semantic_repr.new_zeros((num_graphs, self.graph_output_dim))

        fusion_outputs = self.fusion(
            structure_only_repr, semantic_repr, temporal_repr,
            root_features=self._raw_root_features(batched_graph, root_indices, device),
        )
        temporal_report: Dict[str, torch.Tensor] = build_temporal_report(
            temporal_outputs, fusion_outputs
        )
        source_residual_outputs: Dict[str, torch.Tensor] = {}
        if self.claim_attention is not None:
            evidence = self.claim_attention(
                global_outputs["node_states"],
                semantic_outputs["node_states"],
                root_indices,
                graph_batch,
                num_graphs,
            )
            logits = self.evidence_classifier(
                torch.cat([fusion_outputs["fused_repr"], evidence], dim=-1)
            )
            outputs_evidence = {
                "claim_evidence_repr": evidence,
                "claim_evidence_norm": evidence.norm(dim=-1).mean(),
                **self.claim_attention.diagnostics(),
            }
        else:
            outputs_evidence = {}
        sibling_event_repr = global_outputs.get("sibling_event_repr")
        sibling_relative_norm = global_outputs.get("sibling_relative_norm")
        if self.claim_attention is not None:
            pass
        elif self.sibling_classifier is not None and sibling_event_repr is not None:
            logits = self.sibling_classifier(
                torch.cat([fusion_outputs["fused_repr"], sibling_event_repr], dim=-1)
            )
            if sibling_relative_norm is None:
                # Event-concat analogue of R_sib: how large the appended sibling
                # block is relative to the V1G representation it is appended to.
                sibling_relative_norm = sibling_event_repr.norm(dim=-1) / (
                    fusion_outputs["fused_repr"].norm(dim=-1) + 1e-8
                )
                sibling_relative_norm = sibling_relative_norm.mean()
        elif self.source_residual_enabled:
            source_repr = self._source_root_representation(
                semantic_node_features,
                root_indices,
            )
            source_residual_outputs = self.source_residual_classifier(
                source_repr, fusion_outputs["fused_repr"]
            )
            logits = source_residual_outputs["logits"]
        else:
            logits = fusion_outputs["logits"]
        outputs = {
            "logits": logits,
            "fusion_logits": fusion_outputs["logits"],
            "fused_repr": fusion_outputs["fused_repr"],
            "semantic_repr": semantic_repr,
            **({"masked_input_share": masked_input_share} if masked_input_share else {}),
            **temporal_report,
            "structure_only_repr": structure_only_repr,
            "semantic_loss": semantic_outputs["semantic_loss"],
            "semantic_prediction_outputs": [
                {
                    "target_semantics": semantic_outputs["target_semantics"],
                    "child_semantic_predictions": semantic_outputs["child_semantic_predictions"],
                    "parent_semantic_predictions": semantic_outputs["parent_semantic_predictions"],
                    "child_edge_semantic_predictions": semantic_outputs[
                        "child_edge_semantic_predictions"
                    ],
                    "parent_edge_semantic_predictions": semantic_outputs[
                        "parent_edge_semantic_predictions"
                    ],
                    "semantic_edge_index": semantic_outputs["semantic_edge_index"],
                    "child_semantic_loss": semantic_outputs["child_semantic_loss"],
                    "parent_semantic_loss": semantic_outputs["parent_semantic_loss"],
                    "child_semantic_cosine_gap": semantic_outputs["child_semantic_cosine_gap"],
                    "parent_semantic_cosine_gap": semantic_outputs["parent_semantic_cosine_gap"],
                    "child_prediction_mask": semantic_outputs["child_prediction_mask"],
                    "parent_prediction_mask": semantic_outputs["parent_prediction_mask"],
                }
            ],
            "gat_proj": fusion_outputs["gat_proj"],
            "structure_proj": fusion_outputs["structure_proj"],
            "gagru_proj": fusion_outputs["gagru_proj"],
            "uniformity_loss": fusion_outputs["uniformity_loss"],
            "gat_disabled": not self.use_gat,
            "structure_disabled": not self.use_structure_encoder,
            "structure_encoder": self.structure_encoder_type,
            "gat_input_mode": self.gat_input_mode,
            "gat_readout": self.gat_readout,
            "gat_attention_type": self.gat_attention_type,
            "gat_residual_mode": self.gat_residual_mode,
            "gat_residual_alpha": self.gat_residual_alpha,
            "gat_neighborhood": self.gat_neighborhood,
            "gat_semantic_k": self.gat_semantic_k,
            "gat_topology": self.gat_topology,
            "gat_global_event_node": self.gat_global_event_node,
            "gat_source_evidence_pooling": self.gat_source_evidence_pooling,
            "gat_depth_encoding": self.gat_depth_encoding,
            "gat_relative_time_bias": self.gat_relative_time_bias,
            "joint_lora_enabled": self.use_joint_lora,
            "joint_lora_scope": self.joint_lora_scope,
            **outputs_evidence,
        }
        if self.use_gat and "neighborhood_edge_counts" in global_outputs:
            outputs["gat_neighborhood_edge_counts"] = global_outputs["neighborhood_edge_counts"]
        for diagnostic in (
            "time_gate_mean",
            "time_gate_per_head",
            "mean_abs_time_bias_semantic",
        ):
            if diagnostic in global_outputs:
                outputs[diagnostic] = global_outputs[diagnostic]
        for diagnostic in (
            "sibling_gate",
            "sibling_attention_entropy",
            "sibling_group_count",
            "sibling_gated_node_share",
            "sibling_attention_self_weight",
            "sibling_relative_norm",
            "sibling_event_repr",
            "sibling_event_norm",
        ):
            if diagnostic in global_outputs:
                outputs[diagnostic] = global_outputs[diagnostic]
        if sibling_relative_norm is not None:
            outputs["sibling_relative_norm"] = sibling_relative_norm
        if "acm_channel_names" in global_outputs:
            outputs["acm_channel_names"] = global_outputs["acm_channel_names"]
            outputs["acm_gate_per_layer"] = global_outputs["acm_gate_per_layer"]
            outputs["acm_zero_channel_gate_mass"] = global_outputs[
                "acm_zero_channel_gate_mass"
            ]
            for name in global_outputs["acm_channel_names"]:
                outputs[f"acm_gate_{name}"] = global_outputs[f"acm_gate_{name}"]
        if "reconstruction_loss" in global_outputs:
            outputs["global_semantic_loss"] = global_outputs["reconstruction_loss"]
            outputs["global_semantic_masked_count_mean"] = global_outputs["masked_count"] / max(num_graphs, 1)
            outputs["global_semantic_prediction_outputs"] = [global_outputs["reconstruction"]]
        if "structure_logits" in fusion_outputs:
            outputs["structure_logits"] = fusion_outputs["structure_logits"]
        if "branch_abs_cosine" in fusion_outputs:
            outputs["branch_abs_cosine"] = fusion_outputs["branch_abs_cosine"]
        if self.use_gat and "gat_logits" in fusion_outputs:
            outputs["gat_logits"] = fusion_outputs["gat_logits"]
        if source_residual_outputs:
            outputs["source_logits"] = source_residual_outputs["source_logits"]
            outputs["delta_logits"] = source_residual_outputs["delta_logits"]
            outputs["source_residual_gamma"] = source_residual_outputs["gamma"]
        if self.contrastive_projection_dim > 0:
            outputs["contrastive_repr"] = F.normalize(
                self.contrastive_head(fusion_outputs["fused_repr"]), dim=-1
            )
        return outputs

    def _add_contrastive_loss(
        self,
        outputs: Dict[str, object],
        batch: Dict[str, object],
        device: torch.device,
    ) -> None:
        """Same-event InfoNCE between two adaptively augmented views.

        Both views go through the *shared* encoder (a direct ``_forward_batched``
        call, so the views cannot recursively build their own views).  Labels are
        never used to form positives: the only positive for a view is the other
        view of the same event, which is what separates this from the supervised
        contrastive objective that already failed on this dataset.
        """
        if not self.contrastive_views_enabled:
            return
        batched_graph = batch.get("batched_graph")
        if batched_graph is None:
            return
        num_graphs = int(batched_graph["num_graphs"])
        if num_graphs < 2:
            # InfoNCE needs at least one negative.
            return
        view_drop, view_mask = build_contrastive_views(
            batched_graph,
            num_graphs,
            node_drop_probability=self.contrastive_node_drop_probability,
            attribute_mask_probability=self.contrastive_attribute_mask_probability,
            max_probability=self.contrastive_max_probability,
        )
        first = self._forward_batched({"batched_graph": view_drop}, device)["fused_repr"]
        second = self._forward_batched({"batched_graph": view_mask}, device)["fused_repr"]
        loss, positive_cosine, negative_cosine = symmetric_infonce(
            first, second, self.contrastive_temperature
        )
        outputs["contrastive_loss"] = loss
        outputs["contrastive_positive_cosine"] = positive_cosine
        outputs["contrastive_negative_cosine"] = negative_cosine
        outputs["contrastive_gap"] = positive_cosine - negative_cosine

    def _add_temporal_consistency(
        self,
        outputs: Dict[str, object],
        batch: Dict[str, object],
        device: torch.device,
    ) -> None:
        """Run the same encoder on the early stage and score future consistency."""
        snapshot_graph = batch.get("snapshot_batched_graph")
        if snapshot_graph is None or self.temporal_consistency is None:
            return
        snapshot_outputs = self._forward_batched(
            {"batched_graph": snapshot_graph}, device
        )
        early_repr = snapshot_outputs["fused_repr"]
        full_repr = outputs["fused_repr"]
        outputs["future_loss"] = self.temporal_consistency(early_repr, full_repr)
        # The early stage is classified by the same classifier, never by a copy.
        outputs["snapshot_logits"] = self.fusion.classify(early_repr)
        outputs["snapshot_fused_repr"] = early_repr
        outputs["temporal_consistency"] = True

    @contextlib.contextmanager
    def _stage_auxiliary_losses(self, enabled: bool):
        """Enable or mute the global reconstruction for one snapshot forward.

        Intermediate stages only exist to produce states, so by default the M2
        objective is computed on the full stage alone, exactly as it is in M2.
        """
        module = getattr(self, "global_semantic", None)
        if module is None or not hasattr(module, "mask_ratio"):
            yield
            return
        saved = module.mask_ratio
        if not enabled:
            module.mask_ratio = 0.0
        try:
            yield
        finally:
            module.mask_ratio = saved

    def _forward_temporal_evolution(
        self,
        batch: Dict[str, object],
        device: torch.device,
    ) -> Dict[str, object]:
        """TSE-V1G: encode every cumulative propagation stage, then evolve them.

        The final stage is the complete event, so ``s_K`` is exactly what an M0
        forward produces and the residual starts as a no-op.
        """
        if self.temporal_evolution is None:
            raise RuntimeError("Temporal evolution is disabled on this model.")
        batched_graph = batch["batched_graph"]
        if "node_time" not in batched_graph:
            raise ValueError(
                "Temporal evolution needs node_time (elapsed minutes since the source "
                "post) to order the stages; this dataset does not provide it."
            )
        # Batch dictionaries carry CPU tensors, but the snapshot builder allocates
        # its index tensors on the device of its inputs; move the graph first so the
        # stage deltas end up next to the states they gate.
        graph_on_device = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batched_graph.items()
        }
        snapshots = build_cumulative_snapshots(
            graph_on_device, self.temporal_evolution_stages
        )
        states: List[torch.Tensor] = []
        full_outputs: Dict[str, object] = {}
        stages = snapshots["stages"]
        for index, stage_graph in enumerate(stages):
            is_full_stage = index == len(stages) - 1
            keep_aux = is_full_stage or not self.temporal_evolution_final_stage_aux_only
            with self._stage_auxiliary_losses(keep_aux):
                stage_outputs = self._forward_batched(
                    {"batched_graph": stage_graph}, device
                )
            states.append(stage_outputs["fused_repr"])
            if is_full_stage:
                full_outputs = stage_outputs

        evolution = self.temporal_evolution(states, snapshots["delta_minutes"])
        z_full = full_outputs["fused_repr"]
        outputs = dict(full_outputs)
        # fusion_logits is the untouched M0 head on the complete event, so the
        # dynamics have to earn every point of difference against it.
        outputs["fusion_logits"] = full_outputs["logits"]
        outputs["fused_repr"] = z_full
        outputs["temporal_final_repr"] = evolution["final"]
        outputs["logits"] = self.fusion.classify(evolution["final"])
        outputs["temporal_evolution"] = True
        outputs["temporal_stage_sizes"] = snapshots["stage_sizes"].to(torch.float32).mean(dim=0)
        outputs["temporal_stage_delta_minutes"] = snapshots["delta_minutes"].mean(dim=0)
        outputs["temporal_reply_count_mean"] = snapshots["reply_count"].to(
            torch.float32
        ).mean()
        outputs["temporal_decay_mean"] = evolution["decay_mean"]
        outputs["temporal_decay_rate"] = evolution["decay_rate"]
        outputs["temporal_gate_mean"] = evolution["gate_mean"]
        outputs["temporal_residual_ratio"] = evolution["residual_ratio"]
        outputs["temporal_drift_ratio"] = evolution["drift_ratio"]
        return outputs

    def forward(self, batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
        if "batched_graph" in batch:
            if self.temporal_evolution_enabled:
                return self._forward_temporal_evolution(batch, device)
            outputs = self._forward_batched(batch, device)
            if "snapshot_batched_graph" in batch:
                self._add_temporal_consistency(outputs, batch, device)
            # Training-time only: the callers disable it for evaluation.
            if self.contrastive_views_enabled and self.training:
                self._add_contrastive_loss(outputs, batch, device)
            return outputs

        graph_vectors: List[torch.Tensor] = []
        semantic_vectors: List[torch.Tensor] = []
        source_vectors: List[torch.Tensor] = []
        root_feature_blocks: List[torch.Tensor] = []
        semantic_losses: List[torch.Tensor] = []
        semantic_prediction_outputs: List[Dict[str, torch.Tensor]] = []
        global_semantic_losses: List[torch.Tensor] = []
        global_semantic_masked_counts: List[torch.Tensor] = []
        global_semantic_prediction_outputs: List[Dict[str, torch.Tensor]] = []
        neighborhood_edge_counts: List[torch.Tensor] = []
        acm_gate_means: List[Dict[str, torch.Tensor]] = []
        sibling_reports: List[Dict[str, float]] = []
        sibling_event_reprs: List[torch.Tensor] = []
        graph_node_states: List[torch.Tensor] = []
        semantic_node_states: List[torch.Tensor] = []
        node_owner: List[torch.Tensor] = []
        acm_zero_masses: List[torch.Tensor] = []
        time_gate_means: List[torch.Tensor] = []
        time_gate_heads: List[torch.Tensor] = []
        time_bias_means: List[torch.Tensor] = []
        temporal_feature_blocks: List[torch.Tensor] = []
        for graph in batch["graphs"]:
            if self.masked_input_ratio > 0.0 and self.training:
                # Single-graph layouts name their fields differently, so adapt before
                # reusing the M2a masking helper.
                single_nodes = int(graph["node_features"].size(0))
                masked_graph, _share = self._mask_classification_input(
                    {
                        "node_features": graph["node_features"],
                        "root_indices": graph["root_index"].reshape(-1),
                        "graph_batch": torch.zeros(
                            single_nodes, dtype=torch.long, device=device
                        ),
                        "num_graphs": 1,
                    },
                    device,
                )
                graph = {**graph, "node_features": masked_graph["node_features"]}
                masked_input_share = max(masked_input_share, _share)
            structure_node_features, semantic_node_features, prediction_target_features = self._resolve_node_features(
                graph["node_features"],
                graph,
                device,
            )
            edge_index_td = graph["edge_index_td"].to(device)
            edge_index_bu = graph["edge_index_bu"].to(device)
            root_index = graph["root_index"].to(device)
            if self.root_feature_dim > 0:
                # Same quantity as the batched path: the stored (never masked) source row.
                root_feature_blocks.append(
                    graph["node_features"].to(device)[root_index.reshape(-1)[:1]]
                )
            edge_type_td = graph["edge_type_td"].to(device)
            edge_type_bu = graph["edge_type_bu"].to(device)
            if self.temporal_branch is not None:
                # Uses the untouched reply tree, before _resolve_topology below.
                if "node_time" not in graph:
                    raise ValueError(
                        "The temporal dynamics branch needs node_time (elapsed minutes "
                        "since the source post) in every graph."
                    )
                num_single_nodes = structure_node_features.size(0)
                temporal_feature_blocks.append(
                    compute_temporal_features(
                        graph["node_time"].to(device=device, dtype=torch.float32),
                        torch.zeros(num_single_nodes, dtype=torch.long, device=device),
                        edge_index_td,
                        root_index.reshape(1),
                        1,
                    )["flat"]
                )
            single_graph_batch = torch.zeros(
                structure_node_features.size(0), dtype=torch.long, device=device
            )
            edge_index_td, edge_index_bu, edge_type_td, edge_type_bu = self._resolve_topology(
                edge_index_td,
                edge_index_bu,
                edge_type_td,
                edge_type_bu,
                root_index.reshape(1),
                single_graph_batch,
                1,
            )
            semantic_outputs = self.semantic_encoder(
                node_features=semantic_node_features,
                edge_index_td=edge_index_td,
                root_index=root_index,
                prediction_target_features=prediction_target_features,
            )
            if self.source_residual_enabled:
                source_vectors.append(
                    self._source_root_representation(
                        semantic_node_features, root_index.reshape(1)
                    ).squeeze(0)
                )
            semantic_vectors.append(semantic_outputs["pooled"])
            semantic_losses.append(semantic_outputs["semantic_loss"])
            semantic_prediction_outputs.append(
                {
                    "target_semantics": semantic_outputs["target_semantics"],
                    "child_semantic_predictions": semantic_outputs["child_semantic_predictions"],
                    "parent_semantic_predictions": semantic_outputs["parent_semantic_predictions"],
                    "child_edge_semantic_predictions": semantic_outputs[
                        "child_edge_semantic_predictions"
                    ],
                    "parent_edge_semantic_predictions": semantic_outputs[
                        "parent_edge_semantic_predictions"
                    ],
                    "semantic_edge_index": semantic_outputs["semantic_edge_index"],
                    "child_semantic_loss": semantic_outputs["child_semantic_loss"],
                    "parent_semantic_loss": semantic_outputs["parent_semantic_loss"],
                    "child_semantic_cosine_gap": semantic_outputs["child_semantic_cosine_gap"],
                    "parent_semantic_cosine_gap": semantic_outputs["parent_semantic_cosine_gap"],
                    "child_prediction_mask": semantic_outputs["child_prediction_mask"],
                    "parent_prediction_mask": semantic_outputs["parent_prediction_mask"],
                }
            )
            if self.use_gat:
                gat_node_features = self._gat_node_features(
                    structure_node_features,
                    semantic_outputs,
                    edge_index_td,
                )
                global_outputs = self.global_semantic(
                    gat_node_features,
                    edge_index_td,
                    edge_index_bu,
                    edge_type_td,
                    edge_type_bu,
                    root_index,
                    node_time=(
                        graph["node_time"].to(device=device, dtype=gat_node_features.dtype)
                        if "node_time" in graph else None
                    ),
                )
                graph_vectors.append(global_outputs["pooled"])
                if self.claim_attention is not None:
                    graph_node_states.append(global_outputs["node_states"])
                    semantic_node_states.append(semantic_outputs["node_states"])
                    node_owner.append(
                        torch.full(
                            (global_outputs["node_states"].size(0),),
                            len(graph_node_states) - 1,
                            dtype=torch.long,
                            device=global_outputs["node_states"].device,
                        )
                    )
                neighborhood_edge_counts.append(global_outputs["neighborhood_edge_counts"].reshape(-1, 3)[0])
                if self.sibling_sets is not None:
                    sibling_reports.append(self.sibling_sets.diagnostics())
                if self.sibling_event is not None:
                    sibling_reports.append(self.sibling_event.diagnostics())
                    sibling_event_reprs.append(
                        self.sibling_event.gate()
                        * global_outputs["sibling_event_repr"].reshape(-1)
                    )
                if "acm_channel_names" in global_outputs:
                    acm_gate_means.append(
                        {
                            name: global_outputs[f"acm_gate_{name}"]
                            for name in global_outputs["acm_channel_names"]
                        }
                    )
                    acm_zero_masses.append(global_outputs["acm_zero_channel_gate_mass"])
                if "time_gate_mean" in global_outputs:
                    time_gate_means.append(global_outputs["time_gate_mean"])
                    time_gate_heads.append(global_outputs["time_gate_per_head"])
                    time_bias_means.append(global_outputs["mean_abs_time_bias_semantic"])
                if "reconstruction_loss" in global_outputs:
                    global_semantic_losses.append(global_outputs["reconstruction_loss"])
                    global_semantic_masked_counts.append(global_outputs["masked_count"])
                    global_semantic_prediction_outputs.append(global_outputs["reconstruction"])
            elif self.structure_encoder_type == "set-transformer":
                graph_batch = torch.zeros(
                    structure_node_features.size(0),
                    dtype=torch.long,
                    device=device,
                )
                set_outputs = self.global_set_transformer(
                    node_features=structure_node_features,
                    root_indices=root_index.view(1),
                    graph_batch=graph_batch,
                    num_graphs=1,
                )
                graph_vectors.append(set_outputs["pooled"].squeeze(0))
            else:
                zero_graph = semantic_outputs["pooled"].new_zeros(self.graph_output_dim)
                graph_vectors.append(zero_graph)

        structure_only_repr = torch.stack(graph_vectors, dim=0)
        semantic_repr = torch.stack(semantic_vectors, dim=0)
        semantic_loss = torch.stack(semantic_losses).mean()
        masked_input_share = 0.0
        temporal_repr: Optional[torch.Tensor] = None
        temporal_outputs: Dict[str, torch.Tensor] = {}
        if self.temporal_branch is not None:
            temporal_outputs = self.temporal_branch(
                {"flat": torch.cat(temporal_feature_blocks, dim=0)}
            )
            temporal_repr = temporal_outputs["temporal_repr"]
        fusion_outputs = self.fusion(
            structure_only_repr, semantic_repr, temporal_repr,
            root_features=(
                torch.cat(root_feature_blocks, dim=0) if root_feature_blocks else None
            ),
        )
        temporal_report: Dict[str, torch.Tensor] = build_temporal_report(
            temporal_outputs, fusion_outputs
        )
        source_residual_outputs: Dict[str, torch.Tensor] = {}
        claim_outputs: Dict[str, torch.Tensor] = {}
        sibling_event_stack = (
            torch.stack(sibling_event_reprs, dim=0) if sibling_event_reprs else None
        )
        if self.claim_attention is not None and graph_node_states:
            flat_graph = torch.cat(graph_node_states, dim=0)
            flat_semantic = torch.cat(semantic_node_states, dim=0)
            flat_batch = torch.cat(node_owner, dim=0)
            evidence = self.claim_attention(
                flat_graph,
                flat_semantic,
                torch.arange(len(graph_node_states), dtype=torch.long, device=flat_graph.device),
                flat_batch,
                len(graph_node_states),
            )
            final_logits = self.evidence_classifier(
                torch.cat([fusion_outputs["fused_repr"], evidence], dim=-1)
            )
            claim_outputs = {
                "claim_evidence_repr": evidence,
                "claim_evidence_norm": evidence.norm(dim=-1).mean(),
                **self.claim_attention.diagnostics(),
            }
        elif self.sibling_classifier is not None and sibling_event_stack is not None:
            final_logits = self.sibling_classifier(
                torch.cat([fusion_outputs["fused_repr"], sibling_event_stack], dim=-1)
            )
            outputs["sibling_relative_norm"] = (
                sibling_event_stack.norm(dim=-1)
                / (fusion_outputs["fused_repr"].norm(dim=-1) + 1e-8)
            ).mean()
        elif self.source_residual_enabled and source_vectors:
            source_residual_outputs = self.source_residual_classifier(
                torch.stack(source_vectors, dim=0),
                fusion_outputs["fused_repr"],
            )
            final_logits = source_residual_outputs["logits"]
        else:
            final_logits = fusion_outputs["logits"]

        outputs = {
            "logits": final_logits,
            "fusion_logits": fusion_outputs["logits"],
            "fused_repr": fusion_outputs["fused_repr"],
            "semantic_repr": semantic_repr,
            **({"masked_input_share": masked_input_share} if masked_input_share else {}),
            **temporal_report,
            "structure_only_repr": structure_only_repr,
            "semantic_loss": semantic_loss,
            "semantic_prediction_outputs": semantic_prediction_outputs,
            "gat_proj": fusion_outputs["gat_proj"],
            "structure_proj": fusion_outputs["structure_proj"],
            "gagru_proj": fusion_outputs["gagru_proj"],
            "uniformity_loss": fusion_outputs["uniformity_loss"],
            "gat_disabled": not self.use_gat,
            "structure_disabled": not self.use_structure_encoder,
            "structure_encoder": self.structure_encoder_type,
            "gat_input_mode": self.gat_input_mode,
            "gat_readout": self.gat_readout,
            "gat_attention_type": self.gat_attention_type,
            "gat_residual_mode": self.gat_residual_mode,
            "gat_residual_alpha": self.gat_residual_alpha,
            "gat_neighborhood": self.gat_neighborhood,
            "gat_semantic_k": self.gat_semantic_k,
            "gat_topology": self.gat_topology,
            "gat_global_event_node": self.gat_global_event_node,
            "gat_source_evidence_pooling": self.gat_source_evidence_pooling,
            "gat_depth_encoding": self.gat_depth_encoding,
            "gat_relative_time_bias": self.gat_relative_time_bias,
            "joint_lora_enabled": self.use_joint_lora,
            "joint_lora_scope": self.joint_lora_scope,
            **claim_outputs,
        }
        if neighborhood_edge_counts:
            outputs["gat_neighborhood_edge_counts"] = torch.stack(neighborhood_edge_counts, dim=0)
        if sibling_event_reprs:
            outputs["sibling_event_repr"] = torch.stack(sibling_event_reprs, dim=0)
            outputs["sibling_event_norm"] = (
                torch.stack(sibling_event_reprs, dim=0).norm(dim=-1).mean()
            )
        if sibling_reports:
            for key in sibling_reports[0]:
                outputs[key] = sum(report[key] for report in sibling_reports) / len(
                    sibling_reports
                )
        if acm_zero_masses:
            outputs["acm_zero_channel_gate_mass"] = torch.stack(acm_zero_masses).mean()
        if acm_gate_means:
            outputs["acm_channel_names"] = list(acm_gate_means[0].keys())
            for name in outputs["acm_channel_names"]:
                outputs[f"acm_gate_{name}"] = torch.stack(
                    [entry[name] for entry in acm_gate_means]
                ).mean()
        if time_gate_means:
            outputs["time_gate_mean"] = torch.stack(time_gate_means).mean()
            outputs["time_gate_per_head"] = torch.stack(time_gate_heads).mean(dim=0)
            outputs["mean_abs_time_bias_semantic"] = torch.stack(time_bias_means).mean()
        if global_semantic_losses:
            outputs["global_semantic_loss"] = torch.stack(global_semantic_losses).mean()
            outputs["global_semantic_masked_count_mean"] = torch.stack(global_semantic_masked_counts).mean()
            outputs["global_semantic_prediction_outputs"] = global_semantic_prediction_outputs
        if "structure_logits" in fusion_outputs:
            outputs["structure_logits"] = fusion_outputs["structure_logits"]
        if "branch_abs_cosine" in fusion_outputs:
            outputs["branch_abs_cosine"] = fusion_outputs["branch_abs_cosine"]
        if self.use_gat and "gat_logits" in fusion_outputs:
            outputs["gat_logits"] = fusion_outputs["gat_logits"]
        if source_residual_outputs:
            outputs["source_logits"] = source_residual_outputs["source_logits"]
            outputs["delta_logits"] = source_residual_outputs["delta_logits"]
            outputs["source_residual_gamma"] = source_residual_outputs["gamma"]
        if self.contrastive_projection_dim > 0:
            outputs["contrastive_repr"] = F.normalize(
                self.contrastive_head(fusion_outputs["fused_repr"]), dim=-1
            )
        return outputs
