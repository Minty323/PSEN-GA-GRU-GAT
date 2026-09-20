from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader


class ModelEMA:
    """Exponential moving average of model weights.

    Keeps a shadow copy of the parameters that is updated after each optimizer
    step as ema = decay * ema + (1 - decay) * param. Evaluating / checkpointing
    with the averaged weights smooths out the late-training overfitting
    oscillations, narrowing the train/val and val/test gaps without changing
    the optimization itself. The model uses LayerNorm (no BatchNorm running
    buffers), so averaging the parameters is sufficient.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                self.shadow[name] = param.detach().clone()
            else:
                shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
if not omp_threads.isdigit() or int(omp_threads) <= 0:
    os.environ["OMP_NUM_THREADS"] = "4"

try:
    from .data import build_dataloaders, build_kfold_dataloaders, normalize_dataset_name
    from .model import DualBranchRumorDetector, disabled_edge_families_for
    from .utils import compute_binary_metrics, compute_classification_metrics, resolve_random_seed, save_json, set_seed
except ImportError:
    from data import build_dataloaders, build_kfold_dataloaders, normalize_dataset_name
    from model import DualBranchRumorDetector, disabled_edge_families_for
    from utils import compute_binary_metrics, compute_classification_metrics, resolve_random_seed, save_json, set_seed


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
# Model hyperparameters
# - max_seq_len: legacy root-token parsing length used by some dataset loaders
# - graph_hidden_dim: hidden size of one graph stream after GAT projection
# - fusion_dim: shared projection dimension used before concatenating the graph streams
# - graph_heads: number of attention heads in each GAT layer
# - semantic_hidden_dim: GA-GRU recurrent state size; 0 follows graph_hidden_dim
# - semantic_proj_dim: low-dimensional GA-GRU input bottleneck; 0 follows semantic_hidden_dim
# - semantic_loss_mse_weight: MSE contribution for online/fixed cosine + MSE targets;
#   raw targets always use pure MSE against the original node features
# - semantic_prediction_mode: predict absolute endpoints or parent-child residual changes
# - gagru_direction: use both recursive directions or retain only TD / BU with zero padding
# - gagru_aggregation: attention-weighted or uniformly weighted direct-neighbor context
# - gagru_attention_query: shared learned query or current-node-conditioned query
# - gagru_attention_score_mode: scaled dot-product or bounded additive attention
# - gagru_attention_residual_alpha: interpolation from mean (0) to attention (1)
# - gagru_attention_residual_mode: fixed interpolation or a node-adaptive mean anchor
# - gagru_attention_gate_init: initial adaptive attention contribution
# - gagru_semantic_direction_reduction: sum or average active directional prediction losses
# - gagru_direction_fusion: joint normalization or a TD-preserving BU residual
# - gagru_bu_residual_init: initial BU contribution for TD-residual direction fusion
# - gagru_bu_shared_gradient_scale: BU gradient contribution to the shared TD input projection
# - gagru_readout: all-node mean or TD mean concatenated with the recursive BU root
# - gat_input_mode: encode raw node semantics or detached GA-GRU prediction residuals
# - gat_readout: pool final GAT states or propagation-induced node-state changes
# - gat_attention_type: classic static GAT or destination-conditioned GATv2
# - gat_residual_mode / alpha: fixed residual update or learnable layer-wise interpolation
# - gat_neighborhood / semantic_k: local GAT or MS-GAT v1 multi-scale event-local topology
# - gat_global_event_node: source-conditioned one-way event collector for MS-GAT v1g
# - gat_depth_encoding: five-bucket source-distance embedding for MS-GAT v1d/v1gd
# - gat_relative_time_bias: signed relative-time logit bias on semantic MS-GAT edges only
# - source_residual: V1G-R1 source-anchored residual prediction (final logits are
#   source-only logits plus a gated correction from the fused propagation representation)
# - source_residual_gate_init: initial sigmoid correction gate gamma in (0, 1)
# - global_semantic_mask_ratio: target-eligible node ratio masked for GAT semantic prediction
# - global_semantic_target: latent GAT state or raw node feature
# - global_semantic_feature_std_floor: lower bound used by variance-normalized raw-feature MSE
# - uniformity_temperature: Gaussian potential temperature for event representation uniformity
# - dropout: dropout ratio used in graph / fusion modules
# - input_bottleneck_dim: shared trainable node-feature dimension before both branches; 0 disables it
MODEL_HYPERPARAMETERS = {
    "max_seq_len": 64,
    "graph_hidden_dim": 64,
    "fusion_dim": 128,
    "graph_heads": 4,
    "semantic_hidden_dim": 0,
    "semantic_proj_dim": 0,
    "semantic_loss_mse_weight": 0.1,
    "semantic_prediction_mode": "absolute",
    "semantic_loss_criterion": "auto",
    "root_feature_dim": 0,
    "root_projection_dim": 0,
    "semantic_sce_gamma": 1.0,
    "gagru_direction": "bidirectional",
    "gagru_aggregation": "attention",
    "gagru_attention_query": "global",
    "gagru_attention_score_mode": "dot",
    "gagru_attention_residual_alpha": 1.0,
    "gagru_attention_residual_mode": "fixed",
    "gagru_attention_gate_init": 0.1,
    "gagru_semantic_direction_reduction": "sum",
    "gagru_direction_fusion": "joint",
    "gagru_bu_residual_init": 0.1,
    "gagru_bu_shared_gradient_scale": 1.0,
    "gagru_readout": "mean",
    "gat_input_mode": "raw",
    "gat_readout": "mean",
    "gat_attention_type": "gat",
    "gat_residual_mode": "fixed",
    "gat_residual_alpha": 1.0,
    "gat_neighborhood": "local",
    "gat_semantic_k": 3,
    "gat_topology": "tree",
    "gat_global_event_node": False,
    "gat_source_evidence_pooling": False,
    "gat_depth_encoding": False,
    "gat_relative_time_bias": False,
    "acm_channels": "identity,low,high,2hop",
    "claim_guided_attention": False,
    "propagation_adapter": False,
    "propagation_adapter_bottleneck": 192,
    "contrastive_views": False,
    "contrastive_temperature": 0.2,
    "contrastive_node_drop_probability": 0.1,
    "contrastive_attribute_mask_probability": 0.1,
    "claim_attention_dim": 128,
    "masked_input_ratio": 0.0,
    "gat_edge_families": "real,2hop,semantic",
    "temporal_dynamics": "off",
    "temporal_evolution": False,
    "temporal_evolution_stages": 4,
    "temporal_evolution_hidden": 128,
    "temporal_evolution_dropout": 0.0,
    "temporal_evolution_decay_init": -2.3,
    "temporal_evolution_residual_zero_init": True,
    "temporal_evolution_final_stage_aux_only": True,
    "temporal_dim": 64,
    "temporal_hidden": 64,
    "temporal_channels": 32,
    "temporal_kernel_size": 3,
    "temporal_dropout": 0.0,
    "temporal_normalize": True,
    "temporal_identity_init": True,
    "sibling_interaction": False,
    "sibling_mode": "true",
    "sibling_heads": 4,
    "sibling_gate_init": 0.1,
    "sibling_gate_mode": "learned",
    "sibling_fusion": "node-residual",
    "sibling_dim": 128,
    "source_residual": False,
    "source_residual_gate_init": 0.2,
    "supcon_projection_dim": 0,
    "node_representation": "independent",
    "temporal_consistency": False,
    "temporal_snapshot_mode": "ratio",
    "temporal_snapshot_ratios": (0.2, 0.4, 0.6, 0.8),
    "temporal_cutoffs_minutes": (20.0, 40.0, 60.0, 80.0),
    "hierarchical_evidence": False,
    "hierarchical_slots": 4,
    "claim_response_relation": False,
    "relation_gate_bias": 3.0,
    "global_semantic_mask_ratio": 0.25,
    "global_semantic_target": "latent",
    "global_semantic_sce_gamma": 2.0,
    "global_semantic_feature_std_floor": 0.05,
    "uniformity_temperature": 2.0,
    "dropout": 0.4,
    "input_bottleneck_dim": 0,
    "edge_dropout": 0.0,
}

# Training hyperparameters
# - batch_size: number of samples in one batch
# - epochs: maximum number of training epochs
# - lr: AdamW learning rate
# - weight_decay: L2 regularization strength
# - train_ratio / val_ratio: dataset split ratios; the rest is test ratio
# - num_workers: dataloader worker count
# - patience: early stopping patience on validation F1
# - grad_clip: gradient clipping threshold
# - label_smoothing: cross-entropy label smoothing ratio
# - semantic_loss_weight: GA-GRU parent->child and child->parent semantic prediction loss weight
# - semantic_loss_decay_start_epoch / end_epoch: optional linear decay window for that weight
# - global_semantic_loss_weight: GAT masked global semantic forecasting loss weight
# - gat_aux_loss_weight: supervised GAT-only classification loss used to prevent branch collapse
# - uniformity_loss_weight: event representation uniformity regularizer weight
# - partial_graph_train_prob: fraction of training events rendered as partial subtrees
# - partial_graph_keep_ratios: retained non-root node ratios for partial event views
# - seed: training random seed; -1 means generate a new seed on each run
# - split_seed: dataset split seed
TRAINING_HYPERPARAMETERS = {
    # Clean/mask consistency: lambda_cons, then the mask ratio of the second view. Both are
    # optimiser-side quantities, not model constructor arguments.
    "mask_consistency_loss_weight": 0.0,
    "mask_consistency_ratio": 0.169,
    "batch_size": 4,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "bigcn_inner_val_ratio": 0.0,
    "num_workers": 0,
    "patience": 3,
    "grad_clip": 5.0,
    "label_smoothing": 0.0,
    "semantic_loss_weight": 0.005,
    "semantic_loss_decay_start_epoch": 0,
    "semantic_loss_decay_end_epoch": 0,
    "global_semantic_loss_weight": 0.02,
    "gat_pretrain_epochs": 0,
    "gat_pretrain_lr": 0.0,
    "global_loss_warmup_epochs": 0,
    "gat_aux_loss_weight": 0.0,
    "uniformity_loss_weight": 0.0,
    "supcon_loss_weight": 0.0,
    "supcon_temperature": 0.1,
    "contrastive_loss_weight": 0.05,
    "future_loss_weight": 0.05,
    "early_ce_weight": 0.0,
    "partial_graph_train_prob": 0.0,
    "partial_graph_keep_ratios": (0.7, 0.8, 0.9),
    "disable_class_weights": False,
    "loss_type": "ce",
    "focal_gamma": 2.0,
    "focal_alpha": -1.0,
    "checkpoint_metric": "macro_f1",
    "threshold_selection_metric": "macro_f1",
    "threshold_search_start": 0.05,
    "threshold_search_end": 0.95,
    "threshold_search_step": 0.01,
    "seed": -1,
    "split_seed": 42,
}

TWITTER16_MODEL_OVERRIDES = {
    "graph_hidden_dim": 128,
    "fusion_dim": 128,
    "dropout": 0.5,
}

TWITTER16_TRAINING_OVERRIDES = {
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-3,
    "patience": 4,
}

WEIBO_MODEL_OVERRIDES = {
    "max_seq_len": 128,
    "graph_heads": 2,
    "dropout": 0.35,
}

WEIBO_TRAINING_OVERRIDES = {
    "batch_size": 1,
    "epochs": 20,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "patience": 5,
    "grad_clip": 2.0,
    "label_smoothing": 0.05,
}

WEIBO_RUNTIME_OVERRIDES = {
    "max_graph_nodes": 512,
}

PHEMERAW_MODEL_OVERRIDES = {
    "max_seq_len": 96,
    "graph_hidden_dim": 64,
    "fusion_dim": 128,
    "graph_heads": 4,
    "dropout": 0.4,
}

PHEMERAW_TRAINING_OVERRIDES = {
    "batch_size": 4,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "patience": 4,
}

WEIBO21_MODEL_OVERRIDES = {
    "max_seq_len": 170,
    "graph_hidden_dim": 64,
    "fusion_dim": 128,
    "graph_heads": 2,
    "dropout": 0.35,
}

WEIBO21_TRAINING_OVERRIDES = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 5e-5,
    "patience": 4,
    "grad_clip": 2.0,
    "label_smoothing": 0.05,
}

# Runtime / path hyperparameters
RUNTIME_HYPERPARAMETERS = {
    "dataset_root": "dataset",
    "dataset_name": "Weibo",
    "save_dir": "artifacts/dual_branch_rumor",
    "max_graph_nodes": 0,
}

THRESHOLD_SEARCH_SPACE = [round(i / 100, 2) for i in range(5, 96)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GA-GRU dual-branch rumor detector.")
    parser.add_argument("--dataset-root", type=str, default=RUNTIME_HYPERPARAMETERS["dataset_root"])
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=RUNTIME_HYPERPARAMETERS["dataset_name"],
        help=(
            "Dataset name (case-insensitive): Pheme, PhemeRaw, Twitter15, Twitter15TD, Twitter15Raw, "
            "Twitter15ICDM, Twitter16, Twitter16TD, Twitter16ICDM, Twitter16-tfidf, Weibo, Weibo21"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=TRAINING_HYPERPARAMETERS["batch_size"])
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Accumulate this many physical batches before one optimizer/EMA update. "
            "For joint LoRA, batch-size 4 with 8 steps gives an effective graph batch of 32."
        ),
    )
    parser.add_argument("--epochs", type=int, default=TRAINING_HYPERPARAMETERS["epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING_HYPERPARAMETERS["lr"])
    parser.add_argument(
        "--gat-lr", type=float, default=0.0,
        help="GAT/Global Event Node learning rate; 0 reuses --lr.",
    )
    parser.add_argument(
        "--gagru-lr", type=float, default=0.0,
        help="GA-GRU learning rate; 0 reuses --lr.",
    )
    parser.add_argument(
        "--fusion-lr", type=float, default=0.0,
        help="Fusion classifier learning rate; 0 reuses --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=TRAINING_HYPERPARAMETERS["weight_decay"])
    parser.add_argument(
        "--init-checkpoint", type=str, default=None,
        help="Initialize all model weights from a strict state-dict checkpoint before training.",
    )
    parser.add_argument(
        "--allow-initial-selection",
        action="store_true",
        help=(
            "Evaluate an --init-checkpoint as epoch 0 on the validation split and retain it "
            "when later epochs do not improve the requested checkpoint metric. The option "
            "is rejected for test-set checkpoint selection."
        ),
    )
    parser.add_argument(
        "--freeze-gagru", action="store_true",
        help="Freeze the GA-GRU recursive branch while retaining its representations in fusion.",
    )
    parser.add_argument("--dropout", type=float, default=MODEL_HYPERPARAMETERS["dropout"])
    parser.add_argument(
        "--input-bottleneck-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["input_bottleneck_dim"],
        help=(
            "Shared trainable node-feature bottleneck before GAT and GA-GRU; "
            "zero preserves the original input dimension."
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=MODEL_HYPERPARAMETERS["max_seq_len"])
    parser.add_argument("--graph-hidden-dim", type=int, default=MODEL_HYPERPARAMETERS["graph_hidden_dim"])
    parser.add_argument("--fusion-dim", type=int, default=MODEL_HYPERPARAMETERS["fusion_dim"])
    parser.add_argument("--graph-heads", type=int, default=MODEL_HYPERPARAMETERS["graph_heads"])
    parser.add_argument(
        "--structure-encoder",
        choices=("gat", "acm-gcn", "set-transformer", "none"),
        default="gat",
        help=(
            "Global structure branch paired with GA-GRU. 'gat' is the original MS-GAT "
            "attention stack; 'acm-gcn' replaces it with ACM adaptive channel mixing "
            "(identity / low-pass / high-pass / exact two-hop, per-node softmax gate); "
            "'set-transformer' models all event nodes without edges, order, or "
            "timestamps; 'none' is the GA-GRU-only ablation."
        ),
    )
    parser.add_argument(
        "--parent-shuffle",
        action="store_true",
        help=(
            "Causal structural control: rewire which reply hangs under which parent for "
            "depths >= 2, preserving the node set, per-node depth, per-level counts and "
            "every parent's out-degree. Applied to train, validation AND test, so a "
            "shuffled model is matched to the real-tree model in both conditions; "
            "shuffling only the test set would measure an OOD perturbation instead."
        ),
    )
    parser.add_argument(
        "--parent-shuffle-seed", type=int, default=0,
        help="Seed offset for the parent shuffle (per-event seed = this + dataset index).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force deterministic CUDA kernels. Required for any comparison at the "
            "~1pp level: without it two runs of the same config diverge from about "
            "epoch 8 and land on different checkpoints."
        ),
    )
    parser.add_argument(
        "--save-init-checkpoint",
        type=str,
        default=None,
        help=(
            "Write the freshly initialised state_dict here before training. Use the "
            "baseline arm to freeze one backbone initialisation for every arm."
        ),
    )
    parser.add_argument(
        "--backbone-init-checkpoint",
        type=str,
        default=None,
        help=(
            "Paired initialisation: copy every matching parameter from this checkpoint "
            "after construction, so all arms share an identical backbone and only the "
            "new module is randomly initialised."
        ),
    )
    parser.add_argument(
        "--paired-init-allow-shape-mismatch",
        action="store_true",
        default=False,
        help=(
            "Permit paired init to leave resize-affected layers at their own initialisation "
            "instead of refusing. Required only when the change necessarily resizes existing "
            "layers, e.g. widening the classifier input for root feature enhancement; the "
            "unchanged backbone is still copied, and every skipped layer is printed."
        ),
    )
    parser.add_argument(
        "--sibling-interaction",
        action="store_true",
        help=(
            "Residual sibling-set interaction: replies sharing one parent attend to "
            "each other in a masked self-attention, and the result is added to the "
            "GRAPH branch as H + gamma_sib * delta. GA-GRU is untouched and "
            "gamma_sib = 0 reproduces V1G exactly. Single CE loss, no new edge family."
        ),
    )
    parser.add_argument(
        "--sibling-mode",
        choices=("true", "random-same-depth"),
        default=MODEL_HYPERPARAMETERS["sibling_mode"],
        help=(
            "Grouping for the sibling interaction. 'true' = children of the same "
            "parent. 'random-same-depth' = the control with matched group sizes from "
            "the same event and depth but no shared parent."
        ),
    )
    parser.add_argument("--sibling-heads", type=int,
                        default=MODEL_HYPERPARAMETERS["sibling_heads"])
    parser.add_argument("--sibling-gate-init", type=float,
                        default=MODEL_HYPERPARAMETERS["sibling_gate_init"],
                        help=(
                            "Initial gamma_sib. 0.1 is right when the module is known to "
                            "help; for a diagnostics run that asks whether it helps at "
                            "all, use 1.0 together with --sibling-gate-mode fixed."
                        ))
    parser.add_argument(
        "--sibling-gate-mode",
        choices=("learned", "fixed"),
        default=MODEL_HYPERPARAMETERS["sibling_gate_mode"],
        help=(
            "'learned' trains gamma_sib. 'fixed' freezes it at --sibling-gate-init so "
            "the branch cannot be ignored by the optimiser."
        ),
    )
    parser.add_argument(
        "--sibling-fusion",
        choices=("node-residual", "event-concat"),
        default=MODEL_HYPERPARAMETERS["sibling_fusion"],
        help=(
            "'node-residual' adds gamma*delta to the node features of the graph branch "
            "(bypassable). 'event-concat' builds an independent event-level sibling "
            "vector and concatenates it with the V1G representation before the "
            "classifier, so it cannot be bypassed."
        ),
    )
    parser.add_argument("--sibling-dim", type=int,
                        default=MODEL_HYPERPARAMETERS["sibling_dim"],
                        help="Width of the event-level sibling vector (event-concat).")
    parser.add_argument(
        "--propagation-adapter-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a stage-1 adapter checkpoint (scripts/pretrain_propagation_"
            "adapter.py). Enables the M6 propagation-infused semantic adapter: "
            "x -> x' = x + W2 GELU(W1 LN(x)) is applied before both branches, so the "
            "graph model receives propagation-aware semantics. V1G, GA-GRU, fusion "
            "and the classifier are untouched."
        ),
    )
    parser.add_argument("--propagation-adapter-bottleneck", type=int,
                        default=MODEL_HYPERPARAMETERS["propagation_adapter_bottleneck"])
    parser.add_argument(
        "--propagation-adapter-finetune",
        action="store_true",
        help=(
            "Also keep training the adapter during stage 2. Off by default so a change "
            "in accuracy is attributable to the representation rather than to extra "
            "capacity inside the graph model."
        ),
    )
    parser.add_argument(
        "--contrastive-views",
        action="store_true",
        help=(
            "RAGCL-style adaptive same-event graph contrastive regularisation: two "
            "augmented views (adaptive node drop with ancestor reconnection, adaptive "
            "attribute masking) of the SAME event are pulled together by symmetric "
            "InfoNCE, with every other event as a negative. Labels never define "
            "positives. Training only; inference is the untouched baseline."
        ),
    )
    parser.add_argument("--contrastive-temperature", type=float,
                        default=MODEL_HYPERPARAMETERS["contrastive_temperature"])
    parser.add_argument("--contrastive-loss-weight", type=float,
                        default=TRAINING_HYPERPARAMETERS["contrastive_loss_weight"],
                        help="lambda_c. Fixed diagnostic starting value: 0.05.")
    parser.add_argument("--contrastive-node-drop-probability", type=float,
                        default=MODEL_HYPERPARAMETERS["contrastive_node_drop_probability"],
                        help="Base adaptive node-drop rate p_n (0.1 fixed for the first round).")
    parser.add_argument("--contrastive-attribute-mask-probability", type=float,
                        default=MODEL_HYPERPARAMETERS["contrastive_attribute_mask_probability"],
                        help="Base adaptive attribute-mask rate p_m (0.1 fixed for the first round).")
    parser.add_argument(
        "--claim-guided-attention",
        action="store_true",
        help=(
            "ClaHi-style claim-guided event attention: replies are scored by an "
            "inference relation against the source (u_c, u_i, u_c*u_i, |u_c-u_i|), "
            "pooled into a claim-conditioned evidence vector, and CONCATENATED with "
            "the existing [z_G ; z_R] fusion before the classifier (never folded into "
            "the pooled state). CE only."
        ),
    )
    parser.add_argument(
        "--claim-attention-dim", type=int,
        default=MODEL_HYPERPARAMETERS["claim_attention_dim"],
        help="Width of the claim-conditioned evidence vector.",
    )
    parser.add_argument(
        "--temporal-dynamics",
        choices=("off", "static", "sequence"),
        default=MODEL_HYPERPARAMETERS["temporal_dynamics"],
        help=(
            "M7 event-level temporal dynamics branch. 'static' (T1) adds only the ~13 "
            "event-level time scalars (duration, edge-delay quantiles, peak rate, "
            "time-to-50/80%%), so a gain over M0 proves time information itself is "
            "discriminative. 'sequence' (T2) adds only the 10x10 window growth sequence, "
            "so a further gain proves the SHAPE of the growth adds something. Both are "
            "appended to [z_G ; z_R] before the classifier, CE only."
        ),
    )
    parser.add_argument(
        "--gat-edge-families",
        type=str,
        default=MODEL_HYPERPARAMETERS["gat_edge_families"],
        help=(
            "Which MS-GAT edge families to keep: any comma-separated subset of "
            "real,2hop,semantic ('all' = the current V1G). This is a pure removal arm: "
            "the real reply tree and the global event node always stay, GA-GRU, fusion "
            "and the loss are untouched. 'real,2hop' = V1 (drop semantic-KNN), "
            "'real' = V2 (drop semantic-KNN and two-hop)."
        ),
    )
    parser.add_argument(
        "--mask-consistency-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["mask_consistency_loss_weight"],
        help=(
            "lambda_cons for clean/mask consistency: R-Drop applied to the semantic masking "
            "this model already uses. Each event is classified twice -- clean and with a "
            "16.9%% mask -- the two CE terms are averaged and a symmetric KL pulls the two "
            "class distributions together. Chosen by the stability diagnostic, which found "
            "errors 2.1x less stable under masking than correct events. 0 disables it, and "
            "that is the default so no existing recipe changes."
        ),
    )
    parser.add_argument(
        "--mask-consistency-ratio", type=float,
        default=TRAINING_HYPERPARAMETERS["mask_consistency_ratio"],
        help="Mask ratio of the second view (0.169 = the share M2a masking realises).",
    )
    parser.add_argument(
        "--masked-input-ratio",
        type=float,
        default=MODEL_HYPERPARAMETERS["masked_input_ratio"],
        help=(
            "M2a ablation: during training, zero out ceil(ratio * replies) non-source "
            "nodes per event in the tensor the encoders consume, and add NO reconstruction "
            "objective. Combine with --global-semantic-mask-ratio 0 --global-semantic-loss-weight 0 "
            "so the only difference from M2' is whether the masked view is reconstructed. "
            "The rule matches M2' exactly (source post never masked, per-event rounding)."
        ),
    )
    parser.add_argument(
        "--temporal-evolution",
        action="store_true",
        default=MODEL_HYPERPARAMETERS["temporal_evolution"],
        help=(
            "TSE-V1G. The event becomes a sequence of cumulative propagation states "
            "G^(1) subset ... subset G^(K) = G, every stage is encoded by the SAME "
            "V1G + GA-GRU, consecutive states are combined with a drift term "
            "(s_k - s_{k-1}), a learned decay rho_k = exp(-softplus(gamma) * log1p(dT_k)) "
            "gates the previous state inside a GRU, and only a zero-initialised "
            "residual may change the full-stage prediction. Each snapshot rebuilds its "
            "own semantic-KNN / two-hop / global-node edges from the nodes visible at "
            "that stage, so no future node can leak backwards. CE only by default."
        ),
    )
    parser.add_argument(
        "--temporal-evolution-stages", type=int,
        default=MODEL_HYPERPARAMETERS["temporal_evolution_stages"],
        help="K cumulative snapshots at ceil(k*N/K) replies (25/50/75/100%% for K=4).",
    )
    parser.add_argument(
        "--temporal-evolution-hidden", type=int,
        default=MODEL_HYPERPARAMETERS["temporal_evolution_hidden"],
        help="Hidden width of the state-evolution GRU.",
    )
    parser.add_argument(
        "--temporal-evolution-dropout", type=float,
        default=MODEL_HYPERPARAMETERS["temporal_evolution_dropout"],
        help="Dropout on the state input of the evolution GRU.",
    )
    parser.add_argument(
        "--temporal-evolution-decay-init", type=float,
        default=MODEL_HYPERPARAMETERS["temporal_evolution_decay_init"],
        help=(
            "Initial gamma. softplus(-2.3) ~= 0.093, i.e. rho ~= 0.94 one minute apart "
            "and ~= 0.80 ten minutes apart: mild forgetting that training can sharpen."
        ),
    )
    parser.add_argument(
        "--no-temporal-evolution-zero-init", dest="temporal_evolution_residual_zero_init",
        action="store_false",
        default=MODEL_HYPERPARAMETERS["temporal_evolution_residual_zero_init"],
        help=(
            "Let the dynamic residual perturb the initial logits. By default the "
            "residual head is zero-initialised, so step 0 is exactly M0."
        ),
    )
    parser.add_argument(
        "--no-temporal-evolution-final-stage-aux-only",
        dest="temporal_evolution_final_stage_aux_only", action="store_false",
        default=MODEL_HYPERPARAMETERS["temporal_evolution_final_stage_aux_only"],
        help=(
            "Also compute the M2 masked reconstruction on the intermediate snapshots "
            "(default: only on the complete event, exactly as in M2)."
        ),
    )
    parser.add_argument(
        "--temporal-dim", type=int, default=MODEL_HYPERPARAMETERS["temporal_dim"],
        help="Width of the temporal representation appended to the fusion vector.",
    )
    parser.add_argument(
        "--temporal-hidden", type=int, default=MODEL_HYPERPARAMETERS["temporal_hidden"],
        help="Hidden width of the T1 scalar encoder.",
    )
    parser.add_argument(
        "--temporal-channels", type=int, default=MODEL_HYPERPARAMETERS["temporal_channels"],
        help="Conv1D channel count of the T2 window encoder.",
    )
    parser.add_argument(
        "--temporal-kernel-size", type=int, default=MODEL_HYPERPARAMETERS["temporal_kernel_size"],
        help="Conv1D kernel width along the time axis (windows) of the T2 encoder.",
    )
    parser.add_argument(
        "--temporal-dropout", type=float, default=MODEL_HYPERPARAMETERS["temporal_dropout"],
        help="Dropout inside the temporal encoder.",
    )
    parser.add_argument(
        "--no-temporal-normalize", dest="temporal_normalize", action="store_false",
        default=MODEL_HYPERPARAMETERS["temporal_normalize"],
        help=(
            "Disable z-scoring of the temporal features. The statistics are always "
            "fitted on the training split only, never on val/test."
        ),
    )
    parser.add_argument(
        "--no-temporal-identity-init", dest="temporal_identity_init", action="store_false",
        default=MODEL_HYPERPARAMETERS["temporal_identity_init"],
        help=(
            "Let the temporal branch change the initial logits. By default the widened "
            "classifier columns start at zero, making T1/T2 exact clones of M0 at "
            "initialisation so any difference is attributable to the branch."
        ),
    )
    parser.add_argument(
        "--acm-channels",
        type=str,
        default=MODEL_HYPERPARAMETERS["acm_channels"],
        help=(
            "Comma-separated channels for --structure-encoder acm-gcn, chosen from "
            "identity, low, high, 2hop, semantic. The default 'identity,low,high,2hop' "
            "follows the ACM formulation plus an exact two-hop channel; "
            "'identity,low,2hop' is the H2GCN-style no-high-pass variant."
        ),
    )
    parser.add_argument(
        "--acm-gate-mode",
        choices=("learned", "uniform", "identity-only", "low-only", "high-only"),
        default="learned",
        help=(
            "ACM channel mixing. 'learned' is the model under test (per-node softmax "
            "gate); the others force a fixed channel for the pre-registered "
            "falsification ablations (e.g. 'low-only' = plain neighbourhood aggregation)."
        ),
    )
    parser.add_argument(
        "--gat-input-mode",
        choices=("raw", "semantic-residual", "neighbor-residual"),
        default=MODEL_HYPERPARAMETERS["gat_input_mode"],
        help=(
            "Node view encoded by GAT. 'raw' preserves the original parallel branches; "
            "'neighbor-residual' independently encodes each node's deviation from its "
            "undirected one-hop neighborhood; 'semantic-residual' is the GA-GRU prediction-"
            "error correction ablation."
        ),
    )
    parser.add_argument(
        "--gat-readout",
        choices=("mean", "propagation-change", "evidence"),
        default=MODEL_HYPERPARAMETERS["gat_readout"],
        help=(
            "Graph-level GAT representation. 'evidence' keeps root and attends comments; 'mean' pools final node states; "
            "'propagation-change' pools the mean, standard deviation, and maximum "
            "absolute change between projected inputs and final GAT states."
        ),
    )
    parser.add_argument(
        "--gat-attention-type",
        choices=("gat", "gatv2"),
        default=MODEL_HYPERPARAMETERS["gat_attention_type"],
        help=(
            "Neighbor-attention formulation. 'gat' preserves the classic static score; "
            "'gatv2' uses destination-conditioned dynamic attention."
        ),
    )
    parser.add_argument(
        "--gat-residual-mode",
        choices=("fixed", "learnable"),
        default=MODEL_HYPERPARAMETERS["gat_residual_mode"],
        help=(
            "Residual update inside each GAT layer. 'fixed' preserves h + alpha*m; "
            "'learnable' learns a layer-wise interpolation initialized by alpha."
        ),
    )
    parser.add_argument(
        "--gat-residual-alpha",
        type=float,
        default=MODEL_HYPERPARAMETERS["gat_residual_alpha"],
        help=(
            "Fixed message scale, or initial message interpolation when "
            "--gat-residual-mode learnable is selected."
        ),
    )
    parser.add_argument(
        "--gat-neighborhood",
        choices=("local", "ms-gat-v1"),
        default=MODEL_HYPERPARAMETERS["gat_neighborhood"],
        help=(
            "GAT visible neighborhood. 'local' uses the original one-hop layer; "
            "'ms-gat-v1' adds exact two-hop virtual edges and event-local semantic edges."
        ),
    )
    parser.add_argument(
        "--gat-semantic-k",
        type=int,
        default=MODEL_HYPERPARAMETERS["gat_semantic_k"],
        help="Maximum number of event-local semantic edges per destination in MS-GAT v1.",
    )
    parser.add_argument(
        "--gat-topology",
        choices=("tree", "star", "set"),
        default=MODEL_HYPERPARAMETERS["gat_topology"],
        help=(
            "Propagation topology used by both branches. 'star' connects every reply "
            "directly to its source; 'set' removes all propagation edges and requires "
            "the Set Transformer structure encoder."
        ),
    )
    parser.add_argument(
        "--gat-global-event-node",
        action="store_true",
        help="Add a source-conditioned one-way global event collector to MS-GAT v1.",
    )
    parser.add_argument(
        "--gat-source-evidence-pooling",
        action="store_true",
        help=(
            "Extend the global event node with source-conditioned evidence pooling: "
            "the source queries replies only (source is never a value), and the "
            "evidence view is fused into the event state by a zero-initialized "
            "residual MLP. Requires --gat-global-event-node."
        ),
    )
    parser.add_argument(
        "--gat-depth-encoding",
        action="store_true",
        help="Add propagation-tree depth buckets 0/1/2/3/>=4 to MS-GAT node inputs.",
    )
    parser.add_argument(
        "--gat-relative-time-bias",
        action="store_true",
        help=(
            "Add signed relative-time attention bias to MS-GAT semantic edges only. "
            "Propagation, two-hop, value, node-feature, and global-event paths stay unchanged."
        ),
    )
    parser.add_argument(
        "--source-residual",
        action="store_true",
        help=(
            "V1G-R1: final logits become source-only logits plus a sigmoid-gated "
            "correction from the fused propagation representation. Branch encoders, "
            "fusion, and the loss are unchanged."
        ),
    )
    parser.add_argument(
        "--source-residual-gate-init",
        type=float,
        default=MODEL_HYPERPARAMETERS["source_residual_gate_init"],
        help="Initial value of the learnable correction gate gamma = sigmoid(a) in (0, 1).",
    )
    parser.add_argument(
        "--set-transformer-latents",
        type=int,
        default=8,
        help="Number of learned latent tokens in the global Set Transformer branch.",
    )
    parser.add_argument(
        "--set-transformer-layers",
        type=int,
        default=2,
        help="Number of latent cross/self-attention blocks in the Set Transformer branch.",
    )
    parser.add_argument(
        "--joint-lora",
        action="store_true",
        help=(
            "Encode every PHEME graph node online with a frozen Transformer plus trainable "
            "native LoRA adapters. Requires --node-token-root."
        ),
    )
    parser.add_argument(
        "--joint-lora-scope",
        choices=("all", "gat"),
        default="all",
        help=(
            "Branches receiving online LoRA node features. 'all' is the previous behavior; "
            "'gat' keeps GA-GRU on stored frozen BERT features."
        ),
    )
    parser.add_argument(
        "--node-token-root",
        type=str,
        default=None,
        help="Root containing Phemetokens/*.npz sidecars for online LoRA encoding.",
    )
    parser.add_argument(
        "--joint-lora-model",
        type=str,
        default="bert-base-uncased",
        help="Local or Hugging Face Transformer used by the joint LoRA node encoder.",
    )
    parser.add_argument(
        "--joint-lora-init",
        type=str,
        default="auto",
        help=(
            "Adapter initialization checkpoint. 'auto' loads text_encoder_lora.pt from each "
            "fold feature root; 'none' starts with newly initialized LoRA weights."
        ),
    )
    parser.add_argument("--joint-lora-rank", type=int, default=8)
    parser.add_argument("--joint-lora-alpha", type=float, default=16.0)
    parser.add_argument("--joint-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--joint-lora-last-layers",
        type=int,
        default=4,
        help="Number of final Transformer layers receiving LoRA; -1 selects every layer.",
    )
    parser.add_argument(
        "--joint-lora-pooling",
        choices=("mean", "cls"),
        default="mean",
    )
    parser.add_argument(
        "--joint-lora-text-batch-size",
        type=int,
        default=16,
        help="Maximum number of graph nodes encoded by BERT in one internal microbatch.",
    )
    parser.add_argument(
        "--joint-lora-lr",
        type=float,
        default=2e-5,
        help="Learning rate used only for LoRA parameters.",
    )
    parser.add_argument(
        "--joint-lora-freeze-epochs",
        type=int,
        default=0,
        help=(
            "Keep the initialized LoRA text encoder frozen and in eval mode for this many "
            "epochs while the randomly initialized graph classifier adapts."
        ),
    )
    parser.add_argument(
        "--joint-lora-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay used only for LoRA parameters.",
    )
    parser.add_argument(
        "--joint-lora-gradient-checkpointing",
        action="store_true",
        help="Trade Transformer compute for lower joint-training activation memory.",
    )
    parser.add_argument(
        "--allow-joint-lora-download",
        action="store_true",
        help="Allow Transformers to download --joint-lora-model instead of requiring its local cache.",
    )
    parser.add_argument(
        "--disable-joint-lora-amp",
        action="store_true",
        help="Disable float16 autocast inside the online text encoder.",
    )
    parser.add_argument(
        "--semantic-hidden-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["semantic_hidden_dim"],
        help="GA-GRU recurrent hidden size. Use 0 to follow --graph-hidden-dim.",
    )
    parser.add_argument(
        "--semantic-proj-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["semantic_proj_dim"],
        help=(
            "Low-dimensional input bottleneck before GA-GRU. Use 0 to follow --semantic-hidden-dim. "
            "With --semantic-target raw, prediction still decodes to the original node-feature dimension."
        ),
    )
    parser.add_argument(
        "--semantic-loss-mse-weight",
        type=float,
        default=MODEL_HYPERPARAMETERS["semantic_loss_mse_weight"],
        help=(
            "MSE weight added to cosine loss for online/fixed semantic targets. "
            "Raw targets always use pure MSE, so this option has no effect in raw mode."
        ),
    )
    parser.add_argument(
        "--semantic-loss-criterion",
        type=str,
        choices=("auto", "mse", "cosine", "sce"),
        default=MODEL_HYPERPARAMETERS["semantic_loss_criterion"],
        help=(
            "Criterion for the GA-GRU parent->child and child->parent semantic prediction. "
            "'auto' keeps the historical behaviour (pure MSE for raw targets, otherwise cosine "
            "plus --semantic-loss-mse-weight * MSE). 'mse' forces pure MSE, 'cosine' forces "
            "mean(1 - cos) and 'sce' forces GraphMAE's scaled cosine error "
            "mean((1 - cos)^gamma), which drops the vector-norm/scale sensitivity of MSE and "
            "down-weights edges that are already well reconstructed."
        ),
    )
    parser.add_argument(
        "--semantic-sce-gamma",
        type=float,
        default=MODEL_HYPERPARAMETERS["semantic_sce_gamma"],
        help="Exponent gamma >= 1 for --semantic-loss-criterion sce (ignored otherwise).",
    )
    parser.add_argument(
        "--root-feature-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["root_feature_dim"],
        help=(
            "Root Feature Enhancement (Bi-GCN): width of the source post's own input feature "
            "that is projected and concatenated into the classifier input, e.g. 768 for frozen "
            "BERT. 0 disables it and leaves the classifier input unchanged."
        ),
    )
    parser.add_argument(
        "--root-projection-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["root_projection_dim"],
        help="Output width of the root projection W_r; required when --root-feature-dim > 0.",
    )
    parser.add_argument(
        "--semantic-target",
        type=str,
        choices=("online", "fixed", "raw"),
        default="online",
        help=(
            "Prediction target for the GA-GRU semantic loss. 'online' uses the trainable "
            "semantic projection (the target drifts because classification optimises it, so the "
            "loss may not converge). 'fixed' uses a frozen projection of the raw node features, "
            "a stationary low-dimensional target. 'raw' reconstructs the original, unnormalized "
            "node features with pure MSE while keeping a lower-dimensional GA-GRU state; use "
            "this for frozen BERT."
        ),
    )
    parser.add_argument(
        "--semantic-prediction-mode",
        choices=("absolute", "residual"),
        default=MODEL_HYPERPARAMETERS["semantic_prediction_mode"],
        help=(
            "GA-GRU edge prediction parameterization. 'absolute' directly predicts the "
            "opposite endpoint vector. 'residual' predicts a parent-child change and adds "
            "it to the source endpoint before applying the same semantic loss."
        ),
    )
    parser.add_argument(
        "--gagru-direction",
        choices=("bidirectional", "topdown", "bottomup"),
        default=MODEL_HYPERPARAMETERS["gagru_direction"],
        help=(
            "GA-GRU recursive direction ablation. A disabled direction is replaced by a "
            "same-width zero vector before fusion, preserving classifier dimensions."
        ),
    )
    parser.add_argument(
        "--gagru-aggregation",
        choices=("attention", "mean"),
        default=MODEL_HYPERPARAMETERS["gagru_aggregation"],
        help=(
            "Aggregate already-updated direct neighbors with learned attention or uniform "
            "mean weights. Both variants retain the same value projection and output width."
        ),
    )
    parser.add_argument(
        "--gagru-attention-query",
        choices=("global", "node"),
        default=MODEL_HYPERPARAMETERS["gagru_attention_query"],
        help=(
            "Attention query used by GA-GRU neighbor aggregation. 'global' preserves the "
            "original learned query; 'node' conditions each query on the current node."
        ),
    )
    parser.add_argument(
        "--gagru-attention-score-mode",
        choices=("dot", "additive"),
        default=MODEL_HYPERPARAMETERS["gagru_attention_score_mode"],
        help=(
            "GA-GRU neighbor-attention scoring. 'dot' preserves the original scaled "
            "dot product; 'additive' uses bounded query-key interactions."
        ),
    )
    parser.add_argument(
        "--gagru-attention-residual-alpha",
        type=float,
        default=MODEL_HYPERPARAMETERS["gagru_attention_residual_alpha"],
        help=(
            "Blend coefficient from projected neighbor mean (0) to attention context (1). "
            "The default 1 exactly preserves the original attention aggregation."
        ),
    )
    parser.add_argument(
        "--gagru-attention-residual-mode",
        choices=("fixed", "adaptive"),
        default=MODEL_HYPERPARAMETERS["gagru_attention_residual_mode"],
        help=(
            "Use a fixed mean-to-attention blend or learn a per-node gate anchored to the "
            "projected neighbor mean. Adaptive mode does not change mean aggregation."
        ),
    )
    parser.add_argument(
        "--gagru-attention-gate-init",
        type=float,
        default=MODEL_HYPERPARAMETERS["gagru_attention_gate_init"],
        help=(
            "Initial attention contribution for adaptive residual mode. Values near zero "
            "start close to uniform mean aggregation."
        ),
    )
    parser.add_argument(
        "--gagru-semantic-direction-reduction",
        choices=("sum", "mean"),
        default=MODEL_HYPERPARAMETERS["gagru_semantic_direction_reduction"],
        help=(
            "Combine active TD/BU semantic prediction losses by summation or mean. "
            "Mean keeps the auxiliary-loss scale comparable across direction ablations."
        ),
    )
    parser.add_argument(
        "--gagru-direction-fusion",
        choices=("joint", "td-residual"),
        default=MODEL_HYPERPARAMETERS["gagru_direction_fusion"],
        help=(
            "Combine directional states with the original joint LayerNorm or preserve "
            "the TD state and add an independently normalized, learnable BU residual."
        ),
    )
    parser.add_argument(
        "--gagru-bu-residual-init",
        type=float,
        default=MODEL_HYPERPARAMETERS["gagru_bu_residual_init"],
        help="Initial BU contribution for --gagru-direction-fusion td-residual.",
    )
    parser.add_argument(
        "--gagru-bu-shared-gradient-scale",
        type=float,
        default=MODEL_HYPERPARAMETERS["gagru_bu_shared_gradient_scale"],
        help=(
            "Scale BU gradients entering the shared semantic input projection. "
            "Use 0 to keep the TD representation isolated while retaining BU forward features."
        ),
    )
    parser.add_argument(
        "--gagru-readout",
        choices=("mean", "td-mean-bu-root", "evidence"),
        default=MODEL_HYPERPARAMETERS["gagru_readout"],
        help=(
            "Graph-level GA-GRU readout. 'evidence' keeps root and attends comments; 'mean' preserves the original all-node mean; "
            "'td-mean-bu-root' averages TD states but reads the recursively aggregated "
            "BU state directly from each graph root."
        ),
    )
    parser.add_argument(
        "--global-recon-real-edges-only",
        action="store_true",
        help=(
            "Restrict the masked-reconstruction encoder to the real propagation tree "
            "(no virtual two-hop or semantic-KNN edges). A semantic-KNN edge is built "
            "from node features, so using it to reconstruct a masked node would let the "
            "neighbourhood carry information about that node."
        ),
    )
    parser.add_argument(
        "--global-semantic-mask-ratio",
        type=float,
        default=MODEL_HYPERPARAMETERS["global_semantic_mask_ratio"],
        help="Node ratio masked for GAT global semantic prediction.",
    )
    parser.add_argument(
        "--global-semantic-target",
        choices=("latent", "raw"),
        default=MODEL_HYPERPARAMETERS["global_semantic_target"],
        help=(
            "GAT masked prediction target. 'latent' predicts deterministic, stop-gradient "
            "unmasked GAT node states with plain MSE. 'raw' reconstructs the original input "
            "features. This does not change GA-GRU targets."
        ),
    )
    parser.add_argument(
        "--global-semantic-sce-gamma",
        type=float,
        default=MODEL_HYPERPARAMETERS["global_semantic_sce_gamma"],
        help=(
            "Legacy compatibility option. The current GARD-style global semantic prediction "
            "uses a GAT decoder with masked-node MSE and ignores this value."
        ),
    )
    parser.add_argument(
        "--global-semantic-feature-std-floor",
        type=float,
        default=MODEL_HYPERPARAMETERS["global_semantic_feature_std_floor"],
        help=(
            "Minimum per-feature standard deviation used only by raw-target global "
            "reconstruction. Statistics are computed from the current training split only."
        ),
    )
    parser.add_argument(
        "--disable-global-feature-std",
        action="store_true",
        help="Use ordinary raw-feature MSE instead of train-split variance-normalized MSE.",
    )
    parser.add_argument(
        "--uniformity-temperature",
        type=float,
        default=MODEL_HYPERPARAMETERS["uniformity_temperature"],
        help="Gaussian potential temperature used by the event representation uniformity loss.",
    )
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=MODEL_HYPERPARAMETERS["edge_dropout"],
        help="Probability of dropping graph edges during training.",
    )
    parser.add_argument(
        "--partial-graph-train-prob",
        type=float,
        default=TRAINING_HYPERPARAMETERS["partial_graph_train_prob"],
        help=(
            "Training-only probability of replacing an event with an "
            "ancestor-preserving partial subtree; zero disables augmentation."
        ),
    )
    parser.add_argument(
        "--partial-graph-keep-ratios",
        type=float,
        nargs="+",
        default=list(TRAINING_HYPERPARAMETERS["partial_graph_keep_ratios"]),
        help=(
            "Candidate retained non-root node ratios for stochastic partial-event "
            "training, for example: 0.7 0.8 0.9."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=TRAINING_HYPERPARAMETERS["train_ratio"])
    parser.add_argument("--val-ratio", type=float, default=TRAINING_HYPERPARAMETERS["val_ratio"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_HYPERPARAMETERS["num_workers"])
    parser.add_argument("--patience", type=int, default=TRAINING_HYPERPARAMETERS["patience"])
    parser.add_argument("--grad-clip", type=float, default=TRAINING_HYPERPARAMETERS["grad_clip"])
    parser.add_argument("--seed", type=int, default=TRAINING_HYPERPARAMETERS["seed"])
    parser.add_argument("--split-seed", type=int, default=TRAINING_HYPERPARAMETERS["split_seed"])
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >1, run stratified K-fold cross-validation and report mean+/-std test metrics instead of a single split.",
    )
    parser.add_argument(
        "--cv-fold-index",
        type=int,
        default=0,
        help=(
            "Run only this one-based cross-validation fold (0 runs every fold). "
            "Useful for testing fold-specific text features before generating all folds."
        ),
    )
    parser.add_argument(
        "--cv-fold-dataset-root-template",
        type=str,
        default=None,
        help=(
            "Optional fold-specific dataset root containing a {fold} placeholder, for example "
            "D:/data/pheme_lora/fold{fold}. Each fold is split identically but reads its own "
            "leakage-free node features."
        ),
    )
    parser.add_argument(
        "--cv-protocol",
        type=str,
        choices=("auto", "standard", "bigcn"),
        default="auto",
        help=(
            "Cross-validation protocol. 'standard' uses an independent validation subset; "
            "'bigcn' reproduces BiGCN's class-wise 80/20 five-fold split and uses the test "
            "fold loss for checkpointing/early stopping; 'auto' selects bigcn for Pheme "
            "five-fold runs and standard otherwise."
        ),
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="",
        help=(
            "Explicit train:val:test ratios, e.g. '8:1:1' or '0.8:0.1:0.1'. Requires "
            "--cv-protocol standard. The splitter carves test = 1/n_folds per class and "
            "val = a share of what remains, so 8:1:1 is realised as --cv-folds 10 with "
            "val_ratio 1/9; both are derived and printed. Run one fold index for a "
            "single 80/10/10 split, or folds 1..10 for ten-fold cross-validation."
        ),
    )
    parser.add_argument(
        "--bigcn-inner-val-ratio",
        type=float,
        default=TRAINING_HYPERPARAMETERS["bigcn_inner_val_ratio"],
        help=(
            "Optional validation fraction carved from each BiGCN outer training fold. "
            "A value such as 0.1 keeps the original outer test IDs but selects checkpoints "
            "on an internal validation set. The default 0 preserves legacy BiGCN test-fold selection."
        ),
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "Weight EMA decay in (0,1). 0 disables EMA. When enabled, validation/test/checkpoint "
            "use the exponential moving average of the training weights, which smooths late-training "
            "overfitting. For short runs try 0.99-0.995; for long runs 0.999."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("none", "cosine", "plateau"),
        default="none",
        help=(
            "Learning-rate schedule. 'cosine' anneals from --lr to --min-lr over --epochs "
            "(with optional --warmup-epochs). 'plateau' halves the LR when the validation "
            "checkpoint metric stops improving. 'none' keeps the LR constant."
        ),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup epochs before cosine annealing kicks in.",
    )
    parser.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR decay factor for the plateau scheduler.",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=3,
        help="Epochs without validation improvement before the plateau scheduler decays the LR.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Lower LR bound for the cosine / plateau schedulers.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default=RUNTIME_HYPERPARAMETERS["save_dir"])
    parser.add_argument(
        "--max-graph-nodes",
        type=int,
        default=RUNTIME_HYPERPARAMETERS["max_graph_nodes"],
        help="Maximum nodes kept per graph after BFS truncation. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--early-cutoff-minutes",
        type=float,
        default=None,
        help=(
            "PHEME early-detection deadline in minutes. 0 keeps only the source post; positive "
            "values keep timestamped replies available by the deadline. Requires time metadata."
        ),
    )
    parser.add_argument(
        "--disable-gat",
        action="store_true",
        help="Legacy alias for --structure-encoder none.",
    )
    parser.add_argument("--label-smoothing", type=float, default=TRAINING_HYPERPARAMETERS["label_smoothing"])
    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["semantic_loss_weight"],
        help="Weight for GA-GRU parent-to-child and child-to-parent semantic prediction loss.",
    )
    parser.add_argument(
        "--legacy-eval-semantic-reconstruction",
        action="store_true",
        help=(
            "Compatibility switch for historical GA-GRU experiments: also execute and "
            "report the local semantic-prediction objective during validation and test. "
            "It does not backpropagate auxiliary losses or use the outer test fold for "
            "checkpoint selection."
        ),
    )
    parser.add_argument(
        "--semantic-loss-decay-start-epoch",
        type=int,
        default=TRAINING_HYPERPARAMETERS["semantic_loss_decay_start_epoch"],
        help=(
            "Last epoch that keeps the full --semantic-loss-weight. Use together with "
            "--semantic-loss-decay-end-epoch; both 0 preserve a constant weight."
        ),
    )
    parser.add_argument(
        "--semantic-loss-decay-end-epoch",
        type=int,
        default=TRAINING_HYPERPARAMETERS["semantic_loss_decay_end_epoch"],
        help=(
            "Epoch at which the GA-GRU semantic loss weight reaches 0 by linear decay. "
            "Must be greater than --semantic-loss-decay-start-epoch."
        ),
    )
    parser.add_argument(
        "--global-semantic-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["global_semantic_loss_weight"],
        help="Weight for GAT root-aware masked global semantic forecasting loss.",
    )
    parser.add_argument(
        "--gat-pretrain-epochs",
        type=int,
        default=TRAINING_HYPERPARAMETERS["gat_pretrain_epochs"],
        help="Masked-reconstruction-only pretraining epochs for the shared GAT encoder/decoder.",
    )
    parser.add_argument(
        "--gat-pretrain-lr",
        type=float,
        default=TRAINING_HYPERPARAMETERS["gat_pretrain_lr"],
        help="GAT pretraining learning rate. 0 reuses --lr.",
    )
    parser.add_argument(
        "--global-loss-warmup-epochs",
        type=int,
        default=TRAINING_HYPERPARAMETERS["global_loss_warmup_epochs"],
        help=(
            "Linearly increase the global reconstruction weight from 0 to "
            "--global-semantic-loss-weight over this many supervised epochs."
        ),
    )
    parser.add_argument(
        "--gat-aux-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["gat_aux_loss_weight"],
        help=(
            "Weight for an auxiliary supervised classifier on the normalized GAT branch. "
            "Use a small value such as 0.1 to prevent concatenation from ignoring GAT."
        ),
    )
    parser.add_argument(
        "--uniformity-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["uniformity_loss_weight"],
        help="Weight for the GARD-style event representation uniformity regularizer.",
    )
    parser.add_argument(
        "--supcon-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["supcon_loss_weight"],
        help=(
            "Weight for the cross-event supervised contrastive loss on the fused "
            "event representation. 0 disables it. First trial: 0.05."
        ),
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=TRAINING_HYPERPARAMETERS["supcon_temperature"],
        help="SupCon temperature tau applied to scaled cosine similarities.",
    )
    parser.add_argument(
        "--supcon-projection-dim",
        type=int,
        default=MODEL_HYPERPARAMETERS["supcon_projection_dim"],
        help=(
            "Output dimension of the train-only SupCon projection head on the fused "
            "representation. 0 disables the head; 64 matches the first SupCon trial."
        ),
    )
    parser.add_argument(
        "--supcon-class-balanced",
        action="store_true",
        help=(
            "Sample training batches with a fixed per-class quota so every sample has "
            "cross-event same-label positives. Requires batch_size >= num_classes and "
            "conflicts with --partial-graph-train-prob."
        ),
    )
    parser.add_argument(
        "--node-representation",
        type=str,
        choices=("independent", "relation", "gate", "residual-norm"),
        default=MODEL_HYPERPARAMETERS["node_representation"],
        help=(
            "How the node vector is built from the two frozen text views. "
            "'independent' = plain V1G (baseline). 'relation' = source-conditioned "
            "pair encoding only ('[CLS] source [SEP] reply [SEP]'). 'gate' = "
            "g*c + (1-g)*r with g = sigmoid(W[c;r]). 'residual-norm' = "
            "LayerNorm(W_c c + W_r r). The three non-default modes require a "
            "dual-view dataset providing semantic_x."
        ),
    )
    parser.add_argument(
        "--temporal-consistency",
        action="store_true",
        help=(
            "Future-aware self-supervision: sample an early propagation stage per "
            "training event and require its representation to predict the completed "
            "event's representation (1 - cos, stop-gradient target). Plain CE plus "
            "an optional early-stage CE; no stance or pseudo labels."
        ),
    )
    parser.add_argument(
        "--temporal-snapshot-mode",
        choices=("ratio", "time"),
        default=MODEL_HYPERPARAMETERS["temporal_snapshot_mode"],
        help=(
            "How the early stage is cut: 'ratio' keeps the earliest share of replies "
            "in timestamp order, 'time' keeps replies younger than a cutoff in minutes."
        ),
    )
    parser.add_argument(
        "--temporal-snapshot-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated reply-keep shares sampled for the early stage.",
    )
    parser.add_argument(
        "--temporal-cutoffs-minutes",
        type=str,
        default="20,40,60,80",
        help="Comma-separated minute cutoffs sampled when --temporal-snapshot-mode time.",
    )
    parser.add_argument(
        "--future-loss-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["future_loss_weight"],
        help="Weight of the early-to-full consistency loss (first trial: 0.05).",
    )
    parser.add_argument(
        "--early-ce-weight",
        type=float,
        default=TRAINING_HYPERPARAMETERS["early_ce_weight"],
        help=(
            "Weight of the cross-entropy on the early stage's own logits. 0 means the "
            "early view is trained only by the consistency loss (cleanest first test)."
        ),
    )
    parser.add_argument(
        "--hierarchical-evidence",
        action="store_true",
        help=(
            "Replace the dual-branch detector with the Hierarchical Evidence Network: "
            "events become bags of source-reply subtrees, encoded by per-branch MS-GAT "
            "+ GA-GRU, related by a branch transformer, and aggregated by latent "
            "evidence slots. Uses plain CE only; GAT/GA-GRU branch flags are ignored."
        ),
    )
    parser.add_argument(
        "--hierarchical-slots",
        type=int,
        default=MODEL_HYPERPARAMETERS["hierarchical_slots"],
        help="Number of latent evidence slots in the hierarchical evidence network.",
    )
    parser.add_argument(
        "--claim-response-relation",
        action="store_true",
        help=(
            "Keep the V1G content branch untouched and add a claim-response relation "
            "branch: replies are re-encoded against their source by a bilinear "
            "interaction and propagated over the real tree by a 2-layer GAT, then "
            "gate-fused before the content classifier. Plain CE only."
        ),
    )
    parser.add_argument(
        "--relation-gate-bias",
        type=float,
        default=MODEL_HYPERPARAMETERS["relation_gate_bias"],
        help=(
            "Initial bias of the relation/content fusion gate logit. 3.0 keeps the "
            "model content-dominant (gate~0.95) at initialization because the "
            "relation projection starts at zero."
        ),
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        default=TRAINING_HYPERPARAMETERS["disable_class_weights"],
        help="Disable inverse-frequency class weights in cross entropy.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=("ce", "focal"),
        default=TRAINING_HYPERPARAMETERS["loss_type"],
        help="Training loss: standard cross entropy or focal loss.",
    )
    parser.add_argument("--focal-gamma", type=float, default=TRAINING_HYPERPARAMETERS["focal_gamma"])
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=TRAINING_HYPERPARAMETERS["focal_alpha"],
        help="Binary focal alpha for positive class. Use a negative value to disable alpha balancing.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=("accuracy", "macro_f1", "f1", "loss"),
        default=TRAINING_HYPERPARAMETERS["checkpoint_metric"],
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--selection-split",
        type=str,
        choices=("auto", "validation", "test"),
        default="auto",
        help=(
            "Which split drives checkpoint selection, early stopping and the reported "
            "number. 'auto' keeps the historical behaviour: test for the BiGCN protocol "
            "without an inner validation split (legacy BiGCN selection), validation "
            "otherwise. 'test' reproduces the SAMGAT/GAT_pheme.py protocol on any split "
            "protocol -- every epoch is scored on the test fold, its metric drives "
            "patience, and the reported number is the best test epoch. WARNING: that is "
            "test-set model selection, so the resulting accuracy/macro-F1 are optimistically "
            "biased and are NOT comparable with validation-selected runs. Runs that use it "
            "are marked in config.json and in the metrics."
        ),
    )
    parser.add_argument(
        "--threshold-selection-metric",
        type=str,
        choices=("accuracy", "macro_f1", "f1"),
        default=TRAINING_HYPERPARAMETERS["threshold_selection_metric"],
        help="Validation metric used when searching the decision threshold.",
    )
    parser.add_argument("--threshold-search-start", type=float, default=TRAINING_HYPERPARAMETERS["threshold_search_start"])
    parser.add_argument("--threshold-search-end", type=float, default=TRAINING_HYPERPARAMETERS["threshold_search_end"])
    parser.add_argument("--threshold-search-step", type=float, default=TRAINING_HYPERPARAMETERS["threshold_search_step"])
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use a fixed rumor decision threshold instead of validation threshold search.",
    )
    parser.add_argument(
        "--disable-threshold-search",
        action="store_true",
        help="Disable validation threshold search and evaluate with threshold 0.50 unless --fixed-threshold is set.",
    )
    parser.add_argument(
        "--disable-edge-relations",
        action="store_true",
        help="Disable structural relation bias in graph attention for ablation.",
    )
    parser.add_argument(
        "--disable-semantic-edge-attention",
        action="store_true",
        help="Disable the parent-child semantic-change bias in GAT attention for ablation.",
    )
    explicit_args = set()
    raw_args = sys.argv[1:]
    for action in parser._actions:
        for option in action.option_strings:
            if option in raw_args or any(arg.startswith(f"{option}=") for arg in raw_args):
                explicit_args.add(action.dest)
    args = parser.parse_args()
    args._explicit_args = sorted(explicit_args)
    args.dataset_name = normalize_dataset_name(args.dataset_name)
    unsupported = []
    if args.structure_encoder != "gat":
        unsupported.append(f"structure_encoder={args.structure_encoder}")
    for enabled, name in (
        (args.parent_shuffle, "parent_shuffle"),
        (args.partial_graph_train_prob > 0.0, "partial_graph_train"),
        (args.hierarchical_evidence, "hierarchical_evidence"),
        (args.claim_response_relation, "claim_response_relation"),
        (args.temporal_consistency, "temporal_consistency"),
        (args.sibling_interaction, "sibling_interaction"),
        (args.claim_guided_attention, "claim_guided_attention"),
        (args.propagation_adapter_checkpoint is not None, "propagation_adapter"),
        (args.temporal_dynamics != "off", "temporal_dynamics"),
        (args.temporal_evolution, "temporal_evolution"),
        (args.contrastive_views, "contrastive_views"),
        (args.joint_lora, "joint_lora"),
    ):
        if enabled:
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "The compact release supports only the retained GA-GRU + GAT pipeline; "
            "removed experimental options were requested: " + ", ".join(unsupported)
        )
    args.semantic_hidden_dim = max(args.semantic_hidden_dim, 0)
    args.semantic_proj_dim = max(args.semantic_proj_dim, 0)
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    for branch_lr_name in ("gat_lr", "gagru_lr", "fusion_lr"):
        if getattr(args, branch_lr_name) < 0.0:
            raise ValueError(f"--{branch_lr_name.replace('_', '-')} must be non-negative.")
    if args.freeze_gagru and args.semantic_loss_weight > 0.0:
        raise ValueError("--freeze-gagru requires --semantic-loss-weight 0.")
    if args.disable_gat:
        if "structure_encoder" in explicit_args and args.structure_encoder != "none":
            raise ValueError("--disable-gat conflicts with a non-'none' --structure-encoder.")
        args.structure_encoder = "none"
    args.disable_gat = args.structure_encoder == "none"
    if args.gat_input_mode != "raw":
        if args.structure_encoder not in {"gat", "acm-gcn"}:
            raise ValueError(
                f"--gat-input-mode {args.gat_input_mode} requires --structure-encoder "
                "gat or acm-gcn."
            )
        if args.gat_input_mode == "semantic-residual" and args.semantic_target != "raw":
            raise ValueError("--gat-input-mode semantic-residual requires --semantic-target raw.")
        if args.global_semantic_target != "latent":
            raise ValueError(
                f"--gat-input-mode {args.gat_input_mode} requires --global-semantic-target latent."
            )
    if args.gat_readout != "mean" and args.structure_encoder not in {"gat", "acm-gcn"}:
        raise ValueError(
            f"--gat-readout {args.gat_readout} requires --structure-encoder gat or acm-gcn."
        )
    if args.gat_neighborhood != "local" and args.structure_encoder not in {"gat", "acm-gcn"}:
        raise ValueError(
            f"--gat-neighborhood {args.gat_neighborhood} requires --structure-encoder "
            "gat or acm-gcn."
        )
    if args.gat_topology == "star" and args.structure_encoder != "gat":
        raise ValueError("--gat-topology star requires --structure-encoder gat.")
    if args.gat_topology == "set" and args.structure_encoder != "set-transformer":
        raise ValueError("--gat-topology set requires --structure-encoder set-transformer.")
    if args.gat_semantic_k <= 0:
        raise ValueError("--gat-semantic-k must be positive.")
    if args.input_bottleneck_dim < 0:
        raise ValueError("--input-bottleneck-dim must be zero or positive.")
    if args.gat_global_event_node and args.structure_encoder not in {"gat", "acm-gcn"}:
        raise ValueError(
            "--gat-global-event-node requires --structure-encoder gat or acm-gcn."
        )
    if args.structure_encoder == "acm-gcn":
        if args.gat_relative_time_bias:
            raise ValueError(
                "--gat-relative-time-bias needs attention logits, which the ACM encoder "
                "does not have."
            )
        _validate_acm_channels(args.acm_channels)
        if args.acm_gate_mode != "learned":
            target = (
                args.acm_gate_mode[: -len("-only")]
                if args.acm_gate_mode.endswith("-only")
                else None
            )
            if target is not None and target not in _validate_acm_channels(args.acm_channels):
                raise ValueError(
                    f"--acm-gate-mode {args.acm_gate_mode} needs the {target!r} channel, "
                    f"which --acm-channels {args.acm_channels!r} does not include."
                )
    elif args.acm_channels != MODEL_HYPERPARAMETERS["acm_channels"]:
        raise ValueError("--acm-channels only applies to --structure-encoder acm-gcn.")
    if args.sibling_interaction:
        if args.structure_encoder not in {"gat", "acm-gcn"}:
            raise ValueError(
                "--sibling-interaction requires a graph structure branch "
                "(--structure-encoder gat or acm-gcn)."
            )
        if getattr(args, "sibling_gate_mode", "learned") == "fixed":
            if not 0.0 < args.sibling_gate_init <= 1.0:
                raise ValueError(
                    "--sibling-gate-init must be in (0, 1] when the gate is fixed."
                )
        elif not 0.0 < args.sibling_gate_init < 1.0:
            raise ValueError(
                "--sibling-gate-init must be in (0, 1) for a learned gate; use "
                "--sibling-gate-mode fixed to freeze gamma_sib at 1.0."
            )
    elif args.sibling_mode != MODEL_HYPERPARAMETERS["sibling_mode"]:
        raise ValueError("--sibling-mode only applies with --sibling-interaction.")
    if args.acm_gate_mode != "learned" and args.structure_encoder != "acm-gcn":
        raise ValueError("--acm-gate-mode only applies to --structure-encoder acm-gcn.")
    if args.gat_source_evidence_pooling:
        if not args.gat_global_event_node:
            raise ValueError(
                "--gat-source-evidence-pooling requires --gat-global-event-node."
            )
        if args.structure_encoder not in {"gat", "acm-gcn"}:
            raise ValueError(
                "--gat-source-evidence-pooling requires --structure-encoder gat or acm-gcn."
            )
    if args.gat_depth_encoding and args.structure_encoder not in {"gat", "acm-gcn"}:
        raise ValueError(
            "--gat-depth-encoding requires --structure-encoder gat or acm-gcn."
        )
    if args.gat_relative_time_bias and args.structure_encoder != "gat":
        raise ValueError("--gat-relative-time-bias requires --structure-encoder gat.")
    if args.gat_relative_time_bias and args.gat_neighborhood != "ms-gat-v1":
        raise ValueError("--gat-relative-time-bias requires --gat-neighborhood ms-gat-v1.")
    if args.source_residual and not 0.0 < args.source_residual_gate_init < 1.0:
        raise ValueError("--source-residual-gate-init must be in (0, 1).")
    if args.supcon_loss_weight > 0.0:
        if args.supcon_projection_dim <= 0:
            raise ValueError("--supcon-loss-weight requires --supcon-projection-dim > 0.")
        if args.supcon_temperature <= 0.0:
            raise ValueError("--supcon-temperature must be positive.")
    if args.supcon_projection_dim > 0 and args.supcon_loss_weight <= 0.0:
        raise ValueError(
            "--supcon-projection-dim has no effect without --supcon-loss-weight; "
            "enable the loss or drop the head."
        )
    if args.supcon_class_balanced and args.partial_graph_train_prob > 0.0:
        raise ValueError(
            "--supcon-class-balanced cannot be combined with --partial-graph-train-prob; "
            "partial-event views would break the fixed per-class batch quota."
        )
    if args.hierarchical_evidence and args.hierarchical_slots < 1:
        raise ValueError("--hierarchical-slots must be positive.")
    if args.hierarchical_evidence and args.claim_response_relation:
        raise ValueError(
            "--hierarchical-evidence and --claim-response-relation replace the model "
            "differently; enable at most one."
        )
    if args.claim_response_relation and args.relation_gate_bias > 20.0:
        raise ValueError("--relation-gate-bias above 20.0 makes the gate effectively closed.")
    if args.temporal_consistency:
        try:
            args.temporal_snapshot_ratios = tuple(
                float(value) for value in str(args.temporal_snapshot_ratios).split(",") if value.strip()
            )
            args.temporal_cutoffs_minutes = tuple(
                float(value) for value in str(args.temporal_cutoffs_minutes).split(",") if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "--temporal-snapshot-ratios / --temporal-cutoffs-minutes must be "
                "comma-separated numbers."
            ) from exc
        if not args.temporal_snapshot_ratios or any(
            not 0.0 < value <= 1.0 for value in args.temporal_snapshot_ratios
        ):
            raise ValueError("--temporal-snapshot-ratios must be in (0, 1].")
        if not args.temporal_cutoffs_minutes or any(
            value <= 0.0 for value in args.temporal_cutoffs_minutes
        ):
            raise ValueError("--temporal-cutoffs-minutes must be positive.")
        if args.future_loss_weight < 0.0 or args.early_ce_weight < 0.0:
            raise ValueError(
                "--future-loss-weight and --early-ce-weight must be non-negative."
            )
        if args.temporal_snapshot_mode == "time" and not args.disable_gat:
            pass  # time mode only needs node_time, which the metadata check below enforces.
    if args.structure_encoder == "acm-gcn":
        if args.gat_attention_type != "gat":
            raise ValueError(
                "ACM-GCN has no attention logits, so --gat-attention-type must stay 'gat'."
            )
    if args.structure_encoder not in {"gat", "acm-gcn"}:
        if args.gat_attention_type != "gat":
            raise ValueError(
                f"--gat-attention-type {args.gat_attention_type} requires --structure-encoder gat."
            )
        if args.gat_residual_mode != "fixed" or args.gat_residual_alpha != 1.0:
            raise ValueError(
                "Non-default --gat-residual-mode/--gat-residual-alpha values require "
                "--structure-encoder gat."
            )

    if args.joint_lora:
        if args.dataset_name != "Pheme":
            raise ValueError("--joint-lora is currently supported only with --dataset-name Pheme.")
        if args.joint_lora_scope == "gat" and args.structure_encoder != "gat":
            raise ValueError("--joint-lora-scope gat requires --structure-encoder gat.")
        if not args.node_token_root:
            raise ValueError("--joint-lora requires --node-token-root.")
        if args.joint_lora_rank <= 0:
            raise ValueError("--joint-lora-rank must be positive.")
        if args.joint_lora_alpha <= 0.0:
            raise ValueError("--joint-lora-alpha must be positive.")
        if not 0.0 <= args.joint_lora_dropout < 1.0:
            raise ValueError("--joint-lora-dropout must be in [0, 1).")
        if args.joint_lora_last_layers == 0 or args.joint_lora_last_layers < -1:
            raise ValueError("--joint-lora-last-layers must be -1 or a positive integer.")
        if args.joint_lora_text_batch_size <= 0:
            raise ValueError("--joint-lora-text-batch-size must be positive.")
        if args.joint_lora_lr <= 0.0:
            raise ValueError("--joint-lora-lr must be positive.")
        if args.joint_lora_freeze_epochs < 0:
            raise ValueError("--joint-lora-freeze-epochs must be non-negative.")
        if args.joint_lora_freeze_epochs >= args.epochs:
            raise ValueError("--joint-lora-freeze-epochs must be smaller than --epochs.")
        if args.joint_lora_weight_decay < 0.0:
            raise ValueError("--joint-lora-weight-decay must be non-negative.")
    else:
        if args.node_token_root:
            raise ValueError("--node-token-root has no effect unless --joint-lora is enabled.")
        if args.joint_lora_freeze_epochs > 0:
            raise ValueError("--joint-lora-freeze-epochs requires --joint-lora.")
        if args.joint_lora_scope != "all":
            raise ValueError("--joint-lora-scope has no effect unless --joint-lora is enabled.")

    if args.structure_encoder not in {"gat", "acm-gcn"}:
        gat_only_weights = {
            "global_semantic_loss_weight": "--global-semantic-loss-weight",
            "gat_pretrain_epochs": "--gat-pretrain-epochs",
            "gat_aux_loss_weight": "--gat-aux-loss-weight",
        }
        for key, option in gat_only_weights.items():
            if float(getattr(args, key)) <= 0.0:
                continue
            if key in explicit_args:
                raise ValueError(
                    f"--structure-encoder {args.structure_encoder} cannot be combined with {option} > 0."
                )
            setattr(args, key, 0 if key == "gat_pretrain_epochs" else 0.0)

    if args.set_transformer_latents <= 0:
        raise ValueError("--set-transformer-latents must be positive.")
    if args.set_transformer_layers <= 0:
        raise ValueError("--set-transformer-layers must be positive.")
    if args.graph_heads <= 0:
        raise ValueError("--graph-heads must be positive.")
    if args.structure_encoder == "set-transformer" and args.graph_hidden_dim % args.graph_heads != 0:
        raise ValueError(
            "--graph-hidden-dim must be divisible by --graph-heads for the Set Transformer."
        )
    # Derive cv_folds / val_ratio from --split-ratios before any validation that
    # depends on them (the fold-index checks below, for instance).
    args = apply_split_ratios(args)
    if args.cv_fold_index < 0:
        raise ValueError("--cv-fold-index must be 0 or a positive one-based fold number.")
    if args.cv_fold_index > 0:
        if args.cv_folds <= 1:
            raise ValueError("--cv-fold-index requires --cv-folds > 1.")
        if args.cv_fold_index > args.cv_folds:
            raise ValueError("--cv-fold-index cannot exceed --cv-folds.")
    if args.cv_fold_dataset_root_template:
        if args.cv_folds <= 1:
            raise ValueError("--cv-fold-dataset-root-template requires --cv-folds > 1.")
        try:
            fold_one_root = args.cv_fold_dataset_root_template.format(fold=1)
            fold_two_root = args.cv_fold_dataset_root_template.format(fold=2)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                "--cv-fold-dataset-root-template must be format-compatible with {fold}."
            ) from error
        if fold_one_root == fold_two_root:
            raise ValueError(
                "--cv-fold-dataset-root-template must contain a varying {fold} placeholder."
            )
    if not 0.0 <= args.bigcn_inner_val_ratio < 1.0:
        raise ValueError("--bigcn-inner-val-ratio must be in [0, 1).")
    if not 0.0 <= args.partial_graph_train_prob <= 1.0:
        raise ValueError("--partial-graph-train-prob must be in [0, 1].")
    if not args.partial_graph_keep_ratios or any(
        not 0.0 < ratio <= 1.0 for ratio in args.partial_graph_keep_ratios
    ):
        raise ValueError("--partial-graph-keep-ratios values must be in (0, 1].")
    if not 0.0 <= args.global_semantic_mask_ratio <= 1.0:
        raise ValueError("--global-semantic-mask-ratio must be in [0, 1].")
    if not 0.0 < args.gat_residual_alpha <= 1.0:
        raise ValueError("--gat-residual-alpha must be in (0, 1].")
    if not 0.0 <= args.gagru_attention_residual_alpha <= 1.0:
        raise ValueError("--gagru-attention-residual-alpha must be in [0, 1].")
    if not 0.0 < args.gagru_attention_gate_init < 1.0:
        raise ValueError("--gagru-attention-gate-init must be in (0, 1).")
    if not 0.0 < args.gagru_bu_residual_init < 1.0:
        raise ValueError("--gagru-bu-residual-init must be in (0, 1).")
    if not 0.0 <= args.gagru_bu_shared_gradient_scale <= 1.0:
        raise ValueError("--gagru-bu-shared-gradient-scale must be in [0, 1].")
    if args.semantic_loss_weight < 0.0:
        raise ValueError("--semantic-loss-weight must be non-negative.")
    semantic_decay_start = args.semantic_loss_decay_start_epoch
    semantic_decay_end = args.semantic_loss_decay_end_epoch
    if semantic_decay_start < 0 or semantic_decay_end < 0:
        raise ValueError("Semantic-loss decay epochs must be non-negative.")
    semantic_decay_enabled = semantic_decay_start > 0 or semantic_decay_end > 0
    if semantic_decay_enabled:
        if semantic_decay_start < 1:
            raise ValueError("--semantic-loss-decay-start-epoch must be at least 1 when decay is enabled.")
        if semantic_decay_end <= semantic_decay_start:
            raise ValueError(
                "--semantic-loss-decay-end-epoch must be greater than "
                "--semantic-loss-decay-start-epoch."
            )
        if semantic_decay_end > args.epochs:
            raise ValueError("--semantic-loss-decay-end-epoch cannot exceed --epochs.")
    if args.global_semantic_loss_weight < 0.0:
        raise ValueError("--global-semantic-loss-weight must be non-negative.")
    if args.global_semantic_feature_std_floor <= 0.0:
        raise ValueError("--global-semantic-feature-std-floor must be positive.")
    if args.gat_pretrain_epochs < 0:
        raise ValueError("--gat-pretrain-epochs must be non-negative.")
    if args.gat_pretrain_lr < 0.0:
        raise ValueError("--gat-pretrain-lr must be non-negative.")
    if args.global_loss_warmup_epochs < 0:
        raise ValueError("--global-loss-warmup-epochs must be non-negative.")
    if args.gat_aux_loss_weight < 0.0:
        raise ValueError("--gat-aux-loss-weight must be non-negative.")
    if args.uniformity_loss_weight < 0.0:
        raise ValueError("--uniformity-loss-weight must be non-negative.")
    if args.uniformity_temperature <= 0.0:
        raise ValueError("--uniformity-temperature must be positive.")
    if args.disable_gat and args.global_semantic_loss_weight > 0.0:
        raise ValueError("--disable-gat cannot be combined with --global-semantic-loss-weight > 0.")
    if args.disable_gat and args.gat_aux_loss_weight > 0.0:
        raise ValueError("--disable-gat cannot be combined with --gat-aux-loss-weight > 0.")
    if args.disable_gat and args.gat_pretrain_epochs > 0:
        raise ValueError("--disable-gat cannot be combined with --gat-pretrain-epochs > 0.")
    if args.gat_pretrain_epochs > 0 and args.global_semantic_mask_ratio <= 0.0:
        raise ValueError("--gat-pretrain-epochs requires --global-semantic-mask-ratio > 0.")
    if args.ema_decay != 0.0 and not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be 0 (disabled) or in (0, 1).")
    if args.semantic_loss_mse_weight < 0.0:
        raise ValueError("--semantic-loss-mse-weight must be non-negative.")
    if args.global_semantic_sce_gamma < 1.0:
        raise ValueError("--global-semantic-sce-gamma must be >= 1.")
    if args.semantic_sce_gamma < 1.0:
        raise ValueError("--semantic-sce-gamma must be >= 1.")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative.")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        raise ValueError("--lr-plateau-factor must be in (0, 1).")
    if args.min_lr < 0.0:
        raise ValueError("--min-lr must be non-negative.")
    if args.fixed_threshold is not None and not 0.0 < args.fixed_threshold < 1.0:
        raise ValueError(
            "--fixed-threshold must be strictly between 0 and 1; use 0.5 for standard binary classification."
        )
    if args.early_cutoff_minutes is not None and args.early_cutoff_minutes < 0.0:
        raise ValueError("--early-cutoff-minutes must be non-negative.")
    if (
        args.early_cutoff_minutes is not None
        and args.dataset_name not in {"Pheme", "Weibo"}
    ):
        raise ValueError(
            "--early-cutoff-minutes is currently supported only with "
            "--dataset-name Pheme or Weibo."
        )
    return args


def build_threshold_search_space(args: argparse.Namespace) -> list[float]:
    if args.threshold_search_step <= 0.0:
        raise ValueError("threshold_search_step must be positive.")
    if args.threshold_search_start > args.threshold_search_end:
        raise ValueError("threshold_search_start must be <= threshold_search_end.")

    thresholds = []
    current = args.threshold_search_start
    while current <= args.threshold_search_end + 1e-12:
        thresholds.append(round(float(current), 4))
        current += args.threshold_search_step
    return thresholds


def argument_was_explicit(args: argparse.Namespace, key: str) -> bool:
    return key in set(getattr(args, "_explicit_args", ()))


def uses_bigcn_inner_validation(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "cv_protocol", None) == "bigcn"
        and float(getattr(args, "bigcn_inner_val_ratio", 0.0) or 0.0) > 0.0
    )


def resolve_selection_split(
    args: argparse.Namespace,
    inner_validation: bool,
) -> Tuple[bool, bool]:
    """Decide whether the test fold drives checkpoint selection and patience.

    Returns ``(use_test_for_selection, fell_back_to_test)``.

    * ``auto`` (default) keeps the historical behaviour: the test fold is used for the
      BiGCN protocol without an inner validation split -- that is legacy BiGCN
      checkpointing, which the runners never used -- and the validation split otherwise.
    * ``test`` reproduces the SAMGAT / ``GAT_pheme.py`` protocol on any split protocol:
      the test fold is scored every epoch, its metric drives patience, and the reported
      number is the best test epoch.  That is test-set model selection, so the metrics
      are optimistically biased; run logs and ``config.json`` mark such runs.
    * ``validation`` asks for the honest protocol and falls back to the test fold only
      when the configuration has no validation split to select on.
    """
    request = str(getattr(args, "selection_split", "auto") or "auto").strip().lower()
    if request not in {"auto", "validation", "test"}:
        raise ValueError(
            f"selection_split must be 'auto', 'validation' or 'test', got {request!r}."
        )
    if request == "test":
        return True, False
    if request == "auto":
        legacy_test_selection = args.cv_protocol == "bigcn" and not inner_validation
        return legacy_test_selection, False
    # An explicit request for validation selection can only be honoured when a
    # validation split actually exists.
    no_validation_split = args.cv_protocol == "bigcn" and not inner_validation
    if no_validation_split:
        return True, True
    return False, False


def apply_split_ratios(args: argparse.Namespace) -> argparse.Namespace:
    """Turn an explicit train:val:test ratio into the CV knobs it implies.

    ``stratified_kfold`` builds a fold as "test = 1/n_folds of every class, val =
    val_ratio of what is left", so an 8:1:1 request means ``n_folds = 10`` and
    ``val_ratio = 1/9`` (a share of the 90% pool, not of the whole dataset).
    Deriving that here keeps the command line honest instead of hiding a magic
    0.1111 in every runner.
    """
    raw = str(getattr(args, "split_ratios", "") or "").strip()
    if not raw:
        return args
    parts = [piece.strip() for piece in raw.replace(",", ":").split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"--split-ratios expects train:val:test, e.g. 8:1:1, got {raw!r}."
        )
    try:
        values = [float(piece) for piece in parts]
    except ValueError as exc:
        raise ValueError(f"--split-ratios values must be numbers, got {raw!r}.") from exc
    if any(value <= 0.0 for value in values):
        raise ValueError("--split-ratios values must all be positive.")
    total = sum(values)
    train_ratio, val_ratio, test_ratio = (value / total for value in values)
    if args.cv_protocol == "bigcn":
        raise ValueError(
            "--split-ratios is incompatible with --cv-protocol bigcn: that protocol "
            "fixes the split at class-wise floor(20%) test folds. Use "
            "--cv-protocol standard."
        )
    folds = 1.0 / test_ratio
    if abs(folds - round(folds)) > 1e-6:
        raise ValueError(
            f"--split-ratios implies a test share of {test_ratio:.6f}, which is not 1/k; "
            "the fold-based splitter needs 1/2, 1/5, 1/10, ..."
        )
    if argument_was_explicit(args, "cv_folds") and int(args.cv_folds) != round(folds):
        raise ValueError(
            f"--split-ratios {raw} implies --cv-folds {round(folds)} but "
            f"--cv-folds {args.cv_folds} was also given; drop one of them."
        )
    args.cv_folds = int(round(folds))
    args.val_ratio = val_ratio / (train_ratio + val_ratio)
    args.split_ratios_summary = {
        "requested": raw,
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
        "cv_folds": args.cv_folds,
        "val_ratio_of_pool": args.val_ratio,
    }
    print(
        "Split ratios | "
        f"train={train_ratio:.4f} val={val_ratio:.4f} test={test_ratio:.4f} -> "
        f"cv_folds={args.cv_folds}, val_ratio={args.val_ratio:.6f} of the remaining pool"
    )
    return args


def apply_dataset_specific_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.dataset_name == "Twitter16":
        for key, value in TWITTER16_MODEL_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, MODEL_HYPERPARAMETERS[key]) == MODEL_HYPERPARAMETERS[key]:
                setattr(args, key, value)

        for key, value in TWITTER16_TRAINING_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, TRAINING_HYPERPARAMETERS[key]) == TRAINING_HYPERPARAMETERS[key]:
                setattr(args, key, value)

    if args.dataset_name == "Weibo":
        for key, value in WEIBO_MODEL_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, MODEL_HYPERPARAMETERS[key]) == MODEL_HYPERPARAMETERS[key]:
                setattr(args, key, value)

        for key, value in WEIBO_TRAINING_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, TRAINING_HYPERPARAMETERS[key]) == TRAINING_HYPERPARAMETERS[key]:
                setattr(args, key, value)

        for key, value in WEIBO_RUNTIME_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, RUNTIME_HYPERPARAMETERS[key]) == RUNTIME_HYPERPARAMETERS[key]:
                setattr(args, key, value)

    if args.dataset_name == "PhemeRaw":
        for key, value in PHEMERAW_MODEL_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, MODEL_HYPERPARAMETERS[key]) == MODEL_HYPERPARAMETERS[key]:
                setattr(args, key, value)

        for key, value in PHEMERAW_TRAINING_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, TRAINING_HYPERPARAMETERS[key]) == TRAINING_HYPERPARAMETERS[key]:
                setattr(args, key, value)

    if args.dataset_name == "Weibo21":
        for key, value in WEIBO21_MODEL_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, MODEL_HYPERPARAMETERS[key]) == MODEL_HYPERPARAMETERS[key]:
                setattr(args, key, value)

        for key, value in WEIBO21_TRAINING_OVERRIDES.items():
            if not argument_was_explicit(args, key) and getattr(args, key, TRAINING_HYPERPARAMETERS[key]) == TRAINING_HYPERPARAMETERS[key]:
                setattr(args, key, value)

    if args.cv_protocol == "auto":
        args.cv_protocol = "bigcn" if args.dataset_name == "Pheme" and args.cv_folds == 5 else "standard"

    if args.cv_protocol == "bigcn":
        if args.cv_folds != 5:
            raise ValueError("--cv-protocol bigcn requires --cv-folds 5.")
        args.fixed_threshold = 0.5
        args.disable_threshold_search = True
        if uses_bigcn_inner_validation(args):
            args.val_ratio = float(args.bigcn_inner_val_ratio)
            if not argument_was_explicit(args, "checkpoint_metric"):
                args.checkpoint_metric = "macro_f1"
        else:
            # Legacy BiGCN evaluates each test fold every epoch and selects the
            # minimum-loss checkpoint. Keep that as the default, while honoring
            # an explicitly requested metric for controlled diagnostic runs.
            args.val_ratio = 0.0
            if not argument_was_explicit(args, "checkpoint_metric"):
                args.checkpoint_metric = "loss"
    elif float(getattr(args, "bigcn_inner_val_ratio", 0.0) or 0.0) > 0.0:
        raise ValueError("--bigcn-inner-val-ratio requires --cv-protocol bigcn.")
    return args


def run_epoch(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 5.0,
    class_weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    label_smoothing: float = 0.0,
    semantic_loss_weight: float = 0.0,
    global_semantic_loss_weight: float = 0.0,
    gat_aux_loss_weight: float = 0.0,
    uniformity_loss_weight: float = 0.0,
    mask_consistency_loss_weight: float = 0.0,
    mask_consistency_ratio: float = 0.169,
    supcon_loss_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    contrastive_loss_weight: float = 0.0,
    measure_contrastive_grad_ratio: bool = False,
    future_loss_weight: float = 0.0,
    early_ce_weight: float = 0.0,
    measure_sibling_delta: bool = False,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    ema: Optional[ModelEMA] = None,
    gradient_accumulation_steps: int = 1,
    eval_semantic_reconstruction: bool = False,
) -> Dict[str, object]:
    is_train = optimizer is not None
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    model.train(is_train)
    joint_text_encoder = getattr(model, "joint_text_encoder", None)
    if (
        is_train
        and joint_text_encoder is not None
        and not any(parameter.requires_grad for parameter in joint_text_encoder.parameters())
    ):
        # Frozen warmup should reproduce the deterministic offline feature stream,
        # rather than perturbing it with Transformer and LoRA dropout.
        joint_text_encoder.eval()
    reconstruction_setter = getattr(model, "set_global_semantic_reconstruction_enabled", None)
    if callable(reconstruction_setter):
        reconstruction_setter(is_train and global_semantic_loss_weight > 0.0)

    local_setter = getattr(model, "set_semantic_reconstruction_enabled", None)
    if callable(local_setter):
        # The GARD-style decoders exist only to densify training supervision, so
        # evaluation skips them entirely: training gets more supervision, inference
        # cost is unchanged.  ``set_semantic_reconstruction_enabled`` still forces
        # them on when gat_input_mode consumes their predictions.
        local_setter(
            semantic_loss_weight > 0.0
            and (is_train or eval_semantic_reconstruction)
        )
    semantic_encoder = getattr(model, "semantic_encoder", None)
    if (
        is_train
        and semantic_encoder is not None
        and not any(parameter.requires_grad for parameter in semantic_encoder.parameters())
    ):
        # A frozen Stage-1 representation must also have deterministic dropout.
        semantic_encoder.eval()
    gradient_sums, parameter_norm_sums, update_ratio_sums, gradient_steps = {}, {}, {}, 0
    parameter_lrs: Dict[int, float] = {}
    if is_train:
        for optimizer_group in optimizer.param_groups:
            group_lr = float(optimizer_group["lr"])
            for parameter in optimizer_group["params"]:
                parameter_lrs[id(parameter)] = group_lr
    total_loss = 0.0
    total_final_loss = 0.0
    total_semantic_loss = 0.0
    total_semantic_child_loss = 0.0
    total_semantic_parent_loss = 0.0
    total_child_cosine_gap = 0.0
    total_parent_cosine_gap = 0.0
    has_semantic_directions = False
    has_cosine_gap = False
    total_global_semantic_loss = 0.0
    total_gat_aux_loss = 0.0
    total_mask_consistency_loss = 0.0
    total_mask_consistency_ce = 0.0
    mask_consistency_logits = None
    total_uniformity_loss = 0.0
    total_incon_contrib = 0.0
    total_semantic_drift = 0.0
    total_branch_abs_cosine = 0.0
    total_time_gate_mean = 0.0
    total_time_bias_semantic = 0.0
    total_time_gate_per_head: Optional[torch.Tensor] = None
    # V1G-R1 source-anchored residual diagnostics.
    total_source_residual_gamma = 0.0
    total_correction_ratio = 0.0
    total_source_logit_norm = 0.0
    total_correction_logit_norm = 0.0
    all_source_logits = []
    has_source_residual = False
    # Cross-event SupCon diagnostics.
    total_supcon_loss = 0.0
    all_fused = []
    has_supcon = False
    # Future-aware self-supervision: early-stage consistency and early accuracy.
    total_future_loss = 0.0
    total_early_loss = 0.0
    all_snapshot_logits = []
    has_future_loss = False
    has_early_ce = False
    # Claim-response relation branch: how much the gate actually uses it.
    total_relation_gate = 0.0
    total_relation_norm = 0.0
    has_relation_branch = False
    # ACM channel gates: the evidence for which propagation channels matter.
    acm_gate_sums: Dict[str, float] = {}
    acm_channel_names: list = []
    acm_zero_mass_total = 0.0
    has_acm_zero_mass = False
    # Claim-guided evidence attention: is the evidence view doing anything?
    claim_entropy_total = 0.0
    claim_on_root_total = 0.0
    claim_norm_total = 0.0
    claim_free_total = 0.0
    has_claim_attention = False
    # Same-event contrastive regularisation.
    con_loss_total = 0.0
    con_pos_total = 0.0
    con_neg_total = 0.0
    has_contrastive = False
    grad_ratio_value: Optional[float] = None
    grad_ratio_batches_measured = 0
    # Decision-level influence of the sibling branch, on the SAME weights.
    sibling_logit_l1_total = 0.0
    sibling_logit_samples = 0
    # Sibling-set interaction: is the branch actually used?
    sibling_gate_total = 0.0
    sibling_entropy_total = 0.0
    sibling_group_total = 0.0
    sibling_share_total = 0.0
    sibling_relative_total = 0.0
    has_sibling_relative = False
    has_sibling = False
    has_semantic_loss = False
    has_global_semantic_loss = False
    has_gat_aux_loss = False
    has_mask_consistency = False
    has_uniformity_loss = False
    has_inconsistency = False
    has_branch_abs_cosine = False
    has_time_diagnostics = False
    edge_family_totals = [0.0, 0.0, 0.0]
    has_edge_families = False
    masked_input_total = 0.0
    has_masked_input = False
    has_temporal_repr = False
    temporal_contribution_total = 0.0
    temporal_weight_total = 0.0
    has_temporal_evolution = False
    temporal_evolution_totals = {
        "temporal_residual_ratio": 0.0,
        "temporal_gate_mean": 0.0,
        "temporal_drift_ratio": 0.0,
        "temporal_decay_mean": 0.0,
    }
    all_logits = []
    all_labels = []

    total_batches = len(data_loader)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(data_loader):
        labels = batch["labels"].to(device)
        outputs = model(batch, device)
        source_logits = outputs.get("source_logits")
        if source_logits is not None:
            # Detached diagnostics only: the correction ratio and norms must not
            # extend autograd, and CPU copies must not pin the training graph.
            delta_logits = outputs["delta_logits"].detach()
            gamma_value = outputs["source_residual_gamma"].detach()
            source_detached = source_logits.detach()
            scaled_correction = gamma_value * delta_logits
            correction_ratio = scaled_correction.norm(dim=-1) / (
                source_detached.norm(dim=-1) + 1e-8
            )
            batch_size = labels.size(0)
            all_source_logits.append(source_detached.cpu())
            total_source_residual_gamma += float(gamma_value) * batch_size
            total_correction_ratio += float(correction_ratio.mean()) * batch_size
            total_source_logit_norm += (
                float(source_detached.norm(dim=-1).mean()) * batch_size
            )
            total_correction_logit_norm += (
                float(delta_logits.norm(dim=-1).mean()) * batch_size
            )
            has_source_residual = True
        final_logits = outputs["logits"]
        logits = final_logits
        if measure_sibling_delta and getattr(model, "sibling_sets", None) is not None:
            # D_logit = mean L1 distance between the full decision and the same
            # checkpoint with gamma_sib = 0. Reported only where the extra cost is
            # acceptable (validation / test), never during training.
            model.set_sibling_gate(0.0)
            try:
                with torch.no_grad():
                    zero_logits = model(batch, device)["logits"].detach()
            finally:
                model.set_sibling_gate(None)
            sibling_logit_l1_total += float(
                (final_logits.detach() - zero_logits).abs().sum(dim=-1).mean()
            ) * labels.size(0)
            sibling_logit_samples += int(labels.size(0))
        if measure_sibling_delta and getattr(model, "sibling_event", None) is not None:
            model.set_sibling_gate(0.0)
            try:
                with torch.no_grad():
                    zero_logits = model(batch, device)["logits"].detach()
            finally:
                model.set_sibling_gate(None)
            sibling_logit_l1_total += float(
                (final_logits.detach() - zero_logits).abs().sum(dim=-1).mean()
            ) * labels.size(0)
            sibling_logit_samples += int(labels.size(0))
        final_loss = compute_classification_loss(
            logits=final_logits,
            labels=labels,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        loss = final_loss
        total_final_loss += float(final_loss.detach().cpu().item()) * labels.size(0)
        if mask_consistency_loss_weight > 0.0 and model.training:
            # Clean/mask consistency (R-Drop applied to the semantic masking this model
            # already uses): the same event is classified twice, once with the full node
            # features and once with a 16.9% semantic mask, and the two class distributions
            # are pulled together by a symmetric KL. The CE is averaged over the two views,
            # which is what makes this R-Drop rather than a plain regulariser.
            masked_graph = model.masked_input_view(
                batch["batched_graph"], mask_consistency_ratio, device
            )
            if masked_graph is not None:
                # The masked view must carry exactly the mask built above: with M2a's ratio
                # left in place the model would draw a second independent mask on top of it
                # and the two views would be ~30% masked rather than 16.9%.
                saved_input_ratio = model.masked_input_ratio
                model.masked_input_ratio = 0.0
                try:
                    masked_logits = model({"batched_graph": masked_graph}, device)["logits"]
                finally:
                    model.masked_input_ratio = saved_input_ratio
                masked_ce = compute_classification_loss(
                    logits=masked_logits,
                    labels=labels,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                clean_probabilities = torch.softmax(final_logits.detach().float(), dim=-1)
                masked_probabilities = torch.softmax(masked_logits.detach().float(), dim=-1)
                symmetric_kl = 0.5 * (
                    torch.nn.functional.kl_div(
                        masked_probabilities.clamp_min(1e-8).log(), clean_probabilities,
                        reduction="batchmean",
                    )
                    + torch.nn.functional.kl_div(
                        clean_probabilities.clamp_min(1e-8).log(), masked_probabilities,
                        reduction="batchmean",
                    )
                )
                loss = 0.5 * loss + 0.5 * masked_ce + mask_consistency_loss_weight * symmetric_kl
                has_mask_consistency = True
                total_mask_consistency_loss += float(symmetric_kl.detach()) * labels.size(0)
                total_mask_consistency_ce += float(masked_ce.detach()) * labels.size(0)
                mask_consistency_logits = masked_logits.detach()
        branch_abs_cosine = outputs.get("branch_abs_cosine")
        if branch_abs_cosine is not None:
            total_branch_abs_cosine += (
                float(branch_abs_cosine.detach().cpu().item()) * labels.size(0)
            )
            has_branch_abs_cosine = True
        time_gate_mean = outputs.get("time_gate_mean")
        time_gate_per_head = outputs.get("time_gate_per_head")
        time_bias_semantic = outputs.get("mean_abs_time_bias_semantic")
        if (
            time_gate_mean is not None
            and time_gate_per_head is not None
            and time_bias_semantic is not None
        ):
            batch_size = labels.size(0)
            total_time_gate_mean += float(time_gate_mean.detach().cpu().item()) * batch_size
            total_time_bias_semantic += (
                float(time_bias_semantic.detach().cpu().item()) * batch_size
            )
            head_values = time_gate_per_head.detach().cpu().float() * batch_size
            total_time_gate_per_head = (
                head_values if total_time_gate_per_head is None
                else total_time_gate_per_head + head_values
            )
            has_time_diagnostics = True
        incon_contrib = outputs.get("inconsistency_contribution_mean")
        semantic_drift = outputs.get("semantic_inconsistency_mean")
        if incon_contrib is not None and semantic_drift is not None:
            total_incon_contrib += float(incon_contrib.detach().cpu().item()) * labels.size(0)
            total_semantic_drift += float(semantic_drift.detach().cpu().item()) * labels.size(0)
            has_inconsistency = True
        if semantic_loss_weight > 0.0:
            semantic_loss = outputs.get("semantic_loss")
            if semantic_loss is not None:
                loss = loss + semantic_loss_weight * semantic_loss
                total_semantic_loss += float(semantic_loss.detach().cpu().item()) * labels.size(0)
                has_semantic_loss = True
                # GARD's local objective is MSE(parent->child) + MSE(child->parent);
                # recording both shows whether each direction is actually learning.
                predictions = outputs.get("semantic_prediction_outputs") or []
                if predictions:
                    first = predictions[0]
                    child = first.get("child_semantic_loss")
                    parent = first.get("parent_semantic_loss")
                    if child is not None and parent is not None:
                        total_semantic_child_loss += float(child.detach().cpu().item()) * labels.size(0)
                        total_semantic_parent_loss += float(parent.detach().cpu().item()) * labels.size(0)
                        has_semantic_directions = True
                    # d = 1 - cos(prediction, target): the same quantity under either criterion,
                    # so the MSE and cosine arms can be compared on it.
                    child_gap = first.get("child_semantic_cosine_gap")
                    parent_gap = first.get("parent_semantic_cosine_gap")
                    if child_gap is not None and parent_gap is not None:
                        total_child_cosine_gap += float(child_gap.detach().cpu().item()) * labels.size(0)
                        total_parent_cosine_gap += float(parent_gap.detach().cpu().item()) * labels.size(0)
                        has_cosine_gap = True
        if global_semantic_loss_weight > 0.0:
            global_semantic_loss = outputs.get("global_semantic_loss")
            if global_semantic_loss is not None:
                loss = loss + global_semantic_loss_weight * global_semantic_loss
                total_global_semantic_loss += (
                    float(global_semantic_loss.detach().cpu().item()) * labels.size(0)
                )
                has_global_semantic_loss = True
        if gat_aux_loss_weight > 0.0:
            gat_logits = outputs.get("gat_logits")
            if gat_logits is None:
                raise RuntimeError("GAT auxiliary loss requested, but the model did not return gat_logits.")
            gat_aux_loss = compute_classification_loss(
                logits=gat_logits,
                labels=labels,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
            loss = loss + gat_aux_loss_weight * gat_aux_loss
            total_gat_aux_loss += float(gat_aux_loss.detach().cpu().item()) * labels.size(0)
            has_gat_aux_loss = True
        if uniformity_loss_weight > 0.0:
            uniformity_loss = outputs.get("uniformity_loss")
            if uniformity_loss is not None:
                loss = loss + uniformity_loss_weight * uniformity_loss
                total_uniformity_loss += float(uniformity_loss.detach().cpu().item()) * labels.size(0)
                has_uniformity_loss = True
        future_loss = outputs.get("future_loss")
        if future_loss is not None and future_loss_weight > 0.0:
            loss = loss + future_loss_weight * future_loss
            total_future_loss += float(future_loss.detach()) * labels.size(0)
            has_future_loss = True
        snapshot_logits = outputs.get("snapshot_logits")
        if snapshot_logits is not None:
            if early_ce_weight > 0.0:
                early_loss = compute_classification_loss(
                    logits=snapshot_logits,
                    labels=labels,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                loss = loss + early_ce_weight * early_loss
                total_early_loss += float(early_loss.detach()) * labels.size(0)
                has_early_ce = True
            all_snapshot_logits.append(snapshot_logits.detach().cpu())
        con_loss = outputs.get("contrastive_loss")
        if con_loss is not None and contrastive_loss_weight > 0.0:
            weighted_con = contrastive_loss_weight * con_loss
            loss = loss + weighted_con
            total_con = float(con_loss.detach()) * labels.size(0)
            con_loss_total += total_con
            con_pos_total += float(outputs["contrastive_positive_cosine"]) * labels.size(0)
            con_neg_total += float(outputs["contrastive_negative_cosine"]) * labels.size(0)
            has_contrastive = True
            # conGradRatio: how much gradient pressure the contrastive term puts on
            # the SHARED encoder relative to the classification term. Measured on one
            # batch per epoch so the extra backward stays off the hot path.
            if (
                measure_contrastive_grad_ratio
                and grad_ratio_batches_measured == 0
                and is_train
                and weighted_con.requires_grad
            ):
                shared = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and (
                        name.startswith("global_semantic.")
                        or name.startswith("semantic_encoder.")
                        or name.startswith("content.global_semantic.")
                        or name.startswith("content.semantic_encoder.")
                    )
                ]
                if shared:
                    ce_grads = torch.autograd.grad(
                        final_loss, shared, retain_graph=True, allow_unused=True
                    )
                    con_grads = torch.autograd.grad(
                        weighted_con, shared, retain_graph=True, allow_unused=True
                    )
                    ce_norm = sum(
                        float(g.detach().pow(2).sum()) for g in ce_grads if g is not None
                    ) ** 0.5
                    con_norm = sum(
                        float(g.detach().pow(2).sum()) for g in con_grads if g is not None
                    ) ** 0.5
                    grad_ratio_value = con_norm / max(ce_norm, 1e-12)
                    grad_ratio_batches_measured += 1
        if supcon_loss_weight > 0.0:
            contrastive_repr = outputs.get("contrastive_repr")
            if contrastive_repr is None:
                raise RuntimeError(
                    "SupCon loss requested, but the model did not return contrastive_repr."
                )
            supcon_loss = supervised_contrastive_loss(
                contrastive_repr,
                labels,
                supcon_temperature,
            )
            loss = loss + supcon_loss_weight * supcon_loss
            total_supcon_loss += float(supcon_loss.detach()) * labels.size(0)
            has_supcon = True
        fused_repr = outputs.get("fused_repr")
        if fused_repr is not None:
            # Detached copy for the within/between cosine diagnostics.
            all_fused.append(fused_repr.detach().cpu())
        channel_names = outputs.get("acm_channel_names")
        if channel_names:
            acm_channel_names = list(channel_names)
            batch_size = labels.size(0)
            zero_mass = outputs.get("acm_zero_channel_gate_mass")
            if zero_mass is not None:
                scalar = float(zero_mass.detach()) if torch.is_tensor(zero_mass) else float(zero_mass)
                acm_zero_mass_total += scalar * batch_size
                has_acm_zero_mass = True
            for name in channel_names:
                value = outputs.get(f"acm_gate_{name}")
                if value is not None:
                    # The batched path reports plain floats, the single-graph path
                    # tensors; both must be accepted here.
                    scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                    acm_gate_sums[name] = acm_gate_sums.get(name, 0.0) + scalar * batch_size
        sibling_gate = outputs.get("sibling_gate")
        if sibling_gate is not None:
            batch_size = labels.size(0)
            sibling_gate_total += float(sibling_gate) * batch_size
            sibling_entropy_total += float(outputs.get("sibling_attention_entropy", 0.0)) * batch_size
            sibling_group_total += float(outputs.get("sibling_group_count", 0.0)) * batch_size
            sibling_share_total += float(outputs.get("sibling_gated_node_share", 0.0)) * batch_size
            relative = outputs.get("sibling_relative_norm")
            if relative is not None:
                sibling_relative_total += float(relative) * batch_size
                has_sibling_relative = True
            has_sibling = True
        family_counts = outputs.get("gat_neighborhood_edge_counts")
        if family_counts is not None:
            counts = torch.as_tensor(family_counts).detach().to("cpu").reshape(-1)[:3]
            for index in range(int(counts.numel())):
                edge_family_totals[index] += float(counts[index]) * labels.size(0)
            has_edge_families = True
        masked_share = outputs.get("masked_input_share")
        if masked_share is not None:
            batch_size = labels.size(0)
            masked_input_total += float(masked_share) * batch_size
            has_masked_input = True
        if outputs.get("temporal_evolution"):
            # How much the propagation dynamics actually move the prediction away
            # from the M0 head on the complete event. Exactly zero means the
            # residual never left its initialisation.
            batch_size = labels.size(0)
            for key in temporal_evolution_totals:
                value = outputs.get(key)
                if value is not None:
                    temporal_evolution_totals[key] += float(value.detach()) * batch_size
            has_temporal_evolution = True
        temporal_contribution = outputs.get("temporal_contribution")
        if temporal_contribution is not None:
            # ||W_T z_T|| / ||h_M0||: exactly zero means the branch is still at its
            # identity initialisation and any T1/T2 == T0 result would be vacuous.
            batch_size = labels.size(0)
            temporal_contribution_total += float(temporal_contribution) * batch_size
            temporal_head_weight_norm = outputs.get("temporal_head_weight_norm")
            temporal_weight_total += float(temporal_head_weight_norm) * batch_size
            has_temporal_repr = True
        claim_entropy = outputs.get("claim_attention_entropy")
        if claim_entropy is not None:
            batch_size = labels.size(0)
            claim_entropy_total += float(claim_entropy) * batch_size
            claim_on_root_total += float(outputs.get("claim_attention_on_root", 0.0)) * batch_size
            claim_norm_total += float(outputs.get("claim_evidence_norm", 0.0)) * batch_size
            claim_free_total += float(outputs.get("claim_reply_free_share", 0.0)) * batch_size
            has_claim_attention = True
        relation_gate = outputs.get("relation_gate_mean")
        if relation_gate is not None:
            batch_size = labels.size(0)
            total_relation_gate += float(relation_gate.detach()) * batch_size
            total_relation_norm += (
                float(outputs["relation_repr_norm"].detach()) * batch_size
            )
            has_relation_branch = True

        if is_train:
            window_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
            window_size = min(
                gradient_accumulation_steps,
                total_batches - window_start,
            )
            (loss / float(window_size)).backward()
            is_accumulation_boundary = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            )
            if is_accumulation_boundary:
                # Unscaled, pre-clipping gradients after the accumulation window.
                group_squares, group_parameter_squares, group_lrs = {}, {}, {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    # Attribute each branch's readout to that branch. This makes
                    # the reported norm match all trainable theta_GAT/theta_GRU
                    # parameters instead of placing both readouts in one mixed bin.
                    # Wrapped models nest the content branch under ``content.``.
                    if name.startswith("content."):
                        name = name[len("content."):]
                    branch_group = (
                        "relation" if (
                            name.startswith("relation_encoder.")
                            or name.startswith("relation_graph.")
                            or name.startswith("relation_projection.")
                            or name.startswith("fusion_gate.")
                        ) else
                        "gat" if (
                            name.startswith("global_semantic.")
                            or name.startswith("gat_residual_norm.")
                        ) else
                        "gagru" if name.startswith("semantic_encoder.") else
                        "fusion"
                    )
                    value = parameter.grad.detach().float().square().sum()
                    group_squares[branch_group] = group_squares.get(branch_group, 0) + value
                    parameter_value = parameter.detach().float().square().sum()
                    group_parameter_squares[branch_group] = (
                        group_parameter_squares.get(branch_group, 0) + parameter_value
                    )
                    group_lrs[branch_group] = parameter_lrs.get(id(parameter), float(optimizer.defaults["lr"]))
                    # Retain the legacy combined readout metric for existing
                    # diagnostics while also counting each readout in its branch.
                    if "evidence_readout" in name:
                        group_squares["readout"] = group_squares.get("readout", 0) + value
                for group, value in group_squares.items():
                    gradient_norm = float(value.sqrt())
                    gradient_sums[group] = gradient_sums.get(group, 0.0) + gradient_norm
                    if group in group_parameter_squares:
                        parameter_norm = float(group_parameter_squares[group].sqrt())
                        parameter_norm_sums[group] = (
                            parameter_norm_sums.get(group, 0.0) + parameter_norm
                        )
                        update_ratio_sums[group] = update_ratio_sums.get(group, 0.0) + (
                            group_lrs[group] * gradient_norm / max(parameter_norm, 1e-12)
                        )
                gradient_steps += 1
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(logits, labels, threshold=threshold)
    metrics["loss"] = total_loss / max(labels.size(0), 1)
    sample_count = max(labels.size(0), 1)
    metrics["final_loss"] = total_final_loss / sample_count
    if has_source_residual and all_source_logits:
        # V1G-R1 correction matrix: does propagation evidence fix source mistakes
        # without breaking source-correct samples? Net correction is the metric
        # to watch (wrong->correct minus correct->wrong).
        source_logits_all = torch.cat(all_source_logits, dim=0)
        source_correct = source_logits_all.argmax(dim=-1) == labels
        full_correct = logits.argmax(dim=-1) == labels
        metrics["source_acc"] = float(source_correct.float().mean())
        metrics["correction_cc"] = float(torch.count_nonzero(source_correct & full_correct))
        metrics["correction_wc"] = float(torch.count_nonzero(~source_correct & full_correct))
        metrics["correction_cw"] = float(torch.count_nonzero(source_correct & ~full_correct))
        metrics["correction_ww"] = float(torch.count_nonzero(~source_correct & ~full_correct))
        metrics["correction_net"] = metrics["correction_wc"] - metrics["correction_cw"]
        metrics["source_residual_gamma"] = total_source_residual_gamma / sample_count
        metrics["correction_ratio"] = total_correction_ratio / sample_count
        metrics["source_logit_norm"] = total_source_logit_norm / sample_count
        metrics["correction_logit_norm"] = total_correction_logit_norm / sample_count
    if has_future_loss:
        metrics["future_loss"] = total_future_loss / sample_count
    if has_early_ce:
        metrics["early_ce_loss"] = total_early_loss / sample_count
    if all_snapshot_logits:
        # How well the early propagation stage alone classifies the event.
        snapshot_all = torch.cat(all_snapshot_logits, dim=0)
        metrics["snapshot_acc"] = float(
            (snapshot_all.argmax(dim=-1) == labels).float().mean()
        )
    if has_supcon:
        metrics["supcon_loss"] = total_supcon_loss / sample_count
    if has_relation_branch:
        metrics["relation_gate_mean"] = total_relation_gate / sample_count
        metrics["relation_repr_norm"] = total_relation_norm / sample_count
    if acm_gate_sums:
        metrics["acm_channel_names"] = acm_channel_names
        for name, total in acm_gate_sums.items():
            metrics[f"acm_gate_{name}"] = total / sample_count
    if has_acm_zero_mass:
        metrics["acm_zero_channel_gate_mass"] = acm_zero_mass_total / sample_count
    if has_contrastive:
        metrics["contrastive_loss"] = con_loss_total / sample_count
        metrics["contrastive_positive_cosine"] = con_pos_total / sample_count
        metrics["contrastive_negative_cosine"] = con_neg_total / sample_count
        metrics["contrastive_gap"] = (
            metrics["contrastive_positive_cosine"] - metrics["contrastive_negative_cosine"]
        )
        if grad_ratio_value is not None:
            metrics["contrastive_grad_ratio"] = grad_ratio_value
    if has_claim_attention:
        metrics["claim_attention_entropy"] = claim_entropy_total / sample_count
        metrics["claim_attention_on_root"] = claim_on_root_total / sample_count
        metrics["claim_evidence_norm"] = claim_norm_total / sample_count
        metrics["claim_reply_free_share"] = claim_free_total / sample_count
    if has_sibling:
        metrics["sibling_gate"] = sibling_gate_total / sample_count
        metrics["sibling_attention_entropy"] = sibling_entropy_total / sample_count
        metrics["sibling_group_count"] = sibling_group_total / sample_count
        metrics["sibling_gated_node_share"] = sibling_share_total / sample_count
    if sibling_logit_samples:
        # D_logit is only measured on validation and test, so it is absent while
        # training; it must not gate the other sibling metrics.
        metrics["sibling_logit_l1"] = sibling_logit_l1_total / sibling_logit_samples
    if has_sibling_relative:
        metrics["sibling_relative_norm"] = sibling_relative_total / sample_count
    if has_edge_families:
        # Realised edge counts per family: a disabled family must read exactly 0.
        metrics["edge_family_real"] = edge_family_totals[0] / sample_count
        metrics["edge_family_2hop"] = edge_family_totals[1] / sample_count
        metrics["edge_family_semantic"] = edge_family_totals[2] / sample_count
    if has_masked_input:
        # Realised share of non-source nodes whose semantics the classifier cannot see.
        metrics["masked_input_share"] = masked_input_total / sample_count
    if has_temporal_repr:
        metrics["temporal_contribution"] = temporal_contribution_total / sample_count
        metrics["temporal_head_weight_norm"] = temporal_weight_total / sample_count
    if has_temporal_evolution:
        for key, total in temporal_evolution_totals.items():
            metrics[key] = total / sample_count
    if all_fused:
        # Cross-event representation diagnostics on the representation the
        # classifier actually consumes: S_within should rise under SupCon while
        # S_between stays clearly lower (collapse means S_within -> 1).
        fused_all = torch.cat(all_fused, dim=0)
        normalized = F.normalize(fused_all, dim=-1)
        similarity = normalized @ normalized.t()
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        off_diagonal = ~torch.eye(labels.size(0), dtype=torch.bool)
        within_mask = same_label & off_diagonal
        between_mask = ~same_label & off_diagonal
        if bool(within_mask.any()):
            metrics["s_within"] = float(similarity[within_mask].mean())
        if bool(between_mask.any()):
            metrics["s_between"] = float(similarity[between_mask].mean())
        if "s_within" in metrics and "s_between" in metrics:
            metrics["s_gap"] = metrics["s_within"] - metrics["s_between"]
    for group, value in gradient_sums.items():
        metrics["gradient_norm_" + group] = value / max(1, gradient_steps)
    for group, value in parameter_norm_sums.items():
        metrics[group + "_param_norm"] = value / max(1, gradient_steps)
    for group, value in update_ratio_sums.items():
        metrics[group + "_update_ratio"] = value / max(1, gradient_steps)
    if "gradient_norm_gat" in metrics:
        metrics["gat_grad_norm"] = metrics["gradient_norm_gat"]
    if "gradient_norm_gagru" in metrics:
        metrics["gagru_grad_norm"] = metrics["gradient_norm_gagru"]
    if "gat_grad_norm" in metrics and "gagru_grad_norm" in metrics:
        metrics["gat_gagru_grad_ratio"] = (
            metrics["gat_grad_norm"] / max(metrics["gagru_grad_norm"], 1e-12)
        )
    if has_semantic_loss:
        metrics["semantic_loss"] = total_semantic_loss / sample_count
    if has_semantic_directions:
        metrics["semantic_child_loss"] = total_semantic_child_loss / sample_count
        metrics["semantic_parent_loss"] = total_semantic_parent_loss / sample_count
    if has_cosine_gap:
        metrics["semantic_child_cosine_gap"] = total_child_cosine_gap / sample_count
        metrics["semantic_parent_cosine_gap"] = total_parent_cosine_gap / sample_count
    if has_global_semantic_loss:
        metrics["global_semantic_loss"] = total_global_semantic_loss / sample_count
    if has_gat_aux_loss:
        metrics["gat_aux_loss"] = total_gat_aux_loss / sample_count
    if has_mask_consistency:
        metrics["mask_consistency_kl"] = total_mask_consistency_loss / sample_count
        metrics["mask_consistency_masked_ce"] = total_mask_consistency_ce / sample_count
    if has_uniformity_loss:
        metrics["uniformity_loss"] = total_uniformity_loss / sample_count
    if has_inconsistency:
        metrics["inconsistency_contribution"] = total_incon_contrib / sample_count
        metrics["semantic_drift"] = total_semantic_drift / sample_count
    if has_branch_abs_cosine:
        metrics["branch_abs_cosine"] = total_branch_abs_cosine / sample_count
    if has_time_diagnostics and total_time_gate_per_head is not None:
        metrics["time_gate_mean"] = total_time_gate_mean / sample_count
        metrics["time_gate_per_head"] = (
            total_time_gate_per_head / sample_count
        ).tolist()
        metrics["mean_abs_time_bias_semantic"] = total_time_bias_semantic / sample_count
    metrics["logits"] = logits
    metrics["labels"] = labels
    return metrics


def compute_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
    loss_type: str,
    focal_gamma: float,
    focal_alpha: Optional[float],
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)

    if loss_type != "focal":
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    ce_loss = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    pt = torch.exp(-ce_loss)
    focal_factor = (1.0 - pt).clamp_min(0.0).pow(focal_gamma)

    if focal_alpha is not None and logits.size(-1) == 2:
        alpha_t = torch.where(
            labels == 1,
            torch.full_like(ce_loss, focal_alpha),
            torch.full_like(ce_loss, 1.0 - focal_alpha),
        )
        focal_factor = focal_factor * alpha_t

    return (focal_factor * ce_loss).mean()


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    branch_suffix = ""
    if "final_loss" in metrics:
        branch_suffix += f" final={metrics['final_loss']:.4f}"
    semantic_suffix = ""
    if "semantic_loss" in metrics:
        semantic_suffix = f" semantic={metrics['semantic_loss']:.4f}"
        if "semantic_parent_loss" in metrics:
            semantic_suffix += (
                f" [td={metrics['semantic_child_loss']:.4f}"
                f" bu={metrics['semantic_parent_loss']:.4f}]"
            )
        # d = 1 - cos(prediction, target): the criterion-independent semantic evolution
        # discrepancy, comparable across the MSE and cosine arms.
        if "semantic_child_cosine_gap" in metrics:
            semantic_suffix += (
                f" [d_td={metrics['semantic_child_cosine_gap']:.4f}"
                f" d_bu={metrics['semantic_parent_cosine_gap']:.4f}]"
            )
        if "semantic_child_steps" in metrics:
            semantic_suffix += f" childsteps={metrics['semantic_child_steps']:.2f}"
        if "semantic_parent_steps" in metrics:
            semantic_suffix += f" parentsteps={metrics['semantic_parent_steps']:.2f}"
    if "global_semantic_loss" in metrics:
        semantic_suffix += f" global={metrics['global_semantic_loss']:.4f}"
    if "gat_aux_loss" in metrics:
        semantic_suffix += f" gat_aux={metrics['gat_aux_loss']:.4f}"
    if "mask_consistency_kl" in metrics:
        semantic_suffix += (
            f" consKL={metrics['mask_consistency_kl']:.4f}"
            f" mCE={metrics['mask_consistency_masked_ce']:.4f}"
        )
    if "uniformity_loss" in metrics:
        semantic_suffix += f" uniform={metrics['uniformity_loss']:.4f}"
    if "branch_abs_cosine" in metrics:
        semantic_suffix += f" branchcos={metrics['branch_abs_cosine']:.4f}"
    if "source_acc" in metrics:
        semantic_suffix += (
            f" src_acc={metrics['source_acc']:.4f}"
            f" gamma={metrics.get('source_residual_gamma', 0.0):.4f}"
            f" rdelta={metrics.get('correction_ratio', 0.0):.4f}"
        )
    if "correction_net" in metrics:
        semantic_suffix += (
            f" netcorr={int(metrics['correction_net']):+d}"
            f" [wc={int(metrics.get('correction_wc', 0.0))}"
            f" cw={int(metrics.get('correction_cw', 0.0))}]"
        )
    if "claim_attention_entropy" in metrics:
        semantic_suffix += (
            f" claimH={metrics['claim_attention_entropy']:.4f}"
            f" claimRoot={metrics['claim_attention_on_root']:.4f}"
            f" claimNorm={metrics['claim_evidence_norm']:.3f}"
            f" claimFree={metrics.get('claim_reply_free_share', 0.0):.3f}"
        )
    if "contrastive_loss" in metrics:
        semantic_suffix += (
            f" con={metrics['contrastive_loss']:.4f}"
            f" posCos={metrics['contrastive_positive_cosine']:+.4f}"
            f" negCos={metrics['contrastive_negative_cosine']:+.4f}"
            f" gap={metrics['contrastive_gap']:+.4f}"
        )
        if "contrastive_grad_ratio" in metrics:
            semantic_suffix += f" conGradRatio={metrics['contrastive_grad_ratio']:.3f}"
    if "supcon_loss" in metrics:
        semantic_suffix += f" supcon={metrics['supcon_loss']:.4f}"
    if "s_gap" in metrics:
        semantic_suffix += (
            f" swin={metrics['s_within']:.4f}"
            f" sbet={metrics['s_between']:.4f}"
            f" sgap={metrics['s_gap']:+.4f}"
        )
    if "future_loss" in metrics:
        semantic_suffix += f" future={metrics['future_loss']:.4f}"
        if "snapshot_acc" in metrics:
            semantic_suffix += f" early_acc={metrics['snapshot_acc']:.4f}"
    if "sibling_gate" in metrics:
        # Every companion value is optional: D_logit exists only on val/test and
        # R_sib only when the branch was exercised at all.
        semantic_suffix += (
            f" sibgate={metrics['sibling_gate']:.4f}"
            f" sibH={metrics.get('sibling_attention_entropy', float('nan')):.4f}"
            f" sibG={metrics.get('sibling_group_count', 0.0):.1f}"
            f" sibN={metrics.get('sibling_gated_node_share', 0.0):.3f}"
            f" R_sib={metrics.get('sibling_relative_norm', float('nan')):.4f}"
            f" D_logit={metrics.get('sibling_logit_l1', float('nan')):.4f}"
        )
    if "edge_family_real" in metrics:
        semantic_suffix += (
            f" edges={metrics['edge_family_real']:.1f}/"
            f"{metrics['edge_family_2hop']:.1f}/"
            f"{metrics['edge_family_semantic']:.1f}"
        )
    if "masked_input_share" in metrics:
        semantic_suffix += f" mask={metrics['masked_input_share']:.3f}"
    if "temporal_residual_ratio" in metrics:
        # trel: ||gate * z_D|| / ||z_full||. tdrift: how much the propagation states
        # actually differ between consecutive stages. tdecay: mean rho.
        semantic_suffix += (
            f" trel={metrics['temporal_residual_ratio']:.4e}"
            f" tgate={metrics.get('temporal_gate_mean', float('nan')):.4f}"
            f" tdrift={metrics.get('temporal_drift_ratio', float('nan')):.4f}"
            f" tdecay={metrics.get('temporal_decay_mean', float('nan')):.4f}"
        )
    if "temporal_contribution" in metrics:
        # ||W_T z_T|| / ||h_M0||: zero means the temporal branch is still at its
        # identity initialisation and contributes nothing to the logits.
        semantic_suffix += (
            f" tcontrib={metrics['temporal_contribution']:.4e}"
            f" |W_T|={metrics.get('temporal_head_weight_norm', float('nan')):.4e}"
        )
    if "acm_gate_identity" in metrics:
        parts = " ".join(
            f"{name[:4]}={metrics[f'acm_gate_{name}']:.3f}"
            for name in metrics.get("acm_channel_names", [])
            if f"acm_gate_{name}" in metrics
        )
        semantic_suffix += f" acm[{parts}]"
    if "relation_gate_mean" in metrics:
        semantic_suffix += (
            f" relgate={metrics['relation_gate_mean']:.4f}"
            f" relnorm={metrics.get('relation_repr_norm', 0.0):.3f}"
        )
    if "time_gate_mean" in metrics:
        heads = ",".join(f"{value:.4f}" for value in metrics["time_gate_per_head"])
        semantic_suffix += (
            f" timegate={metrics['time_gate_mean']:.4f}"
            f" timeheads=[{heads}]"
            f" timebias={metrics['mean_abs_time_bias_semantic']:.4f}"
        )
    if "gat_grad_norm" in metrics:
        semantic_suffix += f" gatgrad={metrics['gat_grad_norm']:.4f}"
        semantic_suffix += f" gatparam={metrics['gat_param_norm']:.2f}"
        semantic_suffix += f" gatupdate={metrics['gat_update_ratio']:.2e}"
    if "gagru_grad_norm" in metrics:
        semantic_suffix += f" grugrad={metrics['gagru_grad_norm']:.4f}"
        semantic_suffix += f" gruparam={metrics['gagru_param_norm']:.2f}"
        semantic_suffix += f" gruupdate={metrics['gagru_update_ratio']:.2e}"
    if "gat_gagru_grad_ratio" in metrics:
        semantic_suffix += f" gradratio={metrics['gat_gagru_grad_ratio']:.3f}"
    if "inconsistency_contribution" in metrics:
        semantic_suffix += f" incon={metrics['inconsistency_contribution']:.4f}"
        if "semantic_drift" in metrics:
            semantic_suffix += f" drift={metrics['semantic_drift']:.4f}"
    gate_suffix = ""
    if "structure_semantic_gate" in metrics:
        gate_suffix = f" ssgate={metrics['structure_semantic_gate']:.4f}"
    drift_suffix = ""
    if "root_semantic_drift" in metrics:
        drift_suffix = f" rootdrift={metrics['root_semantic_drift']:.4f}"
    return (
        f"{prefix} | "
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
        f"{branch_suffix}"
        f"{semantic_suffix}"
        f"{gate_suffix}"
        f"{drift_suffix}"
    )


def format_binary_class_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    required_keys = (
        "rumor_precision",
        "rumor_recall",
        "rumor_f1",
        "rumor_support",
        "non_rumor_precision",
        "non_rumor_recall",
        "non_rumor_f1",
        "non_rumor_support",
    )
    if any(key not in metrics for key in required_keys):
        return ""
    return (
        f"{prefix} | "
        f"R: prec={metrics['rumor_precision']:.4f} "
        f"recall={metrics['rumor_recall']:.4f} "
        f"f1={metrics['rumor_f1']:.4f} "
        f"support={int(metrics['rumor_support'])} | "
        f"N: prec={metrics['non_rumor_precision']:.4f} "
        f"recall={metrics['non_rumor_recall']:.4f} "
        f"f1={metrics['non_rumor_f1']:.4f} "
        f"support={int(metrics['non_rumor_support'])}"
    )


def metrics_for_logging(metrics: Dict[str, object]) -> Dict[str, object]:
    logged: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logged[key] = float(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float)) for item in value
        ):
            logged[key] = [float(item) for item in value]
    return logged


def append_metrics_event(path: Path, event: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_checkpoint_atomic(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp_path)
    tmp_path.replace(path)


def get_dataset_label(dataset, index: int) -> int:
    if hasattr(dataset, "get_label"):
        return int(dataset.get_label(index))
    # Subset-style wrappers (kfold splits) remap indices onto the base dataset.
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return int(get_dataset_label(dataset.dataset, int(dataset.indices[index])))

    sample_id = dataset.sample_ids[index]
    return int(dataset.labels[sample_id])


def build_class_weights(train_loader, device: torch.device) -> torch.Tensor:
    train_dataset = train_loader.dataset

    if hasattr(train_dataset, "indices") and hasattr(train_dataset, "dataset"):
        base_dataset = train_dataset.dataset
        indices = train_dataset.indices
    else:
        base_dataset = train_dataset
        indices = range(len(train_dataset))

    labels = [get_dataset_label(base_dataset, int(index)) for index in indices]
    counts = torch.zeros(max(labels) + 1 if labels else 0, dtype=torch.float32)
    if counts.numel() == 0:
        return counts.to(device)

    for label in labels:
        counts[label] += 1.0

    weights = torch.sqrt(counts.sum() / (counts.numel() * counts.clamp_min(1.0)))
    return weights.to(device)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """SupCon with cross-event positives only (Khosla et al., 2020).

    Positives are *other samples in the batch* sharing the label, so the loss
    pulls different events of the same class together and cannot be satisfied by
    memorizing a single event. ``features`` must already be L2-normalized.
    Samples without any same-label partner are excluded from the mean.
    """
    if temperature <= 0.0:
        raise ValueError("supcon_temperature must be positive.")
    count = features.size(0)
    if count < 2:
        return features.new_zeros(())
    self_mask = torch.eye(count, dtype=torch.bool, device=features.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    positive_counts = positive_mask.sum(dim=1)
    if not bool(positive_counts.any()):
        return features.new_zeros(())
    similarity = features @ features.t() / float(temperature)
    logits = similarity - similarity.detach().max(dim=1, keepdim=True).values
    exp_logits = logits.exp().masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob_pos[positive_counts > 0].mean()


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Yield batches with a fixed per-class quota.

    Every batch therefore contains ``per_class`` events of each class, so every
    sample has at least ``per_class - 1`` cross-event positive partners for the
    supervised contrastive loss. Pools are reshuffled per epoch (seed + epoch).
    """

    def __init__(self, labels, batch_size: int, num_classes: int, seed: int) -> None:
        self.num_classes = int(num_classes)
        per_class = int(batch_size) // self.num_classes
        if per_class < 1:
            raise ValueError(
                "class-balanced batches need batch_size >= num_classes: "
                f"{batch_size} < {num_classes}."
            )
        self.per_class = per_class
        self.samples_per_batch = per_class * self.num_classes
        self.class_indices: list[list[int]] = [[] for _ in range(self.num_classes)]
        for index, label in enumerate(labels):
            label = int(label)
            if 0 <= label < self.num_classes:
                self.class_indices[label].append(index)
        empty = [c for c, indices in enumerate(self.class_indices) if not indices]
        if empty:
            raise ValueError(
                f"class-balanced batches require samples of every class; empty classes: {empty}."
            )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _shuffled_pool(self, class_index: int, generator: torch.Generator) -> list[int]:
        indices = self.class_indices[class_index]
        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[position] for position in perm]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = [self._shuffled_pool(c, generator) for c in range(self.num_classes)]
        cursors = [0] * self.num_classes
        for _ in range(len(self)):
            batch: list[int] = []
            for c in range(self.num_classes):
                for _ in range(self.per_class):
                    if cursors[c] >= len(pools[c]):
                        pools[c] = self._shuffled_pool(c, generator)
                        cursors[c] = 0
                    batch.append(pools[c][cursors[c]])
                    cursors[c] += 1
            yield batch

    def __len__(self) -> int:
        return min(len(indices) for indices in self.class_indices) // self.per_class


def build_class_balanced_train_loader(train_loader, num_classes: int, seed: int):
    """Replace the shuffled train loader with a class-balanced batch sampler."""
    dataset = train_loader.dataset
    labels = [int(get_dataset_label(dataset, index)) for index in range(len(dataset))]
    sampler = ClassBalancedBatchSampler(labels, train_loader.batch_size, num_classes, seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn,
        pin_memory=train_loader.pin_memory,
    )


def build_partial_graph_train_loader(train_loader, args):
    """Wrap only the training split in epoch-varying partial event views."""
    if args.partial_graph_train_prob <= 0.0:
        return train_loader, None
    dataset = StochasticPartialGraphDataset(
        dataset=train_loader.dataset,
        partial_probability=args.partial_graph_train_prob,
        keep_ratios=args.partial_graph_keep_ratios,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_loader.collate_fn,
        drop_last=train_loader.drop_last,
        pin_memory=train_loader.pin_memory,
    )
    return loader, dataset


def compute_train_graph_feature_statistics(
    train_loader,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute population feature mean/std from nodes in the current training split only."""
    dataset = train_loader.dataset
    feature_sum = torch.zeros(expected_dim, dtype=torch.float64)
    feature_square_sum = torch.zeros(expected_dim, dtype=torch.float64)
    node_count = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        graph = sample.get("graph") if isinstance(sample, dict) else None
        if not isinstance(graph, dict) or "node_features" not in graph:
            raise ValueError("Training samples must contain graph.node_features.")
        features = torch.as_tensor(graph["node_features"], dtype=torch.float64)
        if features.ndim != 2 or features.size(1) != expected_dim:
            raise ValueError(
                "Expected training node features with shape [nodes, "
                f"{expected_dim}], got {tuple(features.shape)} at dataset index {index}."
            )
        if features.size(0) == 0:
            continue
        feature_sum += features.sum(dim=0)
        feature_square_sum += features.square().sum(dim=0)
        node_count += int(features.size(0))

    if node_count == 0:
        raise ValueError("Cannot compute graph feature statistics from an empty training split.")

    mean = feature_sum / float(node_count)
    variance = (feature_square_sum / float(node_count) - mean.square()).clamp_min(0.0)
    return mean.to(dtype=torch.float32), variance.sqrt().to(dtype=torch.float32)


def compute_train_graph_feature_std(train_loader, expected_dim: int) -> torch.Tensor:
    _, feature_std = compute_train_graph_feature_statistics(train_loader, expected_dim)
    return feature_std


@torch.no_grad()
def compute_train_temporal_statistics(
    train_loader,
    model: "DualBranchRumorDetector",
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the M7 temporal feature mean/std on the training split only.

    The temporal features are computed per event from that event's own
    timestamps, so nothing is shared across events. The single fitted quantity is
    this standardisation, and it deliberately comes from the *natural* training
    loader: if a class-balanced sampler rewrites ``train_loader`` later, fitting
    before that keeps the statistics representative of the real split instead of
    the rebalanced one.
    """
    total: Optional[torch.Tensor] = None
    total_square: Optional[torch.Tensor] = None
    event_count = 0
    was_training = model.training
    model.eval()
    try:
        for batch in train_loader:
            features = model.temporal_features_from_batch(
                batch, device=torch.device("cpu")
            )["flat"]
            features = features.detach().to(dtype=torch.float64)
            if total is None:
                total = features.sum(dim=0)
                total_square = features.square().sum(dim=0)
            else:
                total += features.sum(dim=0)
                total_square += features.square().sum(dim=0)
            event_count += int(features.size(0))
    finally:
        model.train(was_training)
    if total is None or event_count == 0:
        raise ValueError("Cannot compute temporal statistics from an empty training split.")
    mean = total / float(event_count)
    variance = (total_square / float(event_count) - mean.square()).clamp_min(0.0)
    return mean.to(dtype=torch.float32), variance.sqrt().to(dtype=torch.float32)


def summarize_feature_std(feature_std: torch.Tensor) -> Dict[str, float]:
    feature_std = feature_std.detach().cpu().float()
    return {
        "minimum": float(feature_std.min().item()),
        "median": float(feature_std.median().item()),
        "mean": float(feature_std.mean().item()),
        "maximum": float(feature_std.max().item()),
    }


def run_gat_pretraining_epoch(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> Dict[str, float]:
    """Pretrain only the shared GAT encoder/decoder with masked reconstruction."""
    if not hasattr(model, "global_semantic"):
        raise RuntimeError("GAT pretraining requires an enabled global_semantic module.")

    model.train()
    reconstruction_setter = getattr(model, "set_global_semantic_reconstruction_enabled", None)
    if callable(reconstruction_setter):
        reconstruction_setter(True)
    total_weighted_loss = 0.0
    total_masked_nodes = 0.0
    for batch in data_loader:
        if "batched_graph" not in batch:
            raise ValueError("GAT pretraining requires collated batched_graph inputs.")
        graph = batch["batched_graph"]
        outputs = model.global_semantic(
            graph["node_features"].to(device),
            graph["edge_index_td"].to(device),
            graph["edge_index_bu"].to(device),
            graph["edge_type_td"].to(device),
            graph["edge_type_bu"].to(device),
            graph["root_indices"].to(device),
            graph_batch=graph["graph_batch"].to(device),
            num_graphs=int(graph["num_graphs"]),
        )
        reconstruction_loss = outputs.get("reconstruction_loss")
        masked_count = outputs.get("masked_count")
        if reconstruction_loss is None or masked_count is None:
            continue

        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.global_semantic.parameters(), grad_clip)
        optimizer.step()

        masked_nodes = float(masked_count.detach().cpu().item())
        total_weighted_loss += float(reconstruction_loss.detach().cpu().item()) * masked_nodes
        total_masked_nodes += masked_nodes

    if total_masked_nodes <= 0.0:
        raise RuntimeError("GAT pretraining did not sample any masked graph nodes.")
    return {
        "global_semantic_loss": total_weighted_loss / total_masked_nodes,
        "masked_nodes": total_masked_nodes,
    }


def scheduled_global_semantic_weight(target_weight: float, epoch: int, warmup_epochs: int) -> float:
    if target_weight <= 0.0 or warmup_epochs <= 0:
        return target_weight
    if warmup_epochs == 1:
        return target_weight
    progress = (float(epoch) - 1.0) / float(warmup_epochs - 1)
    return target_weight * min(max(progress, 0.0), 1.0)


def scheduled_semantic_loss_weight(
    initial_weight: float,
    epoch: int,
    decay_start_epoch: int,
    decay_end_epoch: int,
) -> float:
    if initial_weight <= 0.0:
        return 0.0
    if decay_start_epoch <= 0 and decay_end_epoch <= 0:
        return initial_weight
    if epoch <= decay_start_epoch:
        return initial_weight
    if epoch >= decay_end_epoch:
        return 0.0
    progress = (float(epoch) - float(decay_start_epoch)) / float(
        decay_end_epoch - decay_start_epoch
    )
    return initial_weight * (1.0 - min(max(progress, 0.0), 1.0))


def enable_deterministic_execution() -> None:
    """Force reproducible kernels so two runs of one config agree exactly.

    Measured need: two runs of the identical V1G fold-5 config match bit-for-bit
    through epoch 7 and then diverge (train accuracy 0.8487 vs 0.8485 at epoch 8,
    growing afterwards), which moves the selected checkpoint and the test
    accuracy by ~0.7pp. Any comparison of two arms at the ~1pp level is
    meaningless until this is on.

    Must be called before the first CUDA operation so CUBLAS picks up the
    workspace configuration.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def is_binary_logits(logits: torch.Tensor) -> bool:
    return logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 2)


def select_best_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold_search_space: list[float],
    selection_metric: str,
) -> Dict[str, float]:
    if not is_binary_logits(logits):
        metrics = compute_classification_metrics(logits, labels)
        metrics["threshold"] = None
        return metrics

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(logits, labels, threshold=best_threshold)

    for threshold in threshold_search_space:
        metrics = compute_binary_metrics(logits, labels, threshold=threshold)
        is_better = metrics[selection_metric] > best_metrics[selection_metric]
        same_selected = metrics[selection_metric] == best_metrics[selection_metric]
        better_macro_f1 = metrics["macro_f1"] > best_metrics["macro_f1"]
        if is_better or (same_selected and better_macro_f1):
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_metrics


def resolve_validation_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fixed_threshold: Optional[float],
    disable_threshold_search: bool,
    threshold_search_space: list[float],
    selection_metric: str,
) -> Dict[str, float]:
    if not is_binary_logits(logits):
        metrics = compute_classification_metrics(logits, labels)
        metrics["threshold"] = None
        return metrics

    if fixed_threshold is not None:
        metrics = compute_binary_metrics(logits, labels, threshold=fixed_threshold)
        metrics["threshold"] = fixed_threshold
        return metrics

    if disable_threshold_search:
        metrics = compute_binary_metrics(logits, labels, threshold=0.5)
        metrics["threshold"] = 0.5
        return metrics

    return select_best_threshold(logits, labels, threshold_search_space, selection_metric)


def format_threshold_suffix(metrics: Dict[str, object]) -> str:
    threshold = metrics.get("threshold")
    if isinstance(threshold, (int, float)):
        return f" threshold={float(threshold):.2f}"
    return ""


def is_better_checkpoint(
    current_metrics: Dict[str, float],
    best_metric_value: float,
    best_macro_f1: float,
    checkpoint_metric: str,
) -> bool:
    if checkpoint_metric == "loss":
        current_value = -float(current_metrics["loss"])
    else:
        current_value = float(current_metrics[checkpoint_metric])

    if current_value > best_metric_value:
        return True
    if current_value == best_metric_value and current_metrics["macro_f1"] > best_macro_f1:
        return True
    return False


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if getattr(args, "joint_lora", False):
        graph_parameters = []
        lora_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("joint_text_encoder."):
                lora_parameters.append(parameter)
            else:
                graph_parameters.append(parameter)
        if not lora_parameters:
            raise RuntimeError("Joint LoRA is enabled, but no trainable LoRA parameters were found.")
        parameter_groups = []
        if graph_parameters:
            parameter_groups.append(
                {
                    "params": graph_parameters,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "name": "graph",
                }
            )
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": args.joint_lora_lr,
                "weight_decay": args.joint_lora_weight_decay,
                "name": "lora",
            }
        )
        return torch.optim.AdamW(parameter_groups)
    branch_lrs = {
        "gat": float(args.gat_lr or args.lr),
        "gagru": float(args.gagru_lr or args.lr),
        "fusion": float(args.fusion_lr or args.lr),
        "relation": float(args.lr),
    }
    branch_parameters = {name: [] for name in branch_lrs}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # A wrapped relational model nests the V1G branch under ``content.``;
        # strip that prefix so each parameter keeps its historical bucket.
        if name.startswith("content."):
            name = name[len("content."):]
        if name.startswith("relation_encoder.") or name.startswith("relation_graph.") or name.startswith("relation_projection.") or name.startswith("fusion_gate."):
            branch = "relation"
        elif name.startswith("global_semantic.") or name.startswith("gat_residual_norm."):
            branch = "gat"
        elif name.startswith("semantic_encoder."):
            branch = "gagru"
        else:
            branch = "fusion"
        branch_parameters[branch].append(parameter)
    parameter_groups = [
        {
            "params": branch_parameters[branch],
            "lr": branch_lrs[branch],
            "weight_decay": args.weight_decay,
            "name": branch,
        }
        for branch in ("gat", "gagru", "fusion", "relation")
        if branch_parameters[branch]
    ]
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))


def resolve_joint_lora_adapter_checkpoint(args: argparse.Namespace) -> Optional[str]:
    if not getattr(args, "joint_lora", False):
        return None
    requested = str(getattr(args, "joint_lora_init", "auto") or "none").strip()
    if requested.lower() in {"none", "null", "off"}:
        return None
    if requested.lower() == "auto":
        checkpoint_path = Path(args.dataset_root) / "text_encoder_lora.pt"
    else:
        checkpoint_path = Path(requested).expanduser()
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Joint LoRA adapter checkpoint not found: {checkpoint_path}. "
            "Use --joint-lora-init none to initialize a fresh adapter."
        )
    return str(checkpoint_path)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    """Build an optional LR scheduler. Returns None when --lr-scheduler is 'none'.

    'cosine' anneals the LR from --lr down to --min-lr across --epochs, with an optional
    linear warmup. 'plateau' decays the LR by --lr-plateau-factor whenever the validation
    checkpoint metric stops improving, which pairs naturally with early stopping.
    """
    if args.lr_scheduler == "cosine":
        warmup = max(int(args.warmup_epochs), 0)
        total = max(int(args.epochs), 1)
        min_factor = min(max(args.min_lr / max(args.lr, 1e-12), 0.0), 1.0)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.lr_scheduler == "plateau":
        mode = "min" if args.checkpoint_metric == "loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.min_lr,
        )

    return None


def step_scheduler(scheduler, args: argparse.Namespace, val_metrics: Dict[str, object]) -> None:
    if scheduler is None:
        return
    if args.lr_scheduler == "plateau":
        if args.checkpoint_metric == "loss":
            scheduler.step(float(val_metrics["loss"]))
        else:
            scheduler.step(float(val_metrics[args.checkpoint_metric]))
    else:
        scheduler.step()


def train_and_test_once(
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
    metadata,
    device: torch.device,
    threshold_search_space,
    focal_alpha,
    save_dir: Path,
    selection_split_name: str = "validation",
) -> Dict[str, object]:
    if selection_split_name not in {"validation", "test"}:
        raise ValueError("selection_split_name must be 'validation' or 'test'.")
    if args.gat_relative_time_bias and not metadata.has_time_metadata:
        raise ValueError(
            "--gat-relative-time-bias requires graph files with time_delta_minutes."
        )
    if args.node_representation != "independent" and not getattr(
        metadata, "has_semantic_features", False
    ):
        raise ValueError(
            f"--node-representation {args.node_representation} requires a dual-view "
            "feature root whose graph npz files contain both bert_x and semantic_x."
        )
    legacy_bigcn_loss_selection = (
        args.cv_protocol == "bigcn"
        and selection_split_name == "test"
        and args.checkpoint_metric == "loss"
    )
    joint_lora_enabled = bool(getattr(args, "joint_lora", False))
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    joint_lora_freeze_epochs = int(getattr(args, "joint_lora_freeze_epochs", 0))
    semantic_decay_start_epoch = int(
        getattr(args, "semantic_loss_decay_start_epoch", 0)
    )
    semantic_decay_end_epoch = int(
        getattr(args, "semantic_loss_decay_end_epoch", 0)
    )
    joint_lora_adapter = resolve_joint_lora_adapter_checkpoint(args)
    if joint_lora_enabled:
        args.joint_lora_resolved_init = joint_lora_adapter

    if args.hierarchical_evidence:
        model = HierarchicalEvidenceNetwork(
            input_dim=metadata.graph_input_dim,
            hidden_dim=args.graph_hidden_dim,
            num_heads=args.graph_heads,
            num_slots=args.hierarchical_slots,
            branch_layers=2,
            transformer_layers=2,
            dropout=args.dropout,
            semantic_hidden_dim=args.semantic_hidden_dim or args.graph_hidden_dim,
            num_classes=metadata.num_classes,
        ).to(device)
    else:
        content_model = DualBranchRumorDetector(
            graph_input_dim=metadata.graph_input_dim,
            graph_hidden_dim=args.graph_hidden_dim,
            fusion_dim=args.fusion_dim,
            graph_heads=args.graph_heads,
            dropout=args.dropout,
            input_bottleneck_dim=args.input_bottleneck_dim,
            num_edge_relations=metadata.num_edge_relations,
            use_edge_relations=not args.disable_edge_relations,
            use_semantic_edge_attention=not args.disable_semantic_edge_attention,
            edge_dropout=args.edge_dropout,
            semantic_hidden_dim=args.semantic_hidden_dim or None,
            semantic_projection_dim=args.semantic_proj_dim or None,
            semantic_loss_mse_weight=args.semantic_loss_mse_weight,
            semantic_target=args.semantic_target,
            semantic_prediction_mode=args.semantic_prediction_mode,
            semantic_loss_criterion=args.semantic_loss_criterion,
            semantic_sce_gamma=args.semantic_sce_gamma,
            node_representation=args.node_representation,
            temporal_consistency=args.temporal_consistency,
            gagru_direction=args.gagru_direction,
            gagru_aggregation=args.gagru_aggregation,
            gagru_attention_query=args.gagru_attention_query,
            gagru_attention_score_mode=args.gagru_attention_score_mode,
            gagru_attention_residual_alpha=args.gagru_attention_residual_alpha,
            gagru_attention_residual_mode=args.gagru_attention_residual_mode,
            gagru_attention_gate_init=args.gagru_attention_gate_init,
            gagru_semantic_direction_reduction=args.gagru_semantic_direction_reduction,
            gagru_direction_fusion=args.gagru_direction_fusion,
            gagru_bu_residual_init=args.gagru_bu_residual_init,
            gagru_bu_shared_gradient_scale=args.gagru_bu_shared_gradient_scale,
            gagru_readout=args.gagru_readout,
            structure_encoder=args.structure_encoder,
            gat_input_mode=args.gat_input_mode,
            gat_readout=args.gat_readout,
            gat_attention_type=args.gat_attention_type,
            gat_residual_mode=args.gat_residual_mode,
            gat_residual_alpha=args.gat_residual_alpha,
            gat_neighborhood=args.gat_neighborhood,
            gat_semantic_k=args.gat_semantic_k,
            gat_topology=args.gat_topology,
            gat_global_event_node=args.gat_global_event_node,
            gat_source_evidence_pooling=args.gat_source_evidence_pooling,
            gat_depth_encoding=args.gat_depth_encoding,
            gat_relative_time_bias=args.gat_relative_time_bias,
            acm_channels=args.acm_channels,
            sibling_interaction=args.sibling_interaction,
            sibling_mode=args.sibling_mode,
            sibling_heads=args.sibling_heads,
            sibling_gate_init=args.sibling_gate_init,
            sibling_gate_mode=args.sibling_gate_mode,
            sibling_fusion=args.sibling_fusion,
            sibling_dim=args.sibling_dim,
            source_residual=args.source_residual,
            source_residual_gate_init=args.source_residual_gate_init,
            contrastive_projection_dim=args.supcon_projection_dim,
            set_transformer_num_latents=args.set_transformer_latents,
            set_transformer_num_layers=args.set_transformer_layers,
            global_semantic_mask_ratio=args.global_semantic_mask_ratio,
            global_semantic_sce_gamma=args.global_semantic_sce_gamma,
            global_semantic_target=args.global_semantic_target,
            global_semantic_variance_normalized_mse=not args.disable_global_feature_std,
            global_semantic_feature_std_floor=args.global_semantic_feature_std_floor,
        global_recon_real_edges_only=args.global_recon_real_edges_only,
        claim_guided_attention=args.claim_guided_attention,
        claim_attention_dim=args.claim_attention_dim,
        propagation_adapter=args.propagation_adapter_checkpoint is not None,
        propagation_adapter_bottleneck=args.propagation_adapter_bottleneck,
        propagation_adapter_finetune=args.propagation_adapter_finetune,
        masked_input_ratio=args.masked_input_ratio,
        temporal_dynamics=args.temporal_dynamics,
        temporal_evolution="on" if args.temporal_evolution else "off",
        temporal_evolution_stages=args.temporal_evolution_stages,
        temporal_evolution_hidden=args.temporal_evolution_hidden,
        temporal_evolution_dropout=args.temporal_evolution_dropout,
        temporal_evolution_decay_init=args.temporal_evolution_decay_init,
        temporal_evolution_residual_zero_init=args.temporal_evolution_residual_zero_init,
        temporal_evolution_final_stage_aux_only=args.temporal_evolution_final_stage_aux_only,
        temporal_dim=args.temporal_dim,
        temporal_hidden=args.temporal_hidden,
        temporal_channels=args.temporal_channels,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_dropout=args.temporal_dropout,
        temporal_normalize=args.temporal_normalize,
        temporal_identity_init=args.temporal_identity_init,
        root_feature_dim=args.root_feature_dim,
        root_projection_dim=args.root_projection_dim,
        contrastive_views=args.contrastive_views,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_node_drop_probability=args.contrastive_node_drop_probability,
        contrastive_attribute_mask_probability=args.contrastive_attribute_mask_probability,
            uniformity_temperature=args.uniformity_temperature,
            joint_lora_model_name=(
            getattr(args, "joint_lora_model", "bert-base-uncased")
            if joint_lora_enabled
            else None
            ),
            joint_lora_rank=getattr(args, "joint_lora_rank", 8),
            joint_lora_alpha=getattr(args, "joint_lora_alpha", 16.0),
            joint_lora_dropout=getattr(args, "joint_lora_dropout", 0.05),
            joint_lora_last_layers=getattr(args, "joint_lora_last_layers", 4),
            joint_lora_pooling=getattr(args, "joint_lora_pooling", "mean"),
            joint_lora_scope=getattr(args, "joint_lora_scope", "all"),
            joint_lora_text_batch_size=getattr(args, "joint_lora_text_batch_size", 16),
            joint_lora_local_files_only=not getattr(args, "allow_joint_lora_download", False),
            joint_lora_adapter_checkpoint=joint_lora_adapter,
            joint_lora_gradient_checkpointing=getattr(
            args,
            "joint_lora_gradient_checkpointing",
            False,
            ),
            joint_lora_use_amp=not getattr(args, "disable_joint_lora_amp", False),
            use_gat=args.structure_encoder in {"gat", "acm-gcn"},
            num_classes=metadata.num_classes,
        ).to(device)
        # Applied to the V1G content model, which owns the MS-GAT structure branch; this
        # must run after the model exists (the relation-branch wrapper is built below).
        edge_families_request = str(
            getattr(args, "gat_edge_families", MODEL_HYPERPARAMETERS["gat_edge_families"])
        )
        disabled_families = disabled_edge_families_for(edge_families_request)
        if disabled_families:
            content_model.set_disabled_edge_families(disabled_families)
            print(
                f"Edge families | keeping {edge_families_request!r}, disabled indices "
                f"{disabled_families} (0=real, 1=2-hop, 2=semantic-KNN); real propagation "
                "edges and the global event node are untouched"
            )
        else:
            print("Edge families | keeping all three (real, 2-hop, semantic-KNN)")

        if args.claim_response_relation:
            # The V1G content branch is constructed first and left untouched;
            # the relation branch is added beside it and gate-fused before the
            # content classifier.
            model = RelationalRumorDetector(
                content=content_model,
                input_dim=metadata.graph_input_dim,
                hidden_dim=args.graph_hidden_dim,
                num_heads=args.graph_heads,
                content_dim=2 * args.fusion_dim,
                num_edge_relations=metadata.num_edge_relations,
                dropout=args.dropout,
                gate_bias=args.relation_gate_bias,
            ).to(device)
            print(
                "Claim-response relation branch | "
                f"hidden={args.graph_hidden_dim} heads={args.graph_heads} "
                f"content_dim={2 * args.fusion_dim} gate_bias={args.relation_gate_bias}"
            )
        else:
            model = content_model

    if args.propagation_adapter_checkpoint:
        adapter_path = Path(args.propagation_adapter_checkpoint).expanduser().resolve()
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Propagation adapter checkpoint not found: {adapter_path}")
        payload = torch.load(adapter_path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or "adapter" not in payload:
            raise ValueError(
                "The propagation adapter checkpoint must contain an 'adapter' state dict; "
                f"got keys {sorted(payload)[:5] if isinstance(payload, dict) else type(payload)}."
            )
        expected_dim = payload.get("dim")
        if expected_dim is not None and int(expected_dim) != int(metadata.graph_input_dim):
            raise ValueError(
                "The adapter was pretrained for a different node-feature dimension: "
                f"checkpoint={expected_dim}, dataset={metadata.graph_input_dim}."
            )
        load_report = model.propagation_adapter.load_state_dict(payload["adapter"], strict=True)
        args.propagation_adapter_checkpoint = str(adapter_path)
        print(
            "Propagation semantic adapter | "
            f"loaded from {adapter_path.name} | bottleneck={args.propagation_adapter_bottleneck} "
            f"| finetune={args.propagation_adapter_finetune} | "
            f"stage1 relation loss={payload.get('relation_val_loss_after')}"
        )
        del load_report
    if args.backbone_init_checkpoint:
        shared = Path(args.backbone_init_checkpoint).expanduser().resolve()
        if not shared.is_file():
            raise FileNotFoundError(f"Backbone init checkpoint not found: {shared}")
        reference = torch.load(shared, map_location=device, weights_only=True)
        own = model.state_dict()
        copied, skipped_shape, new_parameters = 0, [], []
        with torch.no_grad():
            for key, value in reference.items():
                if key not in own:
                    continue
                if own[key].shape != value.shape:
                    skipped_shape.append(key)
                    continue
                own[key].copy_(value)
                copied += 1
            new_parameters = [
                key
                for key in own
                if key not in reference
            ]
        args.backbone_init_checkpoint = str(shared)
        print(
            "Paired init | "
            f"copied={copied} shape_mismatch={len(skipped_shape)} "
            f"new_parameters={len(new_parameters)} (from {shared.name})"
        )
        if skipped_shape and not args.paired_init_allow_shape_mismatch:
            raise RuntimeError(
                f"Paired init found shape mismatches in {skipped_shape[:3]}; refusing to "
                "compare arms with partially copied backbones. Pass "
                "--paired-init-allow-shape-mismatch when the change necessarily resizes those "
                "layers (e.g. widening the classifier input for root feature enhancement); the "
                "shared backbone is still identical and only the resized layers start fresh."
            )
        if skipped_shape:
            # Naming exactly which layers start fresh keeps the exception auditable: a widened
            # classifier input is expected here, an unexpected entry elsewhere is not.
            print(
                "Paired init | leaving these layers at their own initialisation "
                f"({len(skipped_shape)}): {sorted(skipped_shape)}"
            )
    if args.save_init_checkpoint:
        init_path = Path(args.save_init_checkpoint).expanduser().resolve()
        init_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), init_path)
        print(f"Saved initialisation to {init_path}")
    if getattr(args, "acm_gate_mode", "learned") != "learned":
        model.global_semantic.encoder.set_acm_gate_mode(args.acm_gate_mode)
        print(
            f"ACM gate ablation | mode={args.acm_gate_mode} "
            f"channels={args.acm_channels}"
        )
    if args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        initial_state = torch.load(init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial_state, strict=True)
        args.init_checkpoint = str(init_checkpoint)
        print(f"Initialized model from {init_checkpoint}")
    if args.freeze_gagru:
        for parameter in model.semantic_encoder.parameters():
            parameter.requires_grad_(False)
        model.semantic_encoder.eval()
        frozen_count = sum(parameter.numel() for parameter in model.semantic_encoder.parameters())
        print(
            "Stage-2 freeze | GA-GRU parameters frozen="
            f"{frozen_count}; representations remain active in fusion"
        )

    if joint_lora_enabled:
        text_encoder = model.joint_text_encoder
        args.joint_lora_model = text_encoder.model_name
        args.joint_lora_rank = text_encoder.rank
        args.joint_lora_alpha = text_encoder.alpha
        args.joint_lora_dropout = text_encoder.lora_dropout
        args.joint_lora_last_layers = text_encoder.last_layers
        print(
            "Joint LoRA text encoder | "
            f"model={text_encoder.model_name} pooling={text_encoder.pooling} "
            f"scope={model.joint_lora_scope} "
            f"modules={len(text_encoder.injected_modules)} "
            f"trainable={text_encoder.trainable_parameter_count / 1e6:.3f}M/"
            f"{text_encoder.total_parameter_count / 1e6:.1f}M "
            f"adapter={text_encoder.adapter_checkpoint or 'fresh'}"
        )
    if grad_accum_steps > 1:
        print(
            "Gradient accumulation | "
            f"physical_batch={args.batch_size} steps={grad_accum_steps} "
            f"effective_batch~={args.batch_size * grad_accum_steps}"
        )

    if not args.disable_gat and (
        args.global_semantic_loss_weight > 0.0 or args.gat_pretrain_epochs > 0
    ):
        global_target_dim = (
            args.graph_hidden_dim
            if args.global_semantic_target == "latent"
            else metadata.graph_input_dim
        )
        global_loss_name = (
            "variance_normalized_mse"
            if args.global_semantic_target == "raw" and not args.disable_global_feature_std
            else "plain_mse"
        )
        print(
            "GAT global target | "
            f"mode={args.global_semantic_target} dim={global_target_dim} "
            f"loss={global_loss_name}"
        )

    feature_std_summary = None
    # The raw-feature mean warm-starts the reconstruction head, which is useful for
    # a plain MSE as well, so it no longer requires the variance normalisation to be
    # on.  Only the std buffer itself stays conditional.
    needs_raw_feature_statistics = (
        not args.disable_gat
        and args.global_semantic_target == "raw"
        and (args.global_semantic_loss_weight > 0.0 or args.gat_pretrain_epochs > 0)
    )
    if needs_raw_feature_statistics:
        print("Computing raw graph feature statistics from the training split...")
        feature_mean, feature_std = compute_train_graph_feature_statistics(
            train_loader,
            metadata.graph_input_dim,
        )
        if not args.disable_global_feature_std:
            model.set_global_semantic_feature_std(feature_std)
        model.initialize_global_semantic_output(feature_mean)
        feature_std_summary = summarize_feature_std(feature_std)
        print(
            "Training feature std | "
            f"min={feature_std_summary['minimum']:.4f} "
            f"median={feature_std_summary['median']:.4f} "
            f"mean={feature_std_summary['mean']:.4f} "
            f"max={feature_std_summary['maximum']:.4f}"
        )

    if args.temporal_dynamics != "off" and not metadata.has_time_metadata:
        raise ValueError(
            "--temporal-dynamics requires graph files with time_delta_minutes "
            "(node_time); this dataset does not provide per-node timestamps."
        )
    if args.temporal_dynamics != "off":
        print(
            f"Fitting temporal feature statistics on the training split only "
            f"(mode={args.temporal_dynamics})..."
        )
        temporal_mean, temporal_std = compute_train_temporal_statistics(
            train_loader, model, device
        )
        model.set_temporal_statistics(temporal_mean, temporal_std)
        temporal_summary = summarize_feature_std(temporal_std)
        print(
            "Temporal feature std | "
            f"min={temporal_summary['minimum']:.4f} "
            f"median={temporal_summary['median']:.4f} "
            f"mean={temporal_summary['mean']:.4f} "
            f"max={temporal_summary['maximum']:.4f} | "
            f"events_scored={train_loader.dataset.__len__()}"
        )

    if args.temporal_consistency and not metadata.has_time_metadata:
        raise ValueError(
            "--temporal-consistency requires graph files with time_delta_minutes "
            "(node_time) so early propagation stages can be cut."
        )
    class_weights = None if args.disable_class_weights else build_class_weights(train_loader, device)
    if args.supcon_class_balanced:
        balanced_batch_size = (
            max(1, args.batch_size // metadata.num_classes) * metadata.num_classes
        )
        train_loader = build_class_balanced_train_loader(
            train_loader,
            metadata.num_classes,
            args.seed,
        )
        print(
            "SupCon class-balanced batches | "
            f"classes={metadata.num_classes} per_class={balanced_batch_size // metadata.num_classes} "
            f"physical_batch={balanced_batch_size}"
        )
    augmented_train_loader, partial_graph_dataset = build_partial_graph_train_loader(
        train_loader,
        args,
    )
    temporal_dataset = None
    if args.temporal_consistency:
        if partial_graph_dataset is not None:
            raise ValueError(
                "--temporal-consistency cannot be combined with --partial-graph-train-prob; "
                "both replace the training view of every event."
            )
        temporal_dataset = TemporalSnapshotDataset(
            dataset=augmented_train_loader.dataset,
            ratios=args.temporal_snapshot_ratios,
            mode=args.temporal_snapshot_mode,
            cutoffs_minutes=args.temporal_cutoffs_minutes,
            seed=args.seed,
        )
        augmented_train_loader = DataLoader(
            temporal_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_temporal_batch,
            drop_last=train_loader.drop_last,
            pin_memory=train_loader.pin_memory,
        )
        stage_text = (
            ", ".join(f"{value:.2f}" for value in args.temporal_snapshot_ratios)
            if args.temporal_snapshot_mode == "ratio"
            else ", ".join(f"{value:g}min" for value in args.temporal_cutoffs_minutes)
        )
        print(
            "Temporal consistency | "
            f"mode={args.temporal_snapshot_mode} stages=[{stage_text}] "
            f"future_weight={args.future_loss_weight} early_ce={args.early_ce_weight}"
        )
    if partial_graph_dataset is not None:
        ratio_text = ", ".join(f"{ratio:.2f}" for ratio in args.partial_graph_keep_ratios)
        print(
            "Partial-event training | "
            f"probability={args.partial_graph_train_prob:.2f} "
            f"keep_ratios=[{ratio_text}] ancestor_preserving=True"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), save_dir / "config.json")
    save_json(
        {
            "graph_input_dim": metadata.graph_input_dim,
            "num_classes": metadata.num_classes,
            "label_to_index": metadata.label_to_index,
            "graph_feature_key": metadata.graph_feature_key,
            "graph_feature_transposed": metadata.graph_feature_transposed,
            "max_graph_nodes": metadata.max_graph_nodes,
            "early_cutoff_minutes": metadata.early_cutoff_minutes,
            "has_time_metadata": metadata.has_time_metadata,
            "has_node_tokens": getattr(metadata, "has_node_tokens", False),
            "node_token_length": getattr(metadata, "node_token_length", None),
            "num_edge_relations": metadata.num_edge_relations,
            "source_split_files": metadata.source_split_files,
            "uses_synthetic_graph": metadata.uses_synthetic_graph,
            "category_feature_size": metadata.category_feature_size,
            "category_to_index": metadata.category_to_index,
            "edge_dropout": args.edge_dropout,
            "input_bottleneck_dim": args.input_bottleneck_dim,
            "partial_graph_train_prob": args.partial_graph_train_prob,
            "partial_graph_keep_ratios": args.partial_graph_keep_ratios,
            "semantic_hidden_dim": args.semantic_hidden_dim,
            "semantic_proj_dim": args.semantic_proj_dim,
            "semantic_loss_mse_weight": args.semantic_loss_mse_weight,
            "semantic_target": args.semantic_target,
            "semantic_prediction_mode": args.semantic_prediction_mode,
            "semantic_loss_criterion": args.semantic_loss_criterion,
            "root_feature_dim": args.root_feature_dim,
            "root_projection_dim": args.root_projection_dim,
            "semantic_sce_gamma": args.semantic_sce_gamma,
            "node_representation": args.node_representation,
            "temporal_consistency": args.temporal_consistency,
            "temporal_snapshot_mode": args.temporal_snapshot_mode,
            "temporal_snapshot_ratios": list(args.temporal_snapshot_ratios)
            if args.temporal_consistency else list(MODEL_HYPERPARAMETERS["temporal_snapshot_ratios"]),
            "temporal_cutoffs_minutes": list(args.temporal_cutoffs_minutes)
            if args.temporal_consistency else list(MODEL_HYPERPARAMETERS["temporal_cutoffs_minutes"]),
            "future_loss_weight": args.future_loss_weight,
            "early_ce_weight": args.early_ce_weight,
            "gagru_direction": args.gagru_direction,
            "gagru_aggregation": args.gagru_aggregation,
            "gagru_attention_query": args.gagru_attention_query,
            "gagru_attention_score_mode": args.gagru_attention_score_mode,
            "gagru_attention_residual_alpha": args.gagru_attention_residual_alpha,
            "gagru_attention_residual_mode": args.gagru_attention_residual_mode,
            "gagru_attention_gate_init": args.gagru_attention_gate_init,
            "gagru_semantic_direction_reduction": args.gagru_semantic_direction_reduction,
            "gagru_direction_fusion": args.gagru_direction_fusion,
            "gagru_bu_residual_init": args.gagru_bu_residual_init,
            "gagru_bu_shared_gradient_scale": args.gagru_bu_shared_gradient_scale,
            "gagru_readout": args.gagru_readout,
            "gat_input_mode": args.gat_input_mode,
            "gat_readout": args.gat_readout,
            "gat_attention_type": args.gat_attention_type,
            "gat_residual_mode": args.gat_residual_mode,
            "gat_residual_alpha": args.gat_residual_alpha,
            "gat_neighborhood": args.gat_neighborhood,
            "gat_semantic_k": args.gat_semantic_k,
            "gat_topology": args.gat_topology,
            "gat_global_event_node": args.gat_global_event_node,
            "gat_source_evidence_pooling": args.gat_source_evidence_pooling,
            "gat_depth_encoding": args.gat_depth_encoding,
            "gat_relative_time_bias": args.gat_relative_time_bias,
            "acm_channels": args.acm_channels,
            "acm_gate_mode": args.acm_gate_mode,
            "sibling_interaction": args.sibling_interaction,
            "sibling_mode": args.sibling_mode,
            "sibling_heads": args.sibling_heads,
            "sibling_gate_init": args.sibling_gate_init,
            "sibling_gate_mode": args.sibling_gate_mode,
            "sibling_fusion": args.sibling_fusion,
            "sibling_dim": args.sibling_dim,
            "backbone_init_checkpoint": args.backbone_init_checkpoint,
            "deterministic": args.deterministic,
            "parent_shuffle": args.parent_shuffle,
            "parent_shuffle_seed": args.parent_shuffle_seed,
            "source_residual": args.source_residual,
            "source_residual_gate_init": args.source_residual_gate_init,
            "supcon_loss_weight": args.supcon_loss_weight,
            "supcon_temperature": args.supcon_temperature,
            "supcon_projection_dim": args.supcon_projection_dim,
            "supcon_class_balanced": args.supcon_class_balanced,
            "hierarchical_evidence": args.hierarchical_evidence,
            "hierarchical_slots": args.hierarchical_slots,
            "claim_response_relation": args.claim_response_relation,
            "relation_gate_bias": args.relation_gate_bias,
            "global_semantic_mask_ratio": args.global_semantic_mask_ratio,
            "global_semantic_sce_gamma": args.global_semantic_sce_gamma,
            "global_semantic_target": args.global_semantic_target,
            "global_semantic_variance_normalized_mse": not args.disable_global_feature_std,
            "global_semantic_feature_std_floor": args.global_semantic_feature_std_floor,
            "global_recon_real_edges_only": args.global_recon_real_edges_only,
            "claim_guided_attention": args.claim_guided_attention,
            "propagation_adapter_checkpoint": args.propagation_adapter_checkpoint,
            "propagation_adapter_bottleneck": args.propagation_adapter_bottleneck,
            "propagation_adapter_finetune": args.propagation_adapter_finetune,
            "contrastive_views": args.contrastive_views,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "contrastive_node_drop_probability": args.contrastive_node_drop_probability,
            "contrastive_attribute_mask_probability": args.contrastive_attribute_mask_probability,
            "claim_attention_dim": args.claim_attention_dim,
            "global_semantic_feature_std_summary": feature_std_summary,
            "semantic_edge_attention": not args.disable_semantic_edge_attention,
            "uniformity_temperature": args.uniformity_temperature,
            "disable_gat": args.disable_gat,
            "structure_encoder": args.structure_encoder,
            "set_transformer_latents": args.set_transformer_latents,
            "set_transformer_layers": args.set_transformer_layers,
            "joint_lora": joint_lora_enabled,
            "joint_lora_model": getattr(args, "joint_lora_model", None),
            "joint_lora_init": joint_lora_adapter,
            "joint_lora_pooling": getattr(args, "joint_lora_pooling", None),
            "joint_lora_scope": getattr(args, "joint_lora_scope", None),
            "joint_lora_rank": getattr(args, "joint_lora_rank", None),
            "joint_lora_alpha": getattr(args, "joint_lora_alpha", None),
            "joint_lora_dropout": getattr(args, "joint_lora_dropout", None),
            "joint_lora_last_layers": getattr(args, "joint_lora_last_layers", None),
            "joint_lora_lr": getattr(args, "joint_lora_lr", None),
            "joint_lora_freeze_epochs": joint_lora_freeze_epochs,
            "joint_lora_text_batch_size": getattr(args, "joint_lora_text_batch_size", None),
            "semantic_loss_weight": args.semantic_loss_weight,
            "semantic_loss_decay_start_epoch": semantic_decay_start_epoch,
            "semantic_loss_decay_end_epoch": semantic_decay_end_epoch,
            "global_semantic_loss_weight": args.global_semantic_loss_weight,
            "gat_pretrain_epochs": args.gat_pretrain_epochs,
            "gat_pretrain_lr": args.gat_pretrain_lr if args.gat_pretrain_lr > 0.0 else args.lr,
            "global_loss_warmup_epochs": args.global_loss_warmup_epochs,
            "gat_aux_loss_weight": args.gat_aux_loss_weight,
            "uniformity_loss_weight": args.uniformity_loss_weight,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": focal_alpha,
            "checkpoint_metric": args.checkpoint_metric,
            "threshold_selection_metric": args.threshold_selection_metric,
            "fixed_threshold": args.fixed_threshold,
            "disable_threshold_search": args.disable_threshold_search,
            "disable_edge_relations": args.disable_edge_relations,
            "cv_protocol": args.cv_protocol,
            "selection_split": selection_split_name,
            "selection_split_requested": getattr(args, "selection_split", "auto"),
            "test_selection_is_leakage": selection_split_name == "test",
            "selection_loss": "classification_final_loss" if args.cv_protocol == "bigcn" else "combined_loss",
            "test_fold_used_for_model_selection": selection_split_name == "test",
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": args.batch_size * grad_accum_steps,
            "train_samples": len(train_loader.dataset),
            "val_samples": (
                len(val_loader.dataset)
                if selection_split_name == "validation"
                else 0
            ),
            "selection_samples": (
                len(val_loader.dataset)
                if selection_split_name in {"validation", "test"}
                else 0
            ),
            "test_samples": len(test_loader.dataset),
            "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
            "disable_class_weights": args.disable_class_weights,
            "threshold_search_space": threshold_search_space,
            "twitter16_model_overrides": TWITTER16_MODEL_OVERRIDES if args.dataset_name == "Twitter16" else {},
            "twitter16_training_overrides": TWITTER16_TRAINING_OVERRIDES if args.dataset_name == "Twitter16" else {},
            "pheme_raw_model_overrides": PHEMERAW_MODEL_OVERRIDES if args.dataset_name == "PhemeRaw" else {},
            "pheme_raw_training_overrides": PHEMERAW_TRAINING_OVERRIDES if args.dataset_name == "PhemeRaw" else {},
            "weibo_model_overrides": WEIBO_MODEL_OVERRIDES if args.dataset_name == "Weibo" else {},
            "weibo_training_overrides": WEIBO_TRAINING_OVERRIDES if args.dataset_name == "Weibo" else {},
            "weibo_runtime_overrides": WEIBO_RUNTIME_OVERRIDES if args.dataset_name == "Weibo" else {},
            "weibo21_model_overrides": WEIBO21_MODEL_OVERRIDES if args.dataset_name == "Weibo21" else {},
            "weibo21_training_overrides": WEIBO21_TRAINING_OVERRIDES if args.dataset_name == "Weibo21" else {},
        },
        save_dir / "dataset_metadata.json",
    )
    history_path = save_dir / "training_history.jsonl"
    latest_metrics_path = save_dir / "latest_metrics.json"
    history_path.write_text("", encoding="utf-8")

    if args.gat_pretrain_epochs > 0:
        pretrain_lr = args.gat_pretrain_lr if args.gat_pretrain_lr > 0.0 else args.lr
        pretrain_optimizer = torch.optim.AdamW(
            model.global_semantic.parameters(),
            lr=pretrain_lr,
            weight_decay=args.weight_decay,
        )
        for pretrain_epoch in range(1, args.gat_pretrain_epochs + 1):
            pretrain_metrics = run_gat_pretraining_epoch(
                model=model,
                data_loader=train_loader,
                device=device,
                optimizer=pretrain_optimizer,
                grad_clip=args.grad_clip,
            )
            print(
                f"GAT pretrain {pretrain_epoch:02d}/{args.gat_pretrain_epochs:02d} | "
                f"global={pretrain_metrics['global_semantic_loss']:.4f} "
                f"masked_nodes={int(pretrain_metrics['masked_nodes'])}"
            )
            append_metrics_event(
                history_path,
                {
                    "phase": "gat_pretrain",
                    "epoch": pretrain_epoch,
                    "lr": pretrain_lr,
                    **pretrain_metrics,
                },
            )
        model.zero_grad(set_to_none=True)
        save_checkpoint_atomic(
            copy.deepcopy(model.global_semantic.state_dict()),
            save_dir / "gat_pretrained_model.pt",
        )

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    frozen_lora_parameters = []
    if joint_lora_enabled and joint_lora_freeze_epochs > 0:
        frozen_lora_parameters = [
            parameter
            for parameter in model.joint_text_encoder.parameters()
            if parameter.requires_grad
        ]
        for parameter in frozen_lora_parameters:
            parameter.requires_grad_(False)
        print(
            "Joint LoRA warmup | "
            f"frozen_epochs={joint_lora_freeze_epochs}; graph modules train first"
        )

    best_state = None
    best_metric_value = -float("inf")
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0
    best_semantic_loss_weight = args.semantic_loss_weight
    patience_left = args.patience

    if args.allow_initial_selection:
        if not args.init_checkpoint:
            raise ValueError("--allow-initial-selection requires --init-checkpoint.")
        if selection_split_name != "validation":
            raise ValueError(
                "--allow-initial-selection requires validation-based checkpoint selection; "
                "using the test fold would leak test information."
            )
        with (ema.average_parameters(model) if ema is not None else nullcontext()):
            initial_outputs = run_epoch(
                model=model,
                data_loader=val_loader,
                device=device,
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                semantic_loss_weight=args.semantic_loss_weight,
                global_semantic_loss_weight=0.0,
                gat_aux_loss_weight=args.gat_aux_loss_weight,
                mask_consistency_loss_weight=args.mask_consistency_loss_weight,
                mask_consistency_ratio=args.mask_consistency_ratio,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                focal_alpha=focal_alpha,
                eval_semantic_reconstruction=args.legacy_eval_semantic_reconstruction,
            )
        initial_metrics = resolve_validation_threshold(
            logits=initial_outputs["logits"],
            labels=initial_outputs["labels"],
            fixed_threshold=args.fixed_threshold,
            disable_threshold_search=args.disable_threshold_search,
            threshold_search_space=threshold_search_space,
            selection_metric=args.threshold_selection_metric,
        )
        initial_metrics["loss"] = (
            initial_outputs["final_loss"]
            if args.cv_protocol == "bigcn"
            else initial_outputs["loss"]
        )
        if args.checkpoint_metric == "loss":
            best_metric_value = -float(initial_metrics["loss"])
        else:
            best_metric_value = float(initial_metrics[args.checkpoint_metric])
        best_val_accuracy = float(initial_metrics["accuracy"])
        best_val_macro_f1 = float(initial_metrics["macro_f1"])
        if isinstance(initial_metrics.get("threshold"), (int, float)):
            best_threshold = float(initial_metrics["threshold"])
        with (ema.average_parameters(model) if ema is not None else nullcontext()):
            best_state = copy.deepcopy(model.state_dict())
        save_checkpoint_atomic(best_state, save_dir / "best_model.pt")
        initial_event = {
            "phase": "initial_candidate",
            "epoch": 0,
            "val": metrics_for_logging(initial_metrics),
            "selection_split": selection_split_name,
            "test_fold_used_for_model_selection": False,
            "best_epoch": 0,
            "best_checkpoint_metric": args.checkpoint_metric,
            "best_checkpoint_metric_value": float(best_metric_value),
            "best_threshold": float(best_threshold),
            "checkpoint_saved": True,
        }
        append_metrics_event(history_path, initial_event)
        save_json(initial_event, latest_metrics_path)
        print("Initial checkpoint candidate")
        print(
            format_metrics("  val  ", initial_metrics)
            + format_threshold_suffix(initial_metrics)
        )

    for epoch in range(1, args.epochs + 1):
        if frozen_lora_parameters and epoch == joint_lora_freeze_epochs + 1:
            for parameter in frozen_lora_parameters:
                parameter.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            print(f"Joint LoRA unfrozen at epoch {epoch:02d}.")
        effective_global_semantic_weight = scheduled_global_semantic_weight(
            args.global_semantic_loss_weight,
            epoch,
            args.global_loss_warmup_epochs,
        )
        effective_semantic_loss_weight = scheduled_semantic_loss_weight(
            args.semantic_loss_weight,
            epoch,
            semantic_decay_start_epoch,
            semantic_decay_end_epoch,
        )
        if partial_graph_dataset is not None:
            partial_graph_dataset.set_epoch(epoch)
        if temporal_dataset is not None:
            temporal_dataset.set_epoch(epoch)
        balanced_sampler = getattr(augmented_train_loader, "batch_sampler", None)
        if isinstance(balanced_sampler, ClassBalancedBatchSampler):
            balanced_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            data_loader=augmented_train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            class_weights=class_weights,
            threshold=best_threshold,
            label_smoothing=args.label_smoothing,
            semantic_loss_weight=effective_semantic_loss_weight,
            global_semantic_loss_weight=effective_global_semantic_weight,
            gat_aux_loss_weight=args.gat_aux_loss_weight,
            mask_consistency_loss_weight=args.mask_consistency_loss_weight,
            mask_consistency_ratio=args.mask_consistency_ratio,
            uniformity_loss_weight=args.uniformity_loss_weight,
            supcon_loss_weight=args.supcon_loss_weight,
            supcon_temperature=args.supcon_temperature,
            contrastive_loss_weight=args.contrastive_loss_weight,
            measure_contrastive_grad_ratio=args.contrastive_views,
            future_loss_weight=args.future_loss_weight,
            early_ce_weight=args.early_ce_weight,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=focal_alpha,
            ema=ema,
            gradient_accumulation_steps=grad_accum_steps,
        )
        # Evaluate (and later checkpoint) with the EMA-averaged weights when enabled.
        with (ema.average_parameters(model) if ema is not None else nullcontext()):
            val_outputs = run_epoch(
                model=model,
                data_loader=val_loader,
                device=device,
                class_weights=class_weights,
                measure_sibling_delta=args.sibling_interaction,
                contrastive_loss_weight=0.0,
                label_smoothing=args.label_smoothing,
                semantic_loss_weight=effective_semantic_loss_weight,
                global_semantic_loss_weight=0.0,
                gat_aux_loss_weight=args.gat_aux_loss_weight,
                mask_consistency_loss_weight=args.mask_consistency_loss_weight,
                mask_consistency_ratio=args.mask_consistency_ratio,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                focal_alpha=focal_alpha,
                eval_semantic_reconstruction=args.legacy_eval_semantic_reconstruction,
            )
        val_metrics = resolve_validation_threshold(
            logits=val_outputs["logits"],
            labels=val_outputs["labels"],
            fixed_threshold=args.fixed_threshold,
            disable_threshold_search=args.disable_threshold_search,
            threshold_search_space=threshold_search_space,
            selection_metric=args.threshold_selection_metric,
        )
        if args.cv_protocol == "bigcn":
            # BiGCN checkpoints on classification NLL only. Keep the combined
            # objective available for diagnostics, but do not let auxiliary
            # semantic losses alter model selection.
            val_metrics["loss"] = val_outputs["final_loss"]
            val_metrics["combined_evaluation_loss"] = (
                val_outputs["loss"]
                if args.legacy_eval_semantic_reconstruction
                else val_outputs["final_loss"]
            )
        else:
            val_metrics["loss"] = val_outputs["loss"]
        for key in (
            "final_loss",
            "semantic_loss",
            "gat_aux_loss",
            "mask_consistency_kl",
            "mask_consistency_masked_ce",
            "mask_consistency_kl",
            "mask_consistency_masked_ce",
            "semantic_child_steps",
            "semantic_parent_steps",
            "semantic_child_loss",
            "semantic_parent_loss",
            "inconsistency_contribution",
            "semantic_drift",
            "structure_semantic_gate",
            "root_semantic_drift",
            "branch_abs_cosine",
            "source_acc",
            "source_residual_gamma",
            "correction_ratio",
            "source_logit_norm",
            "correction_logit_norm",
            "correction_cc",
            "correction_wc",
            "correction_cw",
            "correction_ww",
            "correction_net",
            "supcon_loss",
            "s_within",
            "s_between",
            "s_gap",
            "relation_gate_mean",
            "relation_repr_norm",
            "future_loss",
            "early_ce_loss",
            "snapshot_acc",
        ):
            if key in val_outputs:
                val_metrics[key] = val_outputs[key]
        # ACM channel gates are named by the configured channel set, so they are
        # copied by prefix rather than enumerated.
        for key in (
            "acm_channel_names",
            "acm_gate_per_layer",
            "acm_zero_channel_gate_mass",
            "sibling_gate",
            "sibling_attention_entropy",
            "sibling_group_count",
            "sibling_gated_node_share",
            "sibling_relative_norm",
            "sibling_logit_l1",
            "temporal_contribution",
            "temporal_head_weight_norm",
            "masked_input_share",
            "edge_family_real",
            "edge_family_2hop",
            "edge_family_semantic",
            "temporal_residual_ratio",
            "temporal_gate_mean",
            "temporal_drift_ratio",
            "temporal_decay_mean",
            *[k for k in val_outputs if k.startswith("acm_gate_")],
        ):
            if key in val_outputs:
                val_metrics[key] = val_outputs[key]
        if isinstance(val_outputs.get("acm_gate_per_layer"), list):
            val_metrics["acm_gate_per_layer"] = val_outputs["acm_gate_per_layer"]

        current_lr = float(optimizer.param_groups[0]["lr"])
        step_scheduler(scheduler, args, val_metrics)

        print(f"Epoch {epoch:02d}")
        global_weight_suffix = ""
        if args.global_semantic_loss_weight > 0.0 and args.global_loss_warmup_epochs > 0:
            global_weight_suffix = f" global_weight={effective_global_semantic_weight:.4f}"
        semantic_weight_suffix = ""
        if semantic_decay_end_epoch > 0:
            semantic_weight_suffix = f" semantic_weight={effective_semantic_loss_weight:.4f}"
        print(
            format_metrics("  train", train_metrics)
            + semantic_weight_suffix
            + global_weight_suffix
        )
        train_class_metrics = format_binary_class_metrics("        class", train_metrics)
        if train_class_metrics:
            print(train_class_metrics)
        lr_suffix = f" lr={current_lr:.2e}" if scheduler is not None else ""
        selection_label = "  test*" if selection_split_name == "test" else "  val  "
        print(format_metrics(selection_label, val_metrics) + format_threshold_suffix(val_metrics) + lr_suffix)
        val_class_metrics = format_binary_class_metrics("        class", val_metrics)
        if val_class_metrics:
            print(val_class_metrics)

        if legacy_bigcn_loss_selection:
            # BiGCN treats equal test-fold loss as an improvement and resets
            # patience, because only strictly worse loss increments its counter.
            better_checkpoint = -float(val_metrics["loss"]) >= best_metric_value
        elif args.allow_initial_selection:
            # A protected epoch-0 checkpoint is replaced only by a strict
            # improvement in the requested metric. In particular, an equal
            # validation accuracy must not discard a known stronger fallback.
            current_selection_value = (
                -float(val_metrics["loss"])
                if args.checkpoint_metric == "loss"
                else float(val_metrics[args.checkpoint_metric])
            )
            better_checkpoint = current_selection_value > best_metric_value
        else:
            better_checkpoint = is_better_checkpoint(
                current_metrics=val_metrics,
                best_metric_value=best_metric_value,
                best_macro_f1=best_val_macro_f1,
                checkpoint_metric=args.checkpoint_metric,
            )
        checkpoint_saved = False

        if better_checkpoint:
            if args.checkpoint_metric == "loss":
                best_metric_value = -float(val_metrics["loss"])
            else:
                best_metric_value = float(val_metrics[args.checkpoint_metric])
            best_val_accuracy = val_metrics["accuracy"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_semantic_loss_weight = effective_semantic_loss_weight
            if isinstance(val_metrics.get("threshold"), (int, float)):
                best_threshold = float(val_metrics["threshold"])
            patience_left = args.patience
            # Checkpoint the EMA-averaged weights (the ones we evaluated) when enabled.
            with (ema.average_parameters(model) if ema is not None else nullcontext()):
                best_state = copy.deepcopy(model.state_dict())
            save_checkpoint_atomic(best_state, save_dir / "best_model.pt")
            checkpoint_saved = True
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")

        epoch_event = {
            "phase": "epoch",
            "epoch": epoch,
            "train": metrics_for_logging(train_metrics),
            "val": metrics_for_logging(val_metrics),
            "selection_split": selection_split_name,
            "test_fold_used_for_model_selection": selection_split_name == "test",
            "best_val_accuracy": float(best_val_accuracy),
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_epoch": int(best_epoch),
            "best_checkpoint_metric": args.checkpoint_metric,
            "best_checkpoint_metric_value": float(best_metric_value),
            "best_threshold": float(best_threshold),
            "patience_left": int(patience_left),
            "checkpoint_saved": checkpoint_saved,
            "lr": current_lr,
            "effective_semantic_loss_weight": effective_semantic_loss_weight,
            "effective_global_semantic_loss_weight": effective_global_semantic_weight,
        }
        append_metrics_event(history_path, epoch_event)
        save_json(epoch_event, latest_metrics_path)

        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    test_outputs = run_epoch(
        model=model,
        data_loader=test_loader,
        device=device,
        class_weights=class_weights,
        measure_sibling_delta=args.sibling_interaction,
        contrastive_loss_weight=0.0,
        threshold=best_threshold,
        label_smoothing=args.label_smoothing,
        semantic_loss_weight=best_semantic_loss_weight,
        global_semantic_loss_weight=0.0,
        gat_aux_loss_weight=args.gat_aux_loss_weight,
        mask_consistency_loss_weight=args.mask_consistency_loss_weight,
        mask_consistency_ratio=args.mask_consistency_ratio,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
        eval_semantic_reconstruction=args.legacy_eval_semantic_reconstruction,
    )
    test_metrics = compute_classification_metrics(
        test_outputs["logits"],
        test_outputs["labels"],
        threshold=best_threshold,
    )
    if args.cv_protocol == "bigcn":
        test_metrics["loss"] = test_outputs["final_loss"]
        test_metrics["combined_evaluation_loss"] = (
            test_outputs["loss"]
            if args.legacy_eval_semantic_reconstruction
            else test_outputs["final_loss"]
        )
    else:
        test_metrics["loss"] = test_outputs["loss"]
    for key in (
        "final_loss",
        "semantic_loss",
        "global_semantic_loss",
        "gat_aux_loss",
        "semantic_child_steps",
        "semantic_parent_steps",
        "semantic_child_loss",
        "semantic_parent_loss",
        "inconsistency_contribution",
        "semantic_drift",
        "structure_semantic_gate",
        "root_semantic_drift",
        "branch_abs_cosine",
        "source_acc",
        "source_residual_gamma",
        "correction_ratio",
        "source_logit_norm",
        "correction_logit_norm",
        "correction_cc",
        "correction_wc",
        "correction_cw",
        "correction_ww",
        "correction_net",
        "supcon_loss",
        "s_within",
        "s_between",
        "s_gap",
        "relation_gate_mean",
        "relation_repr_norm",
        "future_loss",
        "early_ce_loss",
        "snapshot_acc",
    ):
        if key in test_outputs:
            test_metrics[key] = test_outputs[key]
    for key in (
        "acm_channel_names",
        "acm_gate_per_layer",
        "acm_zero_channel_gate_mass",
        "sibling_gate",
        "sibling_attention_entropy",
        "sibling_group_count",
        "sibling_gated_node_share",
        "sibling_relative_norm",
        "sibling_logit_l1",
        "temporal_contribution",
        "temporal_head_weight_norm",
        "masked_input_share",
        "edge_family_real",
        "edge_family_2hop",
        "edge_family_semantic",
        "temporal_residual_ratio",
        "temporal_gate_mean",
        "temporal_drift_ratio",
        "temporal_decay_mean",
        *[k for k in test_outputs if k.startswith("acm_gate_")],
    ):
        if key in test_outputs:
            test_metrics[key] = test_outputs[key]
    if isinstance(test_outputs.get("acm_gate_per_layer"), list):
        test_metrics["acm_gate_per_layer"] = test_outputs["acm_gate_per_layer"]
    if is_binary_logits(test_outputs["logits"]):
        test_metrics["threshold"] = best_threshold
    else:
        test_metrics["threshold"] = None
    test_metrics["best_epoch"] = best_epoch
    test_metrics["effective_semantic_loss_weight"] = best_semantic_loss_weight
    if getattr(args, "sibling_interaction", False):
        # Zero-Sibling control: the same checkpoint with gamma_sib = 0, i.e. the
        # plain V1G forward.  Full vs Zero on identical weights is the direct
        # test of whether the sibling branch is actually used at inference.
        model.set_sibling_gate(0.0)
        zero_outputs = run_epoch(
            model=model,
            data_loader=test_loader,
            device=device,
            class_weights=class_weights,
            threshold=best_threshold,
            label_smoothing=args.label_smoothing,
            semantic_loss_weight=0.0,
            global_semantic_loss_weight=0.0,
            gat_aux_loss_weight=args.gat_aux_loss_weight,
            mask_consistency_loss_weight=args.mask_consistency_loss_weight,
            mask_consistency_ratio=args.mask_consistency_ratio,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=focal_alpha,
        )
        zero_metrics = compute_classification_metrics(
            zero_outputs["logits"], zero_outputs["labels"], threshold=best_threshold
        )
        test_metrics["zero_sibling_accuracy"] = zero_metrics["accuracy"]
        test_metrics["zero_sibling_macro_f1"] = zero_metrics["macro_f1"]
        # Rumor-class F1 uses whichever key this label space provides.
        for source, target in (
            ("rumor_f1", "zero_sibling_rumor_f1"),
            ("class_1_f1", "zero_sibling_rumor_f1"),
        ):
            if source in zero_metrics:
                test_metrics[target] = zero_metrics[source]
                break
        # Identical weights, gate on vs off: the net effect of the sibling branch.
        test_metrics["sibling_gain_pp"] = (
            test_metrics["accuracy"] - zero_metrics["accuracy"]
        ) * 100.0
        test_metrics["sibling_macro_f1_gain_pp"] = (
            test_metrics["macro_f1"] - zero_metrics["macro_f1"]
        ) * 100.0
        print(
            f"test ( gamma_sib = 0 ) | "
            f"acc={zero_metrics['accuracy']:.4f} macro_f1={zero_metrics['macro_f1']:.4f} | "
            f"sibling_gain={test_metrics['sibling_gain_pp']:+.2f}pp"
        )
        model.set_sibling_gate(None)
    print(format_metrics("test ", test_metrics) + format_threshold_suffix(test_metrics))
    test_class_metrics = format_binary_class_metrics("      class", test_metrics)
    if test_class_metrics:
        print(test_class_metrics)
    save_json(test_metrics, save_dir / "test_metrics.json")
    test_event = {
        "phase": "test",
        "test": metrics_for_logging(test_metrics),
        "best_threshold": float(best_threshold),
        "cv_protocol": args.cv_protocol,
        "test_fold_used_for_model_selection": selection_split_name == "test",
    }
    append_metrics_event(history_path, test_event)
    save_json(test_event, latest_metrics_path)
    return test_metrics


def run_cross_validation(
    args: argparse.Namespace,
    device: torch.device,
    threshold_search_space,
    focal_alpha,
) -> None:
    inner_validation = uses_bigcn_inner_validation(args)
    checkpoint_metric = getattr(args, "checkpoint_metric", "loss")
    test_fold_selection, selection_fallback = resolve_selection_split(args, inner_validation)
    if selection_fallback:
        print(
            "Selection | --selection-split validation requested, but the BiGCN protocol "
            "was run without an inner validation split; falling back to test selection. "
            "Pass --bigcn-inner-val-ratio > 0 or use --split-ratios."
        )
    elif test_fold_selection and getattr(args, "selection_split", "auto") == "test" and inner_validation:
        print(
            "Selection | WARNING --selection-split test ignores the inner validation "
            "split: checkpoints, patience and the reported number all come from the "
            "test fold (the SAMGAT / GAT_pheme.py protocol). Those metrics are "
            "optimistically biased and are not comparable with validation-selected runs."
        )
    if inner_validation:
        protocol_suffix = "_bigcn_inner_val"
    elif args.cv_protocol == "bigcn":
        protocol_suffix = "_bigcn"
    else:
        protocol_suffix = ""
    base_dir = Path(args.save_dir) / args.dataset_name / f"cv{args.cv_folds}{protocol_suffix}"
    base_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ("accuracy", "precision", "recall", "f1", "macro_f1")
    class_metric_keys = (
        "rumor_precision",
        "rumor_recall",
        "rumor_f1",
        "non_rumor_precision",
        "non_rumor_recall",
        "non_rumor_f1",
    )
    support_keys = ("rumor_support", "non_rumor_support")
    confusion_keys = ("true_positive", "true_negative", "false_positive", "false_negative")
    fold_record_keys = metric_keys + class_metric_keys + support_keys + confusion_keys
    summary_metric_keys = metric_keys + class_metric_keys + support_keys
    fold_results = []
    fold_dataset_roots: Dict[str, str] = {}
    requested_fold = int(getattr(args, "cv_fold_index", 0) or 0)
    root_template = getattr(args, "cv_fold_dataset_root_template", None)
    fold_indices = [requested_fold - 1] if requested_fold > 0 else list(range(args.cv_folds))

    shared_fold_loaders = None
    shared_metadata = None
    if not root_template:
        shared_fold_loaders, shared_metadata = build_kfold_dataloaders(
            dataset_root=args.dataset_root,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            n_folds=args.cv_folds,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
            num_workers=args.num_workers,
            max_graph_nodes=args.max_graph_nodes,
            early_cutoff_minutes=args.early_cutoff_minutes,
            cv_protocol=args.cv_protocol,
            node_token_root=(
                getattr(args, "node_token_root", None)
                if getattr(args, "joint_lora", False)
                else None
            ),
        )
    if args.cv_protocol == "bigcn":
        if inner_validation:
            print(
                    "BiGCN outer CV with internal validation: class-wise floor(20%) test folds; "
                    f"{args.bigcn_inner_val_ratio:.1%} of each outer training pool is reserved for "
                    f"validation and {checkpoint_metric} selects checkpoints."
            )
        else:
            print(
                "BiGCN-compatible CV: class-wise floor(20%) test folds, no independent validation split; "
                f"test* {checkpoint_metric} is used for checkpointing and early stopping."
            )

    for fold_index in fold_indices:
        fold_number = fold_index + 1
        fold_args = args
        if root_template:
            fold_dataset_root = root_template.format(fold=fold_number)
            all_loaders, metadata = build_kfold_dataloaders(
                dataset_root=fold_dataset_root,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                n_folds=args.cv_folds,
                val_ratio=args.val_ratio,
                seed=args.split_seed,
                num_workers=args.num_workers,
                max_graph_nodes=args.max_graph_nodes,
                early_cutoff_minutes=args.early_cutoff_minutes,
                cv_protocol=args.cv_protocol,
                node_token_root=(
                    getattr(args, "node_token_root", None)
                    if getattr(args, "joint_lora", False)
                    else None
                ),
            )
            train_loader, val_loader, test_loader = all_loaders[fold_index]
            fold_args = copy.copy(args)
            fold_args.dataset_root = fold_dataset_root
            fold_dataset_roots[str(fold_number)] = str(Path(fold_dataset_root).resolve())

            manifest_path = Path(fold_dataset_root) / "fold_manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                expected = {
                    "fold": fold_number,
                    "cv_folds": args.cv_folds,
                    "cv_protocol": args.cv_protocol,
                    "split_seed": args.split_seed,
                    "train_count": len(train_loader.dataset)
                    + (len(val_loader.dataset) if inner_validation else 0),
                    "test_count": len(test_loader.dataset),
                }
                mismatches = {
                    key: (manifest.get(key), value)
                    for key, value in expected.items()
                    if manifest.get(key) != value
                }
                if manifest.get("train_test_overlap"):
                    mismatches["train_test_overlap"] = (
                        manifest.get("train_test_overlap"),
                        [],
                    )
                if mismatches:
                    raise RuntimeError(
                        f"Fold-specific feature manifest mismatch at {manifest_path}: {mismatches}"
                    )
        else:
            if shared_fold_loaders is None or shared_metadata is None:
                raise RuntimeError("Shared cross-validation loaders were not initialized.")
            train_loader, val_loader, test_loader = shared_fold_loaders[fold_index]
            metadata = shared_metadata
            fold_dataset_roots[str(fold_number)] = str(Path(args.dataset_root).resolve())

        print(f"\n===== Cross-validation fold {fold_number}/{args.cv_folds} =====")
        if root_template:
            print(f"features | dataset_root={fold_args.dataset_root}")
        if getattr(args, "parent_shuffle", False):
            train_loader = wrap_loader_with_parent_shuffle(
                train_loader, seed=args.parent_shuffle_seed
            )
            val_loader = wrap_loader_with_parent_shuffle(
                val_loader, seed=args.parent_shuffle_seed
            )
            test_loader = wrap_loader_with_parent_shuffle(
                test_loader, seed=args.parent_shuffle_seed
            )
            print(
                "Parent shuffle | depths>=2 rewired for train/val/test; level sizes, "
                "per-node depths and out-degrees preserved"
            )
        selection_loader = test_loader if test_fold_selection else val_loader
        selection_split_name = "test" if test_fold_selection else "validation"
        print(
            f"split | train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
            f"test={len(test_loader.dataset)} selection={selection_split_name}"
        )
        set_seed(args.seed + fold_index)
        fold_dir = base_dir / f"fold{fold_number}"
        test_metrics = train_and_test_once(
            args=fold_args,
            train_loader=train_loader,
            val_loader=selection_loader,
            test_loader=test_loader,
            metadata=metadata,
            device=device,
            threshold_search_space=threshold_search_space,
            focal_alpha=focal_alpha,
            save_dir=fold_dir,
            selection_split_name=selection_split_name,
        )
        fold_summary = {
            key: float(test_metrics[key])
            for key in fold_record_keys
            if key in test_metrics
        }
        fold_summary["fold"] = fold_number
        fold_summary["train_samples"] = len(train_loader.dataset)
        fold_summary["val_samples"] = len(val_loader.dataset)
        fold_summary["test_samples"] = len(test_loader.dataset)
        fold_results.append(fold_summary)
        print(
            f"[fold {fold_number}/{args.cv_folds}] "
            + " ".join(f"{key}={fold_summary[key]:.4f}" for key in metric_keys if key in fold_summary)
        )
        fold_class_metrics = format_binary_class_metrics(
            f"[fold {fold_number}/{args.cv_folds} class]",
            fold_summary,
        )
        if fold_class_metrics:
            print(fold_class_metrics)

    summary: Dict[str, object] = {
        "n_folds": args.cv_folds,
        "evaluated_folds": [result["fold"] for result in fold_results],
        "cv_protocol": args.cv_protocol,
        "split_seed": args.split_seed,
        "class_labels": {"rumor": 1, "non_rumor": 0},
        "fold_dataset_root_template": root_template,
        "fold_dataset_roots": fold_dataset_roots,
        "bigcn_inner_val_ratio": float(getattr(args, "bigcn_inner_val_ratio", 0.0) or 0.0),
        "test_fold_used_for_model_selection": test_fold_selection,
        "semantic_loss_weight": float(getattr(args, "semantic_loss_weight", 0.0)),
        "gagru_direction": str(getattr(args, "gagru_direction", "bidirectional")),
        "gagru_aggregation": str(getattr(args, "gagru_aggregation", "attention")),
        "gagru_attention_query": str(
            getattr(args, "gagru_attention_query", "global")
        ),
        "gagru_attention_score_mode": str(
            getattr(args, "gagru_attention_score_mode", "dot")
        ),
        "gagru_attention_residual_alpha": float(
            getattr(args, "gagru_attention_residual_alpha", 1.0)
        ),
        "gagru_attention_residual_mode": str(
            getattr(args, "gagru_attention_residual_mode", "fixed")
        ),
        "gagru_attention_gate_init": float(
            getattr(args, "gagru_attention_gate_init", 0.1)
        ),
        "gagru_semantic_direction_reduction": str(
            getattr(args, "gagru_semantic_direction_reduction", "sum")
        ),
        "gagru_direction_fusion": str(
            getattr(args, "gagru_direction_fusion", "joint")
        ),
        "gagru_bu_residual_init": float(
            getattr(args, "gagru_bu_residual_init", 0.1)
        ),
        "gagru_bu_shared_gradient_scale": float(
            getattr(args, "gagru_bu_shared_gradient_scale", 1.0)
        ),
        "gagru_readout": str(getattr(args, "gagru_readout", "mean")),
        "gat_input_mode": str(getattr(args, "gat_input_mode", "raw")),
        "global_recon_real_edges_only": bool(
            getattr(args, "global_recon_real_edges_only", False)
        ),
        "claim_guided_attention": bool(getattr(args, "claim_guided_attention", False)),
        "contrastive_views": bool(getattr(args, "contrastive_views", False)),
        "propagation_adapter_checkpoint": str(
            getattr(args, "propagation_adapter_checkpoint", None)
        ),
        "propagation_adapter_finetune": bool(
            getattr(args, "propagation_adapter_finetune", False)
        ),
        "contrastive_temperature": float(getattr(args, "contrastive_temperature", 0.2)),
        "contrastive_loss_weight": float(getattr(args, "contrastive_loss_weight", 0.05)),
        "gat_readout": str(getattr(args, "gat_readout", "mean")),
        "gat_attention_type": str(getattr(args, "gat_attention_type", "gat")),
        "gat_residual_mode": str(getattr(args, "gat_residual_mode", "fixed")),
        "gat_residual_alpha": float(getattr(args, "gat_residual_alpha", 1.0)),
        "gat_neighborhood": str(getattr(args, "gat_neighborhood", "local")),
        "gat_semantic_k": int(getattr(args, "gat_semantic_k", 3)),
        "gat_topology": str(getattr(args, "gat_topology", "tree")),
        "gat_global_event_node": bool(getattr(args, "gat_global_event_node", False)),
        "gat_source_evidence_pooling": bool(
            getattr(args, "gat_source_evidence_pooling", False)
        ),
        "gat_depth_encoding": bool(getattr(args, "gat_depth_encoding", False)),
        "gat_relative_time_bias": bool(getattr(args, "gat_relative_time_bias", False)),
        "acm_channels": str(getattr(args, "acm_channels", "")),
        "acm_gate_mode": str(getattr(args, "acm_gate_mode", "learned")),
        "sibling_interaction": bool(getattr(args, "sibling_interaction", False)),
        "sibling_mode": str(getattr(args, "sibling_mode", "true")),
        "sibling_gate_mode": str(getattr(args, "sibling_gate_mode", "learned")),
        "sibling_fusion": str(getattr(args, "sibling_fusion", "node-residual")),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "parent_shuffle": bool(getattr(args, "parent_shuffle", False)),
        "source_residual": bool(getattr(args, "source_residual", False)),
        "source_residual_gate_init": float(
            getattr(args, "source_residual_gate_init", 0.2)
        ),
        "supcon_loss_weight": float(getattr(args, "supcon_loss_weight", 0.0)),
        "supcon_temperature": float(getattr(args, "supcon_temperature", 0.1)),
        "supcon_projection_dim": int(getattr(args, "supcon_projection_dim", 0)),
        "node_representation": str(getattr(args, "node_representation", "independent")),
        "temporal_consistency": bool(getattr(args, "temporal_consistency", False)),
        "temporal_snapshot_mode": str(getattr(args, "temporal_snapshot_mode", "ratio")),
        "future_loss_weight": float(getattr(args, "future_loss_weight", 0.05)),
        "early_ce_weight": float(getattr(args, "early_ce_weight", 0.0)),
        "supcon_class_balanced": bool(getattr(args, "supcon_class_balanced", False)),
        "hierarchical_evidence": bool(getattr(args, "hierarchical_evidence", False)),
        "hierarchical_slots": int(getattr(args, "hierarchical_slots", 4)),
        "claim_response_relation": bool(
            getattr(args, "claim_response_relation", False)
        ),
        "relation_gate_bias": float(getattr(args, "relation_gate_bias", 3.0)),
        "semantic_loss_decay_start_epoch": int(
            getattr(args, "semantic_loss_decay_start_epoch", 0)
        ),
        "semantic_loss_decay_end_epoch": int(
            getattr(args, "semantic_loss_decay_end_epoch", 0)
        ),
        "folds": fold_results,
        "mean": {},
        "std": {},
    }
    print(
        "\n===== Cross-validation summary "
        f"(mean +/- std over {len(fold_results)} evaluated fold(s)) ====="
    )
    for key in summary_metric_keys:
        values = [result[key] for result in fold_results if key in result]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance ** 0.5
        summary["mean"][key] = mean_value
        summary["std"][key] = std_value
        if key in metric_keys:
            print(f"  {key:10s} = {mean_value:.4f} +/- {std_value:.4f}")

    if all(key in summary["mean"] for key in class_metric_keys):
        print("\n  Class-wise metrics (mean +/- std)")
        for label, prefix in (("Rumor (R)", "rumor"), ("Non-rumor (N)", "non_rumor")):
            support = summary["mean"].get(f"{prefix}_support", 0.0)
            print(
                f"  {label:13s} | "
                f"precision={summary['mean'][f'{prefix}_precision']:.4f} "
                f"+/- {summary['std'][f'{prefix}_precision']:.4f} "
                f"recall={summary['mean'][f'{prefix}_recall']:.4f} "
                f"+/- {summary['std'][f'{prefix}_recall']:.4f} "
                f"f1={summary['mean'][f'{prefix}_f1']:.4f} "
                f"+/- {summary['std'][f'{prefix}_f1']:.4f} "
                f"support/fold={support:.1f}"
            )

    summary_name = f"cv_summary_fold{requested_fold}.json" if requested_fold > 0 else "cv_summary.json"
    summary_path = base_dir / summary_name
    save_json(summary, summary_path)
    print(f"\nSaved cross-validation summary to {summary_path}")


def main() -> None:
    args = parse_args()
    if getattr(args, "deterministic", False):
        # Before any CUDA work, so CUBLAS sees the workspace configuration.
        enable_deterministic_execution()
        print("Deterministic execution | cudnn.deterministic=True benchmark=False tf32=off")
    args = apply_dataset_specific_overrides(args)
    args.seed = resolve_random_seed(args.seed)
    threshold_search_space = build_threshold_search_space(args)
    focal_alpha = args.focal_alpha if args.focal_alpha >= 0.0 else None
    print(f"Using train_seed={args.seed}, split_seed={args.split_seed}")
    device = torch.device(args.device)

    if args.cv_folds and args.cv_folds > 1:
        run_cross_validation(args, device, threshold_search_space, focal_alpha)
        return

    set_seed(args.seed)
    train_loader, val_loader, test_loader, metadata = build_dataloaders(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        num_workers=args.num_workers,
        max_graph_nodes=args.max_graph_nodes,
        early_cutoff_minutes=args.early_cutoff_minutes,
        node_token_root=(
            getattr(args, "node_token_root", None)
            if getattr(args, "joint_lora", False)
            else None
        ),
    )
    save_dir = Path(args.save_dir) / args.dataset_name
    train_and_test_once(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata=metadata,
        device=device,
        threshold_search_space=threshold_search_space,
        focal_alpha=focal_alpha,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
