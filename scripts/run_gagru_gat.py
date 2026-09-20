"""Train the retained GA-GRU + GAT model on PHEME, Weibo, or DRWeibo.

Every preset uses an outer BiGCN fold for one final test and an inner validation
split for checkpoint selection.  The outer test fold is never used for model
selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


PRESETS = {
    "pheme": {
        "dataset_name": "Pheme",
        "template": WORKSPACE / "dataset_pheme_bert_lora_pair_cls_cv5" / "fold{fold}",
        "fold": 5,
        "max_seq_len": 64,
        "max_graph_nodes": 0,
        "gat_attention_type": "gat",
        "gagru_aggregation": "attention",
        "gagru_readout": "mean",
        "graph_hidden_dim": 128,
        "graph_heads": 4,
        "semantic_hidden_dim": 128,
        "semantic_proj_dim": 256,
        "mask_ratio": 0.75,
        "lr": 0.0005,
        "weight_decay": 0.001,
        "dropout": 0.4,
        "grad_clip": 5.0,
        "patience": 12,
        "legacy_eval": True,
    },
    "weibo": {
        "dataset_name": "Weibo",
        "template": WORKSPACE / "dataset_weibo_independent_mean_cv5" / "fold{fold}",
        "fold": 1,
        "max_seq_len": 128,
        "max_graph_nodes": 512,
        "gat_attention_type": "gatv2",
        "gagru_aggregation": "mean",
        "gagru_readout": "mean",
        "graph_hidden_dim": 64,
        "graph_heads": 2,
        "semantic_hidden_dim": 64,
        "semantic_proj_dim": 64,
        "mask_ratio": 0.15,
        "lr": 0.0002,
        "weight_decay": 0.0005,
        "dropout": 0.2,
        "grad_clip": 2.0,
        "patience": 12,
        "legacy_eval": False,
    },
    "drweibo": {
        "dataset_name": "Weibo",
        "template": WORKSPACE / "dataset_drweibo_independent_mean_cv5" / "fold{fold}",
        "fold": 4,
        "max_seq_len": 128,
        "max_graph_nodes": 512,
        "gat_attention_type": "gatv2",
        "gagru_aggregation": "attention",
        "gagru_readout": "td-mean-bu-root",
        "graph_hidden_dim": 64,
        "graph_heads": 2,
        "semantic_hidden_dim": 64,
        "semantic_proj_dim": 64,
        "mask_ratio": 0.5,
        "lr": 0.0002,
        "weight_decay": 0.0005,
        "dropout": 0.2,
        "grad_clip": 2.0,
        "patience": 10,
        "legacy_eval": False,
    },
}


def build_training_args(
    dataset: str,
    output: Path,
    fold: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> list[str]:
    preset = PRESETS[dataset]
    selected_fold = int(preset["fold"] if fold is None else fold)
    if selected_fold not in range(1, 6):
        raise ValueError("fold must be in 1..5")
    template = Path(preset["template"])
    dataset_root = Path(str(template).format(fold=selected_fold))
    arguments = [
        "--dataset-name", str(preset["dataset_name"]),
        "--dataset-root", str(dataset_root),
        "--cv-fold-dataset-root-template", str(template),
        "--cv-folds", "5",
        "--cv-fold-index", str(selected_fold),
        "--cv-protocol", "bigcn",
        "--bigcn-inner-val-ratio", "0.1",
        "--selection-split", "validation",
        "--seed", str(seed),
        "--split-seed", "42",
        "--device", device,
        "--max-seq-len", str(preset["max_seq_len"]),
        "--max-graph-nodes", str(preset["max_graph_nodes"]),
        "--structure-encoder", "gat",
        "--gat-input-mode", "raw",
        "--gat-neighborhood", "local",
        "--gat-topology", "tree",
        "--gat-readout", "mean",
        "--gat-attention-type", str(preset["gat_attention_type"]),
        "--gat-residual-mode", "fixed",
        "--gat-residual-alpha", "1.0",
        "--disable-semantic-edge-attention",
        "--gagru-direction", "bidirectional",
        "--gagru-aggregation", str(preset["gagru_aggregation"]),
        "--gagru-attention-query", "global",
        "--gagru-attention-score-mode", "dot",
        "--gagru-attention-residual-mode", "fixed",
        "--gagru-attention-residual-alpha", "1.0",
        "--gagru-direction-fusion", "joint",
        "--gagru-semantic-direction-reduction", "sum",
        "--gagru-readout", str(preset["gagru_readout"]),
        "--gagru-bu-residual-init", "0.1",
        "--gagru-bu-shared-gradient-scale", "1.0",
        "--graph-hidden-dim", str(preset["graph_hidden_dim"]),
        "--graph-heads", str(preset["graph_heads"]),
        "--semantic-hidden-dim", str(preset["semantic_hidden_dim"]),
        "--semantic-proj-dim", str(preset["semantic_proj_dim"]),
        "--fusion-dim", "128",
        "--semantic-target", "raw",
        "--semantic-prediction-mode", "residual",
        "--semantic-loss-mse-weight", "0.1",
        "--semantic-loss-weight", "0.1",
        "--semantic-loss-decay-start-epoch", "5",
        "--semantic-loss-decay-end-epoch", "15",
        "--global-semantic-mask-ratio", str(preset["mask_ratio"]),
        "--global-semantic-target", "latent",
        "--global-semantic-loss-weight", "0.1",
        "--global-loss-warmup-epochs", "5",
        "--gat-pretrain-epochs", "0",
        "--gat-aux-loss-weight", "0",
        "--uniformity-loss-weight", "0",
        "--batch-size", "32",
        "--epochs", "60",
        "--patience", str(preset["patience"]),
        "--lr", str(preset["lr"]),
        "--weight-decay", str(preset["weight_decay"]),
        "--dropout", str(preset["dropout"]),
        "--edge-dropout", "0.05",
        "--grad-clip", str(preset["grad_clip"]),
        "--ema-decay", "0.995",
        "--lr-scheduler", "plateau",
        "--lr-plateau-factor", "0.5",
        "--lr-plateau-patience", "2",
        "--min-lr", "0.00001",
        "--label-smoothing", "0.02",
        "--loss-type", "ce",
        "--disable-class-weights",
        "--checkpoint-metric", "accuracy",
        "--fixed-threshold", "0.5",
        "--disable-threshold-search",
        "--num-workers", "4",
        "--deterministic",
        "--save-dir", str(output),
    ]
    if preset["legacy_eval"]:
        arguments.append("--legacy-eval-semantic-reconstruction")
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(PRESETS))
    parser.add_argument("--mode", choices=("check", "train"), default="check")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fold = args.fold or int(PRESETS[args.dataset]["fold"])
    output = args.output or (
        ROOT / "artifacts" / f"{args.dataset}_gagru_gat_fold{fold}_seed{args.seed}"
    )
    training_args = build_training_args(
        args.dataset, output.resolve(), fold=fold, seed=args.seed, device=args.device
    )

    if args.mode == "check":
        from dual_branch_rumor.train import parse_args

        original_argv = sys.argv
        try:
            sys.argv = ["train.py", *training_args]
            config = parse_args()
        finally:
            sys.argv = original_argv
        print(json.dumps(vars(config), ensure_ascii=False, indent=2, default=str))
        return

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    subprocess.run(
        [sys.executable, "-u", "-m", "dual_branch_rumor.train", *training_args],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
