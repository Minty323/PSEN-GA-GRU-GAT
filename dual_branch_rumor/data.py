from __future__ import annotations

import random
import pickle
import json
import math
import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REL_SELF_LOOP = 0
REL_TD_ROOT = 1
REL_TD_INTERNAL = 2
REL_TD_TERMINAL = 3
REL_BU_TO_ROOT = 4
REL_BU_INTERNAL = 5
REL_BU_FROM_LEAF = 6
NUM_EDGE_RELATIONS = 7

DATASET_NAME_ALIASES = {
    "pheme": "Pheme",
    "twitter15": "Twitter15",
    "twitter16": "Twitter16",
    "twitter15td": "Twitter15TD",
    "twitter15_td": "Twitter15TD",
    "twitter15-td": "Twitter15TD",
    "twitter15_text_tree": "Twitter15TD",
    "twitter15texttree": "Twitter15TD",
    "twitter16td": "Twitter16TD",
    "twitter16_td": "Twitter16TD",
    "twitter16-td": "Twitter16TD",
    "twitter16_text_tree": "Twitter16TD",
    "twitter16texttree": "Twitter16TD",
    "twitter15raw": "Twitter15Raw",
    "twitter15_raw": "Twitter15Raw",
    "twitter15-acl2017": "Twitter15Raw",
    "twitter15_acl2017": "Twitter15Raw",
    "acl2017_twitter15": "Twitter15Raw",
    "twitter15icdm": "Twitter15ICDM",
    "twitter15_icdm": "Twitter15ICDM",
    "twitter15-icdm": "Twitter15ICDM",
    "twitter15hetero": "Twitter15ICDM",
    "twitter15_hetero": "Twitter15ICDM",
    "twitter16icdm": "Twitter16ICDM",
    "twitter16_icdm": "Twitter16ICDM",
    "twitter16-icdm": "Twitter16ICDM",
    "twitter16hetero": "Twitter16ICDM",
    "twitter16_hetero": "Twitter16ICDM",
    "twitter16tfidf": "Twitter16TFIDF",
    "twitter16_tfidf": "Twitter16TFIDF",
    "twitter16-tfidf": "Twitter16TFIDF",
    "ragcl-twitter16": "Twitter16TFIDF",
    "ragcl_twitter16": "Twitter16TFIDF",
    "weibo": "Weibo",
    "phemeraw": "PhemeRaw",
    "pheme_raw": "PhemeRaw",
    "pheme6392078": "PhemeRaw",
    "pheme-raw": "PhemeRaw",
    "weibo21": "Weibo21",
    "weibo2021": "Weibo21",
}

PHEME_LABEL_MAP = {
    "non-rumours": 0,
    "non-rumor": 0,
    "rumours": 1,
    "rumor": 1,
}

TWITTER_LABEL_MAP = {
    "news": 0,
    "non-rumor": 0,
    "non-rumours": 0,
    "false": 1,
    "true": 2,
    "unverified": 3,
}

WEIBO_LABEL_MAP = {
    "true": 0,
    "false": 1,
}

WEIBO21_UNKNOWN_CATEGORY = "<unk>"

TWITTER_LABEL_TO_INDEX = {
    "non-rumor": 0,
    "false": 1,
    "true": 2,
    "unverified": 3,
}


def normalize_dataset_name(dataset_name: str) -> str:
    canonical_name = DATASET_NAME_ALIASES.get(dataset_name.strip().lower())
    if canonical_name is None:
        supported = ", ".join(DATASET_NAME_ALIASES.values())
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}.")
    return canonical_name


def resolve_dataset_root(dataset_root: Union[str, Path]) -> Path:
    root_path = Path(dataset_root).expanduser()
    candidate_paths = [root_path]

    if not root_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths.append(repo_root / root_path)

    resolved_candidates: List[Path] = []
    for candidate in candidate_paths:
        resolved = candidate.resolve(strict=False)
        if resolved not in resolved_candidates:
            resolved_candidates.append(resolved)

    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate

    searched_paths = ", ".join(str(path) for path in resolved_candidates)
    raise FileNotFoundError(
        f"Dataset root not found for '{dataset_root}'. Checked: {searched_paths}."
    )


def parse_label_file(dataset_name: str, dataset_dir: Path) -> Dict[str, int]:
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "Pheme":
        label_file = dataset_dir / "Pheme_label_All.txt"
        label_map = PHEME_LABEL_MAP
    elif dataset_name in {"Twitter15", "Twitter16"}:
        label_file = dataset_dir / f"{dataset_name}_label_All.txt"
        label_map = TWITTER_LABEL_MAP
    elif dataset_name == "Weibo":
        label_file = dataset_dir / "Weibo_label_All.txt"
        label_map = WEIBO_LABEL_MAP
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    labels: Dict[str, int] = {}
    with label_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label_name = parts[0].strip().lower()
            sample_id = parts[2].strip()
            if label_name in label_map:
                labels[sample_id] = label_map[label_name]
    return labels


def infer_num_classes_from_labels(labels: Dict[str, int]) -> int:
    if not labels:
        return 0
    return max(labels.values()) + 1


def stratified_split(
    sample_ids: Sequence[str],
    labels: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    by_label: Dict[int, List[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_label[labels[sample_id]].append(sample_id)

    rng = random.Random(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for label_ids in by_label.values():
        label_ids = list(label_ids)
        rng.shuffle(label_ids)
        total = len(label_ids)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        if total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        if n_train + n_val >= total:
            n_val = max(1, total - n_train - 1)
        train_ids.extend(label_ids[:n_train])
        val_ids.extend(label_ids[n_train : n_train + n_val])
        test_ids.extend(label_ids[n_train + n_val :])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def stratified_kfold(
    sample_ids: Sequence[str],
    labels: Dict[str, int],
    n_folds: int,
    val_ratio: float,
    seed: int,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """Stratified K-fold split. Each fold is (train_ids, val_ids, test_ids).

    Fold k uses the k-th stratified chunk of every class as the test set, and
    carves a stratified validation set (val_ratio of the remaining samples) out
    of the training pool for early stopping / threshold selection.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    by_label: Dict[int, List[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_label[labels[sample_id]].append(sample_id)

    # Round-robin assign each class's shuffled ids into n_folds balanced chunks.
    label_chunks: Dict[int, List[List[str]]] = {}
    for label, ids in by_label.items():
        ids = list(ids)
        random.Random(seed + int(label)).shuffle(ids)
        chunks: List[List[str]] = [[] for _ in range(n_folds)]
        for position, sample_id in enumerate(ids):
            chunks[position % n_folds].append(sample_id)
        label_chunks[label] = chunks

    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_index in range(n_folds):
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        for label, chunks in label_chunks.items():
            test_ids.extend(chunks[fold_index])
            rest: List[str] = []
            for other_index in range(n_folds):
                if other_index != fold_index:
                    rest.extend(chunks[other_index])
            random.Random(seed * 131 + fold_index * 17 + int(label)).shuffle(rest)
            n_val = int(len(rest) * val_ratio)
            if len(rest) >= 2 and val_ratio > 0.0:
                n_val = max(1, n_val)
            val_ids.extend(rest[:n_val])
            train_ids.extend(rest[n_val:])
        shuffler = random.Random(seed + fold_index)
        shuffler.shuffle(train_ids)
        shuffler.shuffle(val_ids)
        shuffler.shuffle(test_ids)
        folds.append((train_ids, val_ids, test_ids))
    return folds


def stratified_bigcn_kfold(
    sample_ids: Sequence[str],
    labels: Dict[str, int],
    n_folds: int,
    seed: int,
    val_ratio: float = 0.0,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """Reproduce the class-wise 80/20 five-fold split used by BiGCN.

    Every class is shuffled independently and contributes ``floor(class_size *
    0.2)`` samples to each test fold. Class remainders are therefore present in
    every training fold and never occur in a test fold, matching BiGCN's
    ``Process/rand5fold.py`` behavior. By default, the middle list is empty
    because BiGCN does not create an independent validation split. When
    ``val_ratio`` is positive, a deterministic class-stratified validation set
    is carved out of each fold's outer training pool without changing the
    corresponding outer test-fold membership.

    BiGCN leaves Python's global RNG unseeded. This implementation uses the
    configured split seed so that the otherwise identical protocol is
    reproducible.
    """
    if n_folds != 5:
        raise ValueError("The BiGCN-compatible protocol requires exactly 5 folds.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    by_label: Dict[int, List[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_label[labels[sample_id]].append(sample_id)

    rng = random.Random(seed)
    shuffled_by_label: Dict[int, List[str]] = {}
    fold_sizes: Dict[int, int] = {}
    for label in sorted(by_label):
        ids = list(by_label[label])
        rng.shuffle(ids)
        shuffled_by_label[label] = ids
        fold_sizes[label] = int(len(ids) * 0.2)

    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_index in range(n_folds):
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        for label in sorted(shuffled_by_label):
            ids = shuffled_by_label[label]
            fold_size = fold_sizes[label]
            start = fold_index * fold_size
            end = start + fold_size
            test_ids.extend(ids[start:end])

            outer_train_ids = ids[:start] + ids[end:]
            if val_ratio > 0.0:
                inner_rng = random.Random(seed * 131 + fold_index * 17 + int(label))
                inner_rng.shuffle(outer_train_ids)
                n_val = int(len(outer_train_ids) * val_ratio)
                if len(outer_train_ids) >= 2:
                    n_val = max(1, n_val)
                n_val = min(n_val, max(0, len(outer_train_ids) - 1))
                val_ids.extend(outer_train_ids[:n_val])
                train_ids.extend(outer_train_ids[n_val:])
            else:
                train_ids.extend(outer_train_ids)

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)
        folds.append((train_ids, val_ids, test_ids))
    return folds


@dataclass
class DatasetMetadata:
    graph_input_dim: int
    num_classes: int = 2
    label_to_index: Optional[Dict[str, int]] = None
    graph_feature_key: str = "x"
    graph_feature_transposed: bool = False
    max_graph_nodes: Optional[int] = None
    num_edge_relations: int = NUM_EDGE_RELATIONS
    source_split_files: bool = False
    uses_synthetic_graph: bool = False
    category_feature_size: Optional[int] = None
    category_to_index: Optional[Dict[str, int]] = None
    early_cutoff_minutes: Optional[float] = None
    has_time_metadata: bool = False
    has_node_tokens: bool = False
    node_token_length: Optional[int] = None
    has_semantic_features: bool = False


@dataclass
class Weibo21Record:
    sample_id: str
    content: str
    label: int
    category: str


def normalize_weibo21_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def resolve_weibo21_dir(dataset_root: Union[str, Path]) -> Path:
    root_path = resolve_dataset_root(dataset_root)
    candidate_dirs = [root_path / "weibo21", root_path / "Weibo21"]
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            return candidate_dir
    searched_paths = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"Weibo21 dataset directory not found. Checked: {searched_paths}.")


def load_weibo21_records(dataset_dir: Path, split: str) -> List[Weibo21Record]:
    split_path = dataset_dir / f"{split}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"Weibo21 split file not found: {split_path}")

    try:
        with split_path.open("rb") as handle:
            data_frame = pickle.load(handle)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pandas is required to load Weibo21 .pkl split files. "
            "Install it with: pip install pandas"
        ) from exc

    required_columns = {"content", "label", "category"}
    if not hasattr(data_frame, "columns") or not required_columns.issubset(set(data_frame.columns)):
        raise ValueError(
            f"Weibo21 split {split_path} must contain columns: {sorted(required_columns)}."
        )

    records: List[Weibo21Record] = []
    for row_offset, (_, row) in enumerate(data_frame.iterrows()):
        content = normalize_weibo21_value(row["content"])
        category = normalize_weibo21_value(row["category"]) or WEIBO21_UNKNOWN_CATEGORY
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Weibo21 label at {split}:{row_offset}: {row['label']!r}") from exc
        if label not in {0, 1}:
            raise ValueError(f"Unsupported Weibo21 label at {split}:{row_offset}: {label}.")
        records.append(
            Weibo21Record(
                sample_id=f"{split}-{row_offset}",
                content=content,
                label=label,
                category=category,
            )
        )

    if not records:
        raise RuntimeError(f"No records were loaded from Weibo21 split: {split_path}")
    return records


def build_weibo21_category_vocab(records: Iterable[Weibo21Record]) -> Dict[str, int]:
    category_to_index = {WEIBO21_UNKNOWN_CATEGORY: 0}
    for category in sorted({record.category for record in records if record.category}):
        if category not in category_to_index:
            category_to_index[category] = len(category_to_index)
    return category_to_index


class Weibo21GraphDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Weibo21Record],
        category_to_index: Dict[str, int],
    ) -> None:
        self.records = list(records)
        self.category_to_index = category_to_index
        self.sample_ids = [record.sample_id for record in self.records]
        self.labels = {record.sample_id: record.label for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def get_label(self, index: int) -> int:
        return self.records[index].label

    def _build_root_graph(self, category: str) -> Dict[str, torch.Tensor]:
        category_index = self.category_to_index.get(category, self.category_to_index[WEIBO21_UNKNOWN_CATEGORY])
        node_features = torch.zeros((1, len(self.category_to_index)), dtype=torch.float32)
        node_features[0, category_index] = 1.0
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_edge_type = torch.empty((0,), dtype=torch.long)
        return {
            "node_features": node_features,
            "edge_index_td": empty_edge_index,
            "edge_index_bu": empty_edge_index,
            "edge_type_td": empty_edge_type,
            "edge_type_bu": empty_edge_type,
            "root_index": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Dict[str, torch.Tensor]]:
        record = self.records[index]
        return {
            "id": record.sample_id,
            "label": torch.tensor(record.label, dtype=torch.long),
            "graph": self._build_root_graph(record.category),
        }


@dataclass
class TwitterAcl2017Record:
    sample_id: str
    label: int
    source_text: str
    node_features: np.ndarray
    edge_index: np.ndarray
    root_index: int


TWITTER_ACL2017_GRAPH_FEATURE_NAMES = [
    "is_root",
    "depth_log1p",
    "out_degree_log1p",
    "in_degree_log1p",
    "time_delay_minutes_log1p",
    "is_leaf",
    "has_numeric_user_id",
]


@dataclass
class Twitter15IcdmRecord:
    sample_id: str
    label: int
    source_text: str
    split: str
    node_features: np.ndarray
    edge_index: np.ndarray
    root_index: int = 0


TWITTER15_ICDM_GRAPH_FEATURE_NAMES = [
    "is_root",
    "is_user",
    "relation_weight",
    "relation_weight_log1p",
    "rank_fraction",
    "degree_log1p",
    "mean_relation_weight",
    "max_relation_weight",
]


@dataclass
class TwitterSparseTreeRecord:
    sample_id: str
    label: int
    node_sparse_counts: List[List[Tuple[int, float]]]
    edge_index: np.ndarray
    root_index: int


def resolve_twitter_acl2017_dir(dataset_root: Union[str, Path], subset_name: str = "twitter15") -> Path:
    root_path = resolve_dataset_root(dataset_root)
    candidate_dirs = [
        root_path / "rumdetect2017 (1)" / "rumor_detection_acl2017" / subset_name,
        root_path / "rumdetect2017" / "rumor_detection_acl2017" / subset_name,
        root_path / "rumor_detection_acl2017" / subset_name,
        root_path / subset_name,
        root_path,
    ]

    for candidate_dir in candidate_dirs:
        if (
            candidate_dir.exists()
            and (candidate_dir / "label.txt").exists()
            and (candidate_dir / "source_tweets.txt").exists()
            and (candidate_dir / "tree").exists()
        ):
            return candidate_dir

    searched_paths = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"Twitter ACL2017 directory not found. Checked: {searched_paths}.")


def load_twitter_acl2017_labels(dataset_dir: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with (dataset_dir / "label.txt").open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            label_name, sample_id = line.split(":", 1)
            mapped_label = TWITTER_LABEL_MAP.get(label_name.strip().lower())
            if mapped_label is not None:
                labels[sample_id.strip()] = mapped_label
    return labels


def load_twitter_acl2017_source_texts(dataset_dir: Path) -> Dict[str, str]:
    source_texts: Dict[str, str] = {}
    with (dataset_dir / "source_tweets.txt").open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sample_id, text = parts
            source_texts[sample_id.strip()] = text.strip()
    return source_texts


def parse_acl2017_tree_node(raw_node: str) -> Tuple[str, str, float]:
    parsed = ast.literal_eval(raw_node.strip())
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        raise ValueError(f"Invalid ACL2017 tree node: {raw_node!r}")

    user_id = str(parsed[0])
    tweet_id = str(parsed[1])
    try:
        delay_minutes = float(parsed[2])
    except (TypeError, ValueError):
        delay_minutes = 0.0
    return user_id, tweet_id, max(delay_minutes, 0.0)


def parse_acl2017_tree_file(tree_path: Path, root_id: str) -> Tuple[np.ndarray, np.ndarray, int]:
    node_order: List[str] = []
    node_user_ids: Dict[str, str] = {}
    node_delay: Dict[str, float] = {}
    edges: List[Tuple[str, str]] = []

    def add_node(tweet_id: str, user_id: str, delay_minutes: float) -> None:
        if tweet_id not in node_delay:
            node_order.append(tweet_id)
        node_user_ids[tweet_id] = user_id
        node_delay[tweet_id] = delay_minutes

    with tree_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "->" not in line:
                continue
            parent_raw, child_raw = line.split("->", 1)
            parent_user_id, parent_tweet_id, parent_delay = parse_acl2017_tree_node(parent_raw)
            child_user_id, child_tweet_id, child_delay = parse_acl2017_tree_node(child_raw)

            if parent_tweet_id != "ROOT":
                add_node(parent_tweet_id, parent_user_id, parent_delay)
            add_node(child_tweet_id, child_user_id, child_delay)
            if parent_tweet_id != "ROOT":
                edges.append((parent_tweet_id, child_tweet_id))

    if root_id not in node_delay:
        add_node(root_id, "ROOT", 0.0)

    node_to_index = {node_id: index for index, node_id in enumerate(node_order)}
    indexed_edges = [
        (node_to_index[parent_id], node_to_index[child_id])
        for parent_id, child_id in edges
        if parent_id in node_to_index and child_id in node_to_index
    ]

    out_degree: Dict[str, int] = defaultdict(int)
    in_degree: Dict[str, int] = defaultdict(int)
    children_by_node: Dict[str, List[str]] = defaultdict(list)
    for parent_id, child_id in edges:
        out_degree[parent_id] += 1
        in_degree[child_id] += 1
        children_by_node[parent_id].append(child_id)

    depths = {node_id: 0 for node_id in node_order}
    queue: deque[Tuple[str, int]] = deque([(root_id, 0)])
    seen = {root_id}
    while queue:
        node_id, depth = queue.popleft()
        depths[node_id] = depth
        for child_id in children_by_node.get(node_id, []):
            if child_id not in seen:
                seen.add(child_id)
                queue.append((child_id, depth + 1))

    features: List[List[float]] = []
    for node_id in node_order:
        user_id = node_user_ids.get(node_id, "")
        features.append(
            [
                1.0 if node_id == root_id else 0.0,
                safe_log1p(depths.get(node_id, 0)),
                safe_log1p(out_degree.get(node_id, 0)),
                safe_log1p(in_degree.get(node_id, 0)),
                safe_log1p(node_delay.get(node_id, 0.0)),
                1.0 if out_degree.get(node_id, 0) == 0 else 0.0,
                1.0 if user_id.isdigit() else 0.0,
            ]
        )

    if indexed_edges:
        edge_index = np.asarray(indexed_edges, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return np.asarray(features, dtype=np.float32), edge_index, node_to_index[root_id]


def load_twitter_acl2017_records(dataset_root: Union[str, Path], subset_name: str = "twitter15") -> List[TwitterAcl2017Record]:
    dataset_dir = resolve_twitter_acl2017_dir(dataset_root, subset_name=subset_name)
    labels = load_twitter_acl2017_labels(dataset_dir)
    source_texts = load_twitter_acl2017_source_texts(dataset_dir)
    tree_dir = dataset_dir / "tree"

    records: List[TwitterAcl2017Record] = []
    for tree_path in sorted(tree_dir.glob("*.txt")):
        sample_id = tree_path.stem
        if sample_id not in labels or sample_id not in source_texts:
            continue
        node_features, edge_index, root_index = parse_acl2017_tree_file(tree_path, sample_id)
        records.append(
            TwitterAcl2017Record(
                sample_id=sample_id,
                label=labels[sample_id],
                source_text=source_texts[sample_id],
                node_features=node_features,
                edge_index=edge_index,
                root_index=root_index,
            )
        )

    if not records:
        raise RuntimeError(f"No Twitter ACL2017 records were loaded from: {dataset_dir}")
    return records


def resolve_twitter_icdm_dir(dataset_root: Union[str, Path], subset_name: str) -> Path:
    root_path = resolve_dataset_root(dataset_root)
    nested_name = "rumor-detection-include-twitter15-twitter16data--master"
    candidate_dirs = [
        root_path / subset_name,
        root_path / nested_name / nested_name / "dataset" / subset_name,
        root_path / nested_name / "dataset" / subset_name,
        root_path / "dataset" / subset_name,
        root_path,
    ]
    for candidate_dir in candidate_dirs:
        if (
            candidate_dir.exists()
            and (candidate_dir / f"{subset_name}.train").exists()
            and (candidate_dir / f"{subset_name}.dev").exists()
            and (candidate_dir / f"{subset_name}.test").exists()
            and (candidate_dir / f"{subset_name}_graph.txt").exists()
        ):
            return candidate_dir

    searched_paths = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"{subset_name} ICDM dataset directory not found. Checked: {searched_paths}.")


def resolve_twitter15_icdm_dir(dataset_root: Union[str, Path]) -> Path:
    return resolve_twitter_icdm_dir(dataset_root, "twitter15")


def read_twitter_icdm_split(dataset_dir: Path, split: str, subset_name: str) -> List[Tuple[str, str, int]]:
    split_path = dataset_dir / f"{subset_name}.{split}"
    records: List[Tuple[str, str, int]] = []
    with split_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid {subset_name} ICDM row at {split_path}:{line_number}: {line!r}")
            sample_id, source_text, label_name = parts
            mapped_label = TWITTER_LABEL_MAP.get(label_name.strip().lower())
            if mapped_label is None:
                raise ValueError(f"Unsupported {subset_name} ICDM label at {split_path}:{line_number}: {label_name!r}")
            records.append((sample_id.strip(), source_text.strip(), mapped_label))
    return records


def read_twitter15_icdm_split(dataset_dir: Path, split: str) -> List[Tuple[str, str, int]]:
    return read_twitter_icdm_split(dataset_dir, split, "twitter15")


def load_twitter_icdm_user_relations(dataset_dir: Path, subset_name: str) -> Dict[str, List[Tuple[str, float]]]:
    relations: Dict[str, List[Tuple[str, float]]] = {}
    graph_path = dataset_dir / f"{subset_name}_graph.txt"
    with graph_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts:
                continue
            sample_id = parts[0]
            user_weights: List[Tuple[str, float]] = []
            for item in parts[1:]:
                if ":" not in item:
                    continue
                user_id, weight_text = item.split(":", 1)
                try:
                    weight = float(weight_text)
                except ValueError:
                    continue
                user_weights.append((user_id, max(weight, 0.0)))
            relations[sample_id] = user_weights
    return relations


def load_twitter15_icdm_user_relations(dataset_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    return load_twitter_icdm_user_relations(dataset_dir, "twitter15")


def build_twitter_icdm_graph(user_weights: Sequence[Tuple[str, float]]) -> Tuple[np.ndarray, np.ndarray, int]:
    weights = [float(weight) for _, weight in user_weights]
    degree = len(weights)
    mean_weight = float(np.mean(weights)) if weights else 0.0
    max_weight = float(np.max(weights)) if weights else 0.0
    degree_log = math.log1p(degree)

    node_features: List[List[float]] = [
        [
            1.0,
            0.0,
            1.0 if max_weight > 0.0 else 0.0,
            math.log1p(max_weight),
            0.0,
            degree_log,
            mean_weight,
            max_weight,
        ]
    ]

    edge_list: List[Tuple[int, int]] = []
    denom = max(degree - 1, 1)
    sorted_weights = sorted(enumerate(weights), key=lambda item: item[1], reverse=True)
    rank_by_index = {original_index: rank for rank, (original_index, _) in enumerate(sorted_weights)}
    for user_offset, (_, weight) in enumerate(user_weights, start=1):
        rank_fraction = rank_by_index[user_offset - 1] / denom if degree > 1 else 0.0
        node_features.append(
            [
                0.0,
                1.0,
                float(weight),
                math.log1p(float(weight)),
                float(rank_fraction),
                degree_log,
                mean_weight,
                max_weight,
            ]
        )
        edge_list.append((0, user_offset))

    if edge_list:
        edge_index = np.asarray(edge_list, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    return np.asarray(node_features, dtype=np.float32), edge_index, 0


def build_twitter15_icdm_graph(user_weights: Sequence[Tuple[str, float]]) -> Tuple[np.ndarray, np.ndarray, int]:
    return build_twitter_icdm_graph(user_weights)


def load_twitter_icdm_records(
    dataset_root: Union[str, Path],
    subset_name: str,
) -> Tuple[List[Twitter15IcdmRecord], Dict[str, int]]:
    dataset_dir = resolve_twitter_icdm_dir(dataset_root, subset_name)
    user_relations = load_twitter_icdm_user_relations(dataset_dir, subset_name)
    records: List[Twitter15IcdmRecord] = []
    split_counts: Dict[str, int] = {}
    for split in ("train", "dev", "test"):
        split_rows = read_twitter_icdm_split(dataset_dir, split, subset_name)
        split_counts[split] = len(split_rows)
        for sample_id, source_text, label in split_rows:
            node_features, edge_index, root_index = build_twitter_icdm_graph(user_relations.get(sample_id, []))
            records.append(
                Twitter15IcdmRecord(
                    sample_id=sample_id,
                    label=label,
                    source_text=source_text,
                    split=split,
                    node_features=node_features,
                    edge_index=edge_index,
                    root_index=root_index,
                )
            )

    if not records:
        raise RuntimeError(f"No {subset_name} ICDM records were loaded from: {dataset_dir}")
    return records, split_counts


def load_twitter15_icdm_records(dataset_root: Union[str, Path]) -> Tuple[List[Twitter15IcdmRecord], Dict[str, int]]:
    return load_twitter_icdm_records(dataset_root, "twitter15")


def parse_sparse_count_pairs(token_blob: str) -> List[Tuple[int, float]]:
    sparse_counts: List[Tuple[int, float]] = []
    for item in token_blob.strip().split():
        if ":" not in item:
            continue
        token_idx_text, count_text = item.split(":", 1)
        try:
            token_idx = int(token_idx_text)
            count = float(count_text)
        except ValueError:
            continue
        if token_idx < 0 or count <= 0:
            continue
        sparse_counts.append((token_idx, count))
    return sparse_counts


def resolve_twitter_sparse_tree_dir(dataset_root: Union[str, Path], dataset_name: str) -> Path:
    root_path = resolve_dataset_root(dataset_root)
    candidate_dir = root_path / dataset_name
    if (
        candidate_dir.exists()
        and (candidate_dir / f"{dataset_name}_label_All.txt").exists()
        and (candidate_dir / "data.TD_RvNN.vol_5000.txt").exists()
    ):
        return candidate_dir
    raise FileNotFoundError(f"Twitter sparse tree dataset directory not found: {candidate_dir}")


def load_twitter_sparse_tree_records(
    dataset_root: Union[str, Path],
    dataset_name: str,
    max_seq_len: int,
    graph_feature_dim: int = 5000,
) -> List[TwitterSparseTreeRecord]:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name in {"Twitter15TD", "Twitter16TD"}:
        label_dataset_name = dataset_name.replace("TD", "")
        dataset_dir_name = label_dataset_name
    else:
        label_dataset_name = dataset_name
        dataset_dir_name = dataset_name

    dataset_dir = resolve_twitter_sparse_tree_dir(dataset_root, dataset_dir_name)
    labels = parse_label_file(label_dataset_name, dataset_dir)
    tree_nodes: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(dict)
    edge_pairs_by_root: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    root_node_id_by_root: Dict[str, str] = {}
    td_path = dataset_dir / "data.TD_RvNN.vol_5000.txt"
    with td_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            sample_id, parent_id, current_id = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if sample_id not in labels:
                continue
            sparse_counts = parse_sparse_count_pairs(parts[5])
            tree_nodes[sample_id][current_id] = sparse_counts
            if parent_id == "None":
                root_node_id_by_root[sample_id] = current_id
            else:
                edge_pairs_by_root[sample_id].append((parent_id, current_id))

    records: List[TwitterSparseTreeRecord] = []
    for sample_id in sorted(labels):
        nodes = tree_nodes.get(sample_id)
        root_node_id = root_node_id_by_root.get(sample_id)
        if not nodes or root_node_id is None:
            continue
        if len(nodes) < 2:
            continue

        node_ids = sorted(nodes, key=lambda node_id: int(node_id))
        node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
        sparse_nodes = [nodes.get(node_id, []) for node_id in node_ids]
        edge_list = sorted(
            (
                (node_to_index[parent_id], node_to_index[child_id])
                for parent_id, child_id in edge_pairs_by_root.get(sample_id, [])
                if parent_id in node_to_index and child_id in node_to_index
            ),
            key=lambda edge: (edge[0], edge[1]),
        )
        edge_index = (
            np.asarray(edge_list, dtype=np.int64).T
            if edge_list
            else np.empty((2, 0), dtype=np.int64)
        )
        records.append(
            TwitterSparseTreeRecord(
                sample_id=sample_id,
                label=labels[sample_id],
                node_sparse_counts=sparse_nodes,
                edge_index=edge_index,
                root_index=node_to_index[root_node_id],
            )
        )

    if not records:
        raise RuntimeError(f"No Twitter sparse tree records were loaded from: {dataset_dir}")
    return records


class TwitterSparseTreeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TwitterSparseTreeRecord],
        max_seq_len: int,
        graph_feature_dim: int = 5000,
        max_graph_nodes: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.max_seq_len = max_seq_len
        self.graph_feature_dim = graph_feature_dim
        self.max_graph_nodes = max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None
        self.sample_ids = [record.sample_id for record in self.records]
        self.labels = {record.sample_id: record.label for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def get_label(self, index: int) -> int:
        return self.records[index].label

    def _build_node_features(self, record: TwitterSparseTreeRecord) -> np.ndarray:
        node_features = np.zeros((len(record.node_sparse_counts), self.graph_feature_dim), dtype=np.float32)
        for node_index, sparse_counts in enumerate(record.node_sparse_counts):
            for token_idx, count in sparse_counts:
                if 0 <= token_idx < self.graph_feature_dim:
                    node_features[node_index, token_idx] = float(count)
        return node_features

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Dict[str, torch.Tensor]]:
        record = self.records[index]
        node_features_np = self._build_node_features(record)
        edge_index_np = record.edge_index
        root_index = record.root_index
        node_features_np, edge_index_np, root_index = RumorGraphDataset._truncate_graph(
            node_features=node_features_np,
            edge_index=edge_index_np,
            root_index=root_index,
            max_graph_nodes=self.max_graph_nodes,
        )
        edge_type_td = RumorGraphDataset._build_td_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )
        edge_type_bu = RumorGraphDataset._build_bu_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )
        edge_index_td = torch.tensor(edge_index_np, dtype=torch.long)
        return {
            "id": record.sample_id,
            "label": torch.tensor(record.label, dtype=torch.long),
            "graph": {
                "node_features": torch.tensor(node_features_np, dtype=torch.float32),
                "edge_index_td": edge_index_td,
                "edge_index_bu": edge_index_td.flip(0),
                "edge_type_td": torch.tensor(edge_type_td, dtype=torch.long),
                "edge_type_bu": torch.tensor(edge_type_bu, dtype=torch.long),
                "root_index": torch.tensor(root_index, dtype=torch.long),
            },
        }


@dataclass
class RagclTFIDFRecord:
    sample_id: str
    label: int
    node_sparse_counts: List[List[Tuple[int, float]]]
    edge_index: np.ndarray
    root_index: int = 0


def resolve_ragcl_tfidf_dir(dataset_root: Union[str, Path], dataset_name: str) -> Path:
    root_path = resolve_dataset_root(dataset_root)
    folder_name = "Twitter16-tfidf" if dataset_name == "Twitter16TFIDF" else dataset_name
    candidate_dirs = [
        root_path / folder_name,
        root_path,
    ]
    for candidate_dir in candidate_dirs:
        if (candidate_dir / "source").exists():
            return candidate_dir
    searched = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"RAGCL TF-IDF dataset not found. Checked: {searched}.")


def load_ragcl_tfidf_records(
    dataset_root: Union[str, Path],
    dataset_name: str,
    max_seq_len: int,
) -> List[RagclTFIDFRecord]:
    dataset_dir = resolve_ragcl_tfidf_dir(dataset_root, dataset_name)
    source_dir = dataset_dir / "source"
    records: List[RagclTFIDFRecord] = []

    for json_path in sorted(source_dir.glob("*.json")):
        post = read_json_file(json_path)
        source = post.get("source", {})
        sample_id = str(source.get("tweet id") or json_path.stem)
        try:
            label = int(source["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RAGCL label in {json_path}") from exc

        comments = sorted(
            post.get("comment", []),
            key=lambda item: int(item.get("comment id", 0)),
        )
        comment_id_to_node = {
            int(comment.get("comment id", index)): index + 1
            for index, comment in enumerate(comments)
        }

        node_sparse_counts = [parse_sparse_count_pairs(str(source.get("content", "")))]
        for comment in comments:
            node_sparse_counts.append(parse_sparse_count_pairs(str(comment.get("content", ""))))

        edges: List[Tuple[int, int]] = []
        for comment in comments:
            child_id = int(comment.get("comment id", -1))
            child_node = comment_id_to_node.get(child_id)
            if child_node is None:
                continue
            parent_id = int(comment.get("parent", -1))
            parent_node = 0 if parent_id == -1 else comment_id_to_node.get(parent_id)
            if parent_node is None:
                continue
            edges.append((parent_node, child_node))

        edge_index = (
            np.asarray(edges, dtype=np.int64).T
            if edges
            else np.empty((2, 0), dtype=np.int64)
        )
        records.append(
            RagclTFIDFRecord(
                sample_id=sample_id,
                label=label,
                node_sparse_counts=node_sparse_counts,
                edge_index=edge_index,
            )
        )

    if not records:
        raise RuntimeError(f"No RAGCL TF-IDF records were loaded from: {source_dir}")
    return records


class RagclTFIDFDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RagclTFIDFRecord],
        max_seq_len: int,
        graph_feature_dim: int = 5000,
        max_graph_nodes: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.max_seq_len = max_seq_len
        self.graph_feature_dim = graph_feature_dim
        self.max_graph_nodes = max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None
        self.sample_ids = [record.sample_id for record in self.records]
        self.labels = {record.sample_id: record.label for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def get_label(self, index: int) -> int:
        return self.records[index].label

    def _build_node_features(self, record: RagclTFIDFRecord) -> np.ndarray:
        node_features = np.zeros((len(record.node_sparse_counts), self.graph_feature_dim), dtype=np.float32)
        for node_index, sparse_counts in enumerate(record.node_sparse_counts):
            for token_idx, count in sparse_counts:
                if 0 <= token_idx < self.graph_feature_dim:
                    node_features[node_index, token_idx] = float(count)
        return node_features

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Dict[str, torch.Tensor]]:
        record = self.records[index]
        node_features_np = self._build_node_features(record)
        edge_index_np = record.edge_index
        root_index = record.root_index
        node_features_np, edge_index_np, root_index = RumorGraphDataset._truncate_graph(
            node_features=node_features_np,
            edge_index=edge_index_np,
            root_index=root_index,
            max_graph_nodes=self.max_graph_nodes,
        )
        edge_type_td = RumorGraphDataset._build_td_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )
        edge_type_bu = RumorGraphDataset._build_bu_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )
        edge_index_td = torch.tensor(edge_index_np, dtype=torch.long)
        return {
            "id": record.sample_id,
            "label": torch.tensor(record.label, dtype=torch.long),
            "graph": {
                "node_features": torch.tensor(node_features_np, dtype=torch.float32),
                "edge_index_td": edge_index_td,
                "edge_index_bu": edge_index_td.flip(0),
                "edge_type_td": torch.tensor(edge_type_td, dtype=torch.long),
                "edge_type_bu": torch.tensor(edge_type_bu, dtype=torch.long),
                "root_index": torch.tensor(root_index, dtype=torch.long),
            },
        }


PHEME_RAW_GRAPH_FEATURE_NAMES = [
    "is_root",
    "depth_log1p",
    "out_degree_log1p",
    "text_length_log1p",
    "retweet_count_log1p",
    "favorite_count_log1p",
    "followers_count_log1p",
    "friends_count_log1p",
    "statuses_count_log1p",
    "favourites_count_log1p",
    "listed_count_log1p",
    "user_verified",
    "user_default_profile",
    "has_url",
    "url_count_log1p",
    "hashtag_count_log1p",
    "mention_count_log1p",
    "is_reply",
    "time_delta_minutes_log1p",
    "is_missing_tweet_json",
]


@dataclass
class PhemeRawRecord:
    sample_id: str
    label: int
    source_text: str
    event_name: str
    node_features: np.ndarray
    edge_index: np.ndarray
    root_index: int


def read_json_file(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


def resolve_pheme_raw_threads_root(dataset_root: Union[str, Path]) -> Path:
    root_path = resolve_dataset_root(dataset_root)
    candidate_dirs = [
        root_path / "6392078" / "PHEME_veracity" / "all-rnr-annotated-threads",
        root_path / "PHEME_veracity" / "all-rnr-annotated-threads",
        root_path / "all-rnr-annotated-threads",
    ]

    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            return candidate_dir

    searched_paths = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"PhemeRaw thread root not found. Checked: {searched_paths}.")


def parse_twitter_created_at(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def safe_log1p(value: object) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric_value <= 0.0:
        return 0.0
    return float(math.log1p(numeric_value))


def extract_tweet_text(tweet: Dict[str, object]) -> str:
    text = tweet.get("full_text") or tweet.get("text") or ""
    return str(text).replace("\n", " ").strip()


def flatten_pheme_structure(structure: Dict[str, object]) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, int]]:
    node_ids: List[str] = []
    edges: List[Tuple[str, str]] = []
    depths: Dict[str, int] = {}

    def add_node(node_id: str, depth: int) -> None:
        if node_id not in depths:
            node_ids.append(node_id)
            depths[node_id] = depth
        else:
            depths[node_id] = min(depths[node_id], depth)

    def visit(node: object, depth: int, parent_id: Optional[str] = None) -> None:
        if not isinstance(node, dict):
            return
        for raw_node_id, children in node.items():
            node_id = str(raw_node_id)
            add_node(node_id, depth)
            if parent_id is not None:
                edges.append((parent_id, node_id))
            if isinstance(children, dict):
                visit(children, depth + 1, node_id)

    visit(structure, depth=0)
    return node_ids, edges, depths


def find_source_tweet_path(sample_dir: Path, sample_id: str) -> Optional[Path]:
    for dirname in ("source-tweets", "source-tweet"):
        source_dir = sample_dir / dirname
        if not source_dir.exists():
            continue
        exact_path = source_dir / f"{sample_id}.json"
        if exact_path.exists():
            return exact_path
        json_paths = sorted(source_dir.glob("*.json"))
        if json_paths:
            return json_paths[0]
    return None


def load_pheme_tweet_jsons(sample_dir: Path, sample_id: str) -> Tuple[Dict[str, Dict[str, object]], str, Optional[datetime]]:
    tweets: Dict[str, Dict[str, object]] = {}
    source_text = ""
    root_time = None

    source_path = find_source_tweet_path(sample_dir, sample_id)
    if source_path is not None:
        source_tweet = read_json_file(source_path)
        source_tweet_id = str(source_tweet.get("id_str") or source_tweet.get("id") or sample_id)
        tweets[source_tweet_id] = source_tweet
        tweets[str(sample_id)] = source_tweet
        source_text = extract_tweet_text(source_tweet)
        root_time = parse_twitter_created_at(source_tweet.get("created_at"))

    reactions_dir = sample_dir / "reactions"
    if reactions_dir.exists():
        for reaction_path in reactions_dir.glob("*.json"):
            try:
                reaction_tweet = read_json_file(reaction_path)
            except (json.JSONDecodeError, OSError):
                continue
            reaction_id = str(reaction_tweet.get("id_str") or reaction_tweet.get("id") or reaction_path.stem)
            tweets[reaction_id] = reaction_tweet

    return tweets, source_text, root_time


def build_pheme_raw_node_features(
    node_ids: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    depths: Dict[str, int],
    tweets: Dict[str, Dict[str, object]],
    root_id: str,
    root_time: Optional[datetime],
) -> np.ndarray:
    out_degree: Dict[str, int] = defaultdict(int)
    for src, _ in edges:
        out_degree[src] += 1

    features: List[List[float]] = []
    for node_id in node_ids:
        tweet = tweets.get(node_id, {})
        user = tweet.get("user", {}) if isinstance(tweet.get("user", {}), dict) else {}
        entities = tweet.get("entities", {}) if isinstance(tweet.get("entities", {}), dict) else {}
        text = extract_tweet_text(tweet)
        created_at = parse_twitter_created_at(tweet.get("created_at"))
        time_delta_minutes = 0.0
        if created_at is not None and root_time is not None:
            time_delta_minutes = max((created_at - root_time).total_seconds() / 60.0, 0.0)

        urls = entities.get("urls", []) if isinstance(entities.get("urls", []), list) else []
        hashtags = entities.get("hashtags", []) if isinstance(entities.get("hashtags", []), list) else []
        mentions = entities.get("user_mentions", []) if isinstance(entities.get("user_mentions", []), list) else []

        features.append(
            [
                1.0 if node_id == root_id else 0.0,
                safe_log1p(depths.get(node_id, 0)),
                safe_log1p(out_degree.get(node_id, 0)),
                safe_log1p(len(text)),
                safe_log1p(tweet.get("retweet_count", 0)),
                safe_log1p(tweet.get("favorite_count", 0)),
                safe_log1p(user.get("followers_count", 0)),
                safe_log1p(user.get("friends_count", 0)),
                safe_log1p(user.get("statuses_count", 0)),
                safe_log1p(user.get("favourites_count", 0)),
                safe_log1p(user.get("listed_count", 0)),
                1.0 if user.get("verified") else 0.0,
                1.0 if user.get("default_profile") else 0.0,
                1.0 if urls else 0.0,
                safe_log1p(len(urls)),
                safe_log1p(len(hashtags)),
                safe_log1p(len(mentions)),
                1.0 if tweet.get("in_reply_to_status_id") is not None else 0.0,
                safe_log1p(time_delta_minutes),
                0.0 if tweet else 1.0,
            ]
        )

    return np.asarray(features, dtype=np.float32)


def load_pheme_raw_records(dataset_root: Union[str, Path]) -> List[PhemeRawRecord]:
    threads_root = resolve_pheme_raw_threads_root(dataset_root)
    records: List[PhemeRawRecord] = []

    for event_dir in sorted(threads_root.glob("*-all-rnr-threads")):
        if event_dir.name.startswith("._"):
            continue
        event_name = event_dir.name.replace("-all-rnr-threads", "")
        for label_name, label in (("non-rumours", 0), ("rumours", 1)):
            label_dir = event_dir / label_name
            if not label_dir.exists():
                continue
            for sample_dir in sorted(path for path in label_dir.iterdir() if path.is_dir() and not path.name.startswith("._")):
                structure_path = sample_dir / "structure.json"
                if not structure_path.exists():
                    continue
                try:
                    structure = read_json_file(structure_path)
                except (json.JSONDecodeError, OSError):
                    continue

                node_ids, edges, depths = flatten_pheme_structure(structure)
                if not node_ids:
                    node_ids = [sample_dir.name]
                root_id = node_ids[0]
                tweets, source_text, root_time = load_pheme_tweet_jsons(sample_dir, sample_dir.name)
                if not source_text:
                    source_text = extract_tweet_text(tweets.get(root_id, {}))
                if not source_text:
                    continue

                node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
                edge_pairs = [
                    (node_to_index[src], node_to_index[dst])
                    for src, dst in edges
                    if src in node_to_index and dst in node_to_index
                ]
                if edge_pairs:
                    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
                else:
                    edge_index = np.empty((2, 0), dtype=np.int64)

                node_features = build_pheme_raw_node_features(
                    node_ids=node_ids,
                    edges=edges,
                    depths=depths,
                    tweets=tweets,
                    root_id=root_id,
                    root_time=root_time,
                )
                records.append(
                    PhemeRawRecord(
                        sample_id=f"{event_name}:{sample_dir.name}",
                        label=label,
                        source_text=source_text,
                        event_name=event_name,
                        node_features=node_features,
                        edge_index=edge_index,
                        root_index=node_to_index[root_id],
                    )
                )

    if not records:
        raise RuntimeError(f"No PhemeRaw records were loaded from: {threads_root}")
    return records


class PhemeRawConversationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PhemeRawRecord],
        max_graph_nodes: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.max_graph_nodes = max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None
        self.sample_ids = [record.sample_id for record in self.records]
        self.labels = {record.sample_id: record.label for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def get_label(self, index: int) -> int:
        return self.records[index].label

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Dict[str, torch.Tensor]]:
        record = self.records[index]
        node_features_np, edge_index_np, root_index = RumorGraphDataset._truncate_graph(
            node_features=record.node_features,
            edge_index=record.edge_index,
            root_index=record.root_index,
            max_graph_nodes=self.max_graph_nodes,
        )
        edge_type_td = RumorGraphDataset._build_td_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )
        edge_type_bu = RumorGraphDataset._build_bu_edge_types(
            edge_index_np,
            root_index,
            node_features_np.shape[0],
        )

        edge_index_td = torch.tensor(edge_index_np, dtype=torch.long)
        return {
            "id": record.sample_id,
            "label": torch.tensor(record.label, dtype=torch.long),
            "graph": {
                "node_features": torch.tensor(node_features_np, dtype=torch.float32),
                "edge_index_td": edge_index_td,
                "edge_index_bu": edge_index_td.flip(0),
                "edge_type_td": torch.tensor(edge_type_td, dtype=torch.long),
                "edge_type_bu": torch.tensor(edge_type_bu, dtype=torch.long),
                "root_index": torch.tensor(root_index, dtype=torch.long),
            },
        }


class RumorGraphDataset(Dataset):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        dataset_name: str,
        max_graph_nodes: Optional[int] = None,
        early_cutoff_minutes: Optional[float] = None,
        node_token_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset_root = resolve_dataset_root(dataset_root)
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.max_graph_nodes = max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None
        if early_cutoff_minutes is not None and early_cutoff_minutes < 0.0:
            raise ValueError("early_cutoff_minutes must be non-negative.")
        self.early_cutoff_minutes = (
            float(early_cutoff_minutes) if early_cutoff_minutes is not None else None
        )

        self.dataset_dir = self.dataset_root / self.dataset_name
        self.graph_dir = self.dataset_root / f"{self.dataset_name}graph"
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.graph_dir.exists():
            raise FileNotFoundError(
                f"Graph directory not found: {self.graph_dir}. "
                "This model requires graph npz files; the current workspace does not provide them for this dataset."
            )

        self.labels = parse_label_file(self.dataset_name, self.dataset_dir)
        graph_ids = sorted(path.stem for path in self.graph_dir.glob("*.npz"))

        self.sample_ids = [sample_id for sample_id in graph_ids if sample_id in self.labels]
        if not self.sample_ids:
            raise RuntimeError(f"No matched samples were found for dataset '{dataset_name}'.")

        self.node_token_dir: Optional[Path] = None
        self.has_node_tokens = node_token_root is not None
        self.node_token_length: Optional[int] = None
        if node_token_root is not None:
            token_root = Path(node_token_root).expanduser().resolve()
            candidates = [token_root / f"{self.dataset_name}tokens", token_root]
            self.node_token_dir = next(
                (candidate for candidate in candidates if candidate.is_dir()),
                None,
            )
            if self.node_token_dir is None:
                raise FileNotFoundError(
                    f"Node token directory not found under: {token_root}. "
                    f"Expected {self.dataset_name}tokens/*.npz."
                )
            missing_tokens = [
                sample_id
                for sample_id in self.sample_ids
                if not (self.node_token_dir / f"{sample_id}.npz").is_file()
            ]
            if missing_tokens:
                preview = ", ".join(missing_tokens[:5])
                raise FileNotFoundError(
                    f"Node token sidecar is missing {len(missing_tokens)} graph(s), including: {preview}"
                )

        first_graph = np.load(self.graph_dir / f"{self.sample_ids[0]}.npz", allow_pickle=True)
        self.has_time_metadata = "time_delta_minutes" in first_graph.files
        if self.early_cutoff_minutes is not None and not self.has_time_metadata:
            raise ValueError(
                f"Early detection at {self.early_cutoff_minutes:g} minutes requires "
                "'time_delta_minutes' in every graph npz. Add PHEME time metadata before training."
            )
        self.graph_feature_key = "bert_x" if "bert_x" in first_graph.files else "x"
        self.semantic_feature_key = "semantic_x" if "semantic_x" in first_graph.files else None
        self.transpose_graph_features = self._infer_graph_feature_orientation()
        first_root_index = self._extract_root_index(first_graph["rootindex"])
        first_edge_index = self._normalize_edge_index(first_graph["edgeindex"])
        first_node_features = self._normalize_node_features(
            first_graph[self.graph_feature_key],
            first_edge_index,
            first_root_index,
        )
        self.graph_input_dim = int(first_node_features.shape[-1])
        if self.semantic_feature_key is not None:
            first_semantic_features = self._normalize_node_features(
                first_graph[self.semantic_feature_key],
                first_edge_index,
                first_root_index,
            )
            if first_semantic_features.shape != first_node_features.shape:
                raise ValueError(
                    "semantic_x must align with bert_x and use the same feature dimension: "
                    f"{first_semantic_features.shape} vs {first_node_features.shape}."
                )
        if self.node_token_dir is not None:
            with np.load(
                self.node_token_dir / f"{self.sample_ids[0]}.npz",
                allow_pickle=False,
            ) as first_tokens:
                input_ids = np.asarray(first_tokens["input_ids"])
                attention_mask = np.asarray(first_tokens["attention_mask"])
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Node token sidecars require matching rank-2 input_ids and attention_mask arrays."
                )
            if input_ids.shape[0] != first_node_features.shape[0]:
                raise ValueError(
                    "Node token sidecar and graph feature node counts do not match for "
                    f"{self.sample_ids[0]}: {input_ids.shape[0]} vs {first_node_features.shape[0]}."
                )
            self.node_token_length = int(input_ids.shape[1])

        self.metadata = DatasetMetadata(
            graph_input_dim=self.graph_input_dim,
            num_classes=infer_num_classes_from_labels(
                {sample_id: self.labels[sample_id] for sample_id in self.sample_ids}
            ),
            label_to_index=TWITTER_LABEL_TO_INDEX if self.dataset_name in {"Twitter15", "Twitter16"} else None,
            graph_feature_key=self.graph_feature_key,
            graph_feature_transposed=self.transpose_graph_features,
            max_graph_nodes=self.max_graph_nodes,
            num_edge_relations=NUM_EDGE_RELATIONS,
            early_cutoff_minutes=self.early_cutoff_minutes,
            has_time_metadata=self.has_time_metadata,
            has_node_tokens=self.has_node_tokens,
            node_token_length=self.node_token_length,
            has_semantic_features=self.semantic_feature_key is not None,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get_label(self, index: int) -> int:
        return self.labels[self.sample_ids[index]]

    @staticmethod
    def _extract_root_index(root_index_value: np.ndarray) -> int:
        if np.asarray(root_index_value).shape == ():
            return int(root_index_value)
        return int(np.asarray(root_index_value).reshape(-1)[0])

    @staticmethod
    def _normalize_edge_index(edge_index: np.ndarray) -> np.ndarray:
        if edge_index.ndim != 2:
            raise ValueError(f"edgeindex must be rank-2, got shape {edge_index.shape}.")
        if edge_index.shape[0] == 2:
            return edge_index
        if edge_index.shape[1] == 2:
            return edge_index.T
        raise ValueError(f"Unsupported edgeindex shape: {edge_index.shape}")

    @classmethod
    def _should_transpose_graph_features(
        cls,
        feature_matrix: np.ndarray,
        edge_index: np.ndarray,
        root_index: int,
    ) -> Optional[bool]:
        if feature_matrix.ndim != 2:
            raise ValueError(f"node feature matrix must be rank-2, got shape {feature_matrix.shape}.")

        max_node_index = max(root_index, int(edge_index.max()) if edge_index.size else root_index)
        rows_valid = feature_matrix.shape[0] > max_node_index
        cols_valid = feature_matrix.shape[1] > max_node_index

        if cols_valid and not rows_valid:
            return True
        if rows_valid and not cols_valid:
            return False
        return None

    def _infer_graph_feature_orientation(self) -> bool:
        if self.graph_feature_key == "bert_x":
            return False

        max_checks = min(len(self.sample_ids), 64)
        fallback_shape = None

        for sample_id in self.sample_ids[:max_checks]:
            graph_npz = np.load(self.graph_dir / f"{sample_id}.npz", allow_pickle=True)
            feature_matrix = np.asarray(graph_npz[self.graph_feature_key])
            edge_index = self._normalize_edge_index(graph_npz["edgeindex"])
            root_index = self._extract_root_index(graph_npz["rootindex"])
            decision = self._should_transpose_graph_features(feature_matrix, edge_index, root_index)
            if decision is not None:
                return decision
            if fallback_shape is None:
                fallback_shape = feature_matrix.shape

        if fallback_shape is None:
            return False

        rows, cols = fallback_shape
        # Some graph dumps store features as [feature_dim, num_nodes], e.g. Weibo's 3 x N matrices.
        return rows <= 8 and cols > rows * 4

    def _normalize_node_features(
        self,
        feature_matrix: np.ndarray,
        edge_index: np.ndarray,
        root_index: int,
    ) -> np.ndarray:
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix.ndim != 2:
            raise ValueError(f"node feature matrix must be rank-2, got shape {feature_matrix.shape}.")

        transpose_features = self._should_transpose_graph_features(feature_matrix, edge_index, root_index)
        if transpose_features is None:
            transpose_features = self.transpose_graph_features

        if transpose_features:
            feature_matrix = feature_matrix.T
        return feature_matrix

    @staticmethod
    def _compute_out_degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        if edge_index.size == 0:
            return np.zeros(num_nodes, dtype=np.int64)
        return np.bincount(edge_index[0].astype(np.int64), minlength=num_nodes).astype(np.int64, copy=False)

    @classmethod
    def _build_td_edge_types(cls, edge_index: np.ndarray, root_index: int, num_nodes: int) -> np.ndarray:
        if edge_index.size == 0:
            return np.zeros((0,), dtype=np.int64)

        out_degree = cls._compute_out_degree(edge_index, num_nodes)
        is_leaf = out_degree == 0
        src_nodes = edge_index[0].astype(np.int64)
        dst_nodes = edge_index[1].astype(np.int64)

        edge_types = np.full(edge_index.shape[1], REL_TD_INTERNAL, dtype=np.int64)
        edge_types[is_leaf[dst_nodes]] = REL_TD_TERMINAL
        edge_types[src_nodes == int(root_index)] = REL_TD_ROOT
        return edge_types

    @classmethod
    def _build_bu_edge_types(cls, edge_index_td: np.ndarray, root_index: int, num_nodes: int) -> np.ndarray:
        if edge_index_td.size == 0:
            return np.zeros((0,), dtype=np.int64)

        out_degree = cls._compute_out_degree(edge_index_td, num_nodes)
        is_leaf = out_degree == 0
        parent_nodes = edge_index_td[0].astype(np.int64)
        child_nodes = edge_index_td[1].astype(np.int64)

        edge_types = np.full(edge_index_td.shape[1], REL_BU_INTERNAL, dtype=np.int64)
        edge_types[is_leaf[child_nodes]] = REL_BU_FROM_LEAF
        edge_types[parent_nodes == int(root_index)] = REL_BU_TO_ROOT
        return edge_types

    @staticmethod
    def _select_truncated_nodes(
        edge_index: np.ndarray,
        root_index: int,
        num_nodes: int,
        max_graph_nodes: int,
    ) -> np.ndarray:
        neighbors: Dict[int, List[int]] = defaultdict(list)
        for src, dst in edge_index.T:
            src_idx = int(src)
            dst_idx = int(dst)
            if 0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes:
                neighbors[src_idx].append(dst_idx)
                neighbors[dst_idx].append(src_idx)

        selected: List[int] = []
        visited = {root_index}
        queue = deque([root_index])

        while queue and len(selected) < max_graph_nodes:
            node_index = queue.popleft()
            selected.append(node_index)
            for neighbor_index in neighbors.get(node_index, []):
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append(neighbor_index)

        if len(selected) < max_graph_nodes:
            for node_index in range(num_nodes):
                if node_index in visited:
                    continue
                visited.add(node_index)
                selected.append(node_index)
                if len(selected) >= max_graph_nodes:
                    break

        return np.asarray(selected, dtype=np.int64)

    @classmethod
    def _truncate_graph(
        cls,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        root_index: int,
        max_graph_nodes: Optional[int],
        return_node_indices: bool = False,
    ) -> tuple:
        num_nodes = int(node_features.shape[0])
        if max_graph_nodes is None or num_nodes <= max_graph_nodes:
            if return_node_indices:
                return node_features, edge_index, root_index, np.arange(num_nodes, dtype=np.int64)
            return node_features, edge_index, root_index

        keep_nodes = cls._select_truncated_nodes(edge_index, root_index, num_nodes, max_graph_nodes)
        node_mapping = np.full(num_nodes, -1, dtype=np.int64)
        node_mapping[keep_nodes] = np.arange(len(keep_nodes), dtype=np.int64)

        truncated_features = node_features[keep_nodes]
        if edge_index.size == 0:
            truncated_edge_index = edge_index[:, :0].astype(np.int64, copy=False)
        else:
            edge_mask = (node_mapping[edge_index[0]] >= 0) & (node_mapping[edge_index[1]] >= 0)
            truncated_edge_index = node_mapping[edge_index[:, edge_mask]]
        truncated_root_index = int(node_mapping[root_index])
        if return_node_indices:
            return truncated_features, truncated_edge_index, truncated_root_index, keep_nodes
        return truncated_features, truncated_edge_index, truncated_root_index

    @classmethod
    def _truncate_graph_by_time(
        cls,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        root_index: int,
        time_delta_minutes: np.ndarray,
        cutoff_minutes: float,
        return_node_indices: bool = False,
    ) -> tuple:
        num_nodes = int(node_features.shape[0])
        node_times = np.asarray(time_delta_minutes, dtype=np.float32).reshape(-1)
        if node_times.size != num_nodes:
            raise ValueError(
                "time_delta_minutes must contain one value per graph node: "
                f"got {node_times.size} times for {num_nodes} nodes."
            )
        if not 0 <= root_index < num_nodes:
            raise ValueError(f"rootindex {root_index} is outside a graph with {num_nodes} nodes.")

        # The zero-minute point represents source-post-only detection.
        eligible = np.zeros(num_nodes, dtype=bool)
        eligible[root_index] = True
        if cutoff_minutes > 0.0:
            eligible |= (
                np.isfinite(node_times)
                & (node_times >= 0.0)
                & (node_times <= float(cutoff_minutes) + 1e-6)
            )

        # Keep only eligible nodes reachable from the source through eligible ancestors.
        children: Dict[int, List[int]] = defaultdict(list)
        for src, dst in edge_index.T:
            src_idx = int(src)
            dst_idx = int(dst)
            if 0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes:
                children[src_idx].append(dst_idx)

        reachable = np.zeros(num_nodes, dtype=bool)
        reachable[root_index] = True
        queue = deque([root_index])
        while queue:
            parent = queue.popleft()
            for child in children.get(parent, []):
                if reachable[child] or not eligible[child]:
                    continue
                reachable[child] = True
                queue.append(child)

        keep_nodes = np.flatnonzero(reachable).astype(np.int64, copy=False)
        node_mapping = np.full(num_nodes, -1, dtype=np.int64)
        node_mapping[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

        truncated_features = node_features[keep_nodes]
        if edge_index.size == 0:
            truncated_edge_index = edge_index[:, :0].astype(np.int64, copy=False)
        else:
            edge_mask = (node_mapping[edge_index[0]] >= 0) & (node_mapping[edge_index[1]] >= 0)
            truncated_edge_index = node_mapping[edge_index[:, edge_mask]]
        truncated_root_index = int(node_mapping[root_index])
        if return_node_indices:
            return truncated_features, truncated_edge_index, truncated_root_index, keep_nodes
        return truncated_features, truncated_edge_index, truncated_root_index

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Dict[str, torch.Tensor]]:
        sample_id = self.sample_ids[index]
        graph_path = self.graph_dir / f"{sample_id}.npz"
        graph_npz = np.load(graph_path, allow_pickle=True)

        root_index = self._extract_root_index(graph_npz["rootindex"])
        edge_index_td_np = self._normalize_edge_index(graph_npz["edgeindex"]).astype(np.int64, copy=False)
        node_features_np = self._normalize_node_features(
            graph_npz[self.graph_feature_key],
            edge_index_td_np,
            root_index,
        )
        semantic_features_np: Optional[np.ndarray] = None
        if self.semantic_feature_key is not None:
            if self.semantic_feature_key not in graph_npz.files:
                raise ValueError(
                    f"Graph {graph_path} is missing {self.semantic_feature_key}, "
                    "but the dataset was detected as dual-view."
                )
            semantic_features_np = self._normalize_node_features(
                graph_npz[self.semantic_feature_key],
                edge_index_td_np,
                root_index,
            )
            if semantic_features_np.shape != node_features_np.shape:
                raise ValueError(
                    f"Graph {graph_path} has misaligned dual-view features: "
                    f"structure={node_features_np.shape}, semantic={semantic_features_np.shape}."
                )
        original_node_indices = np.arange(node_features_np.shape[0], dtype=np.int64)
        if "time_delta_minutes" in graph_npz.files:
            original_node_times = np.asarray(
                graph_npz["time_delta_minutes"], dtype=np.float32
            ).reshape(-1)
            if original_node_times.size != node_features_np.shape[0]:
                raise ValueError(
                    f"Graph {graph_path} has {original_node_times.size} node times for "
                    f"{node_features_np.shape[0]} nodes."
                )
            original_node_times = original_node_times.copy()
        else:
            original_node_times = np.zeros(node_features_np.shape[0], dtype=np.float32)
        # The source post defines event time zero even when a legacy sidecar contains
        # a noisy or missing root timestamp.
        original_node_times[root_index] = 0.0
        node_input_ids_np: Optional[np.ndarray] = None
        node_attention_mask_np: Optional[np.ndarray] = None
        if self.node_token_dir is not None:
            token_path = self.node_token_dir / f"{sample_id}.npz"
            with np.load(token_path, allow_pickle=False) as token_npz:
                node_input_ids_np = np.asarray(token_npz["input_ids"])
                node_attention_mask_np = np.asarray(token_npz["attention_mask"])
            if (
                node_input_ids_np.ndim != 2
                or node_attention_mask_np.shape != node_input_ids_np.shape
                or node_input_ids_np.shape[0] != node_features_np.shape[0]
            ):
                raise ValueError(
                    f"Node token sidecar {token_path} does not align with graph nodes: "
                    f"ids={node_input_ids_np.shape}, mask={node_attention_mask_np.shape}, "
                    f"nodes={node_features_np.shape[0]}."
                )
        # Always retain the original-node mapping so timestamps undergo exactly the
        # same early-time and max-node truncation as every other node-level tensor.
        track_node_indices = True
        if self.early_cutoff_minutes is not None:
            if "time_delta_minutes" not in graph_npz.files:
                raise ValueError(f"Graph {graph_path} is missing time_delta_minutes.")
            if track_node_indices:
                (
                    node_features_np,
                    edge_index_td_np,
                    root_index,
                    kept_indices,
                ) = self._truncate_graph_by_time(
                    node_features=node_features_np,
                    edge_index=edge_index_td_np,
                    root_index=root_index,
                    time_delta_minutes=graph_npz["time_delta_minutes"],
                    cutoff_minutes=self.early_cutoff_minutes,
                    return_node_indices=True,
                )
                original_node_indices = original_node_indices[kept_indices]
            else:
                node_features_np, edge_index_td_np, root_index = self._truncate_graph_by_time(
                    node_features=node_features_np,
                    edge_index=edge_index_td_np,
                    root_index=root_index,
                    time_delta_minutes=graph_npz["time_delta_minutes"],
                    cutoff_minutes=self.early_cutoff_minutes,
                )
        if track_node_indices:
            (
                node_features_np,
                edge_index_td_np,
                root_index,
                kept_indices,
            ) = self._truncate_graph(
                node_features=node_features_np,
                edge_index=edge_index_td_np,
                root_index=root_index,
                max_graph_nodes=self.max_graph_nodes,
                return_node_indices=True,
            )
            original_node_indices = original_node_indices[kept_indices]
        else:
            node_features_np, edge_index_td_np, root_index = self._truncate_graph(
                node_features=node_features_np,
                edge_index=edge_index_td_np,
                root_index=root_index,
                max_graph_nodes=self.max_graph_nodes,
            )

        node_features = torch.tensor(node_features_np, dtype=torch.float32)
        semantic_node_features = (
            None
            if semantic_features_np is None
            else torch.tensor(
                semantic_features_np[original_node_indices],
                dtype=torch.float32,
            )
        )
        edge_index_td = torch.tensor(edge_index_td_np, dtype=torch.long)
        edge_index_bu = edge_index_td.flip(0)
        edge_type_td = torch.tensor(
            self._build_td_edge_types(edge_index_td_np, root_index, node_features_np.shape[0]),
            dtype=torch.long,
        )
        edge_type_bu = torch.tensor(
            self._build_bu_edge_types(edge_index_td_np, root_index, node_features_np.shape[0]),
            dtype=torch.long,
        )

        label = torch.tensor(self.labels[sample_id], dtype=torch.long)

        graph = {
            "node_features": node_features,
            "node_time": torch.tensor(
                original_node_times[original_node_indices], dtype=torch.float32
            ),
            "edge_index_td": edge_index_td,
            "edge_index_bu": edge_index_bu,
            "edge_type_td": edge_type_td,
            "edge_type_bu": edge_type_bu,
            "root_index": torch.tensor(root_index, dtype=torch.long),
        }
        if semantic_node_features is not None:
            graph["semantic_node_features"] = semantic_node_features
        if node_input_ids_np is not None and node_attention_mask_np is not None:
            graph["node_input_ids"] = torch.tensor(
                node_input_ids_np[original_node_indices],
                dtype=torch.long,
            )
            graph["node_attention_mask"] = torch.tensor(
                node_attention_mask_np[original_node_indices],
                dtype=torch.long,
            )

        return {
            "id": sample_id,
            "label": label,
            "graph": graph,
        }


def collate_graph_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    graphs = [item["graph"] for item in batch]
    node_features = []
    node_times = []
    edge_indices_td = []
    edge_indices_bu = []
    edge_types_td = []
    edge_types_bu = []
    graph_batch = []
    root_indices = []
    graph_ptr = [0]
    node_offset = 0
    semantic_presence = ["semantic_node_features" in graph for graph in graphs]
    if any(semantic_presence) and not all(semantic_presence):
        raise ValueError("A graph batch cannot mix single-view and dual-view samples.")
    has_semantic_features = bool(semantic_presence and semantic_presence[0])
    semantic_node_features = []
    token_presence = ["node_input_ids" in graph for graph in graphs]
    if any(token_presence) and not all(token_presence):
        raise ValueError("A graph batch cannot mix samples with and without node tokens.")
    has_node_tokens = bool(token_presence and token_presence[0])
    node_input_ids = []
    node_attention_masks = []

    for graph_index, graph in enumerate(graphs):
        features = graph["node_features"]
        num_nodes = int(features.size(0))
        node_features.append(features)
        time = graph.get("node_time")
        if time is None:
            time = torch.zeros(num_nodes, dtype=torch.float32)
        time = time.reshape(-1).to(dtype=torch.float32)
        if time.numel() != num_nodes:
            raise ValueError("Node times must align with node feature count.")
        node_times.append(time)
        if has_semantic_features:
            semantic_features = graph["semantic_node_features"]
            if semantic_features.shape != features.shape:
                raise ValueError(
                    "Semantic node features must align with structure node features."
                )
            semantic_node_features.append(semantic_features)
        graph_batch.append(torch.full((num_nodes,), graph_index, dtype=torch.long))
        if has_node_tokens:
            input_ids = graph["node_input_ids"].long()
            attention_mask = graph["node_attention_mask"].long()
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError("Each graph requires aligned rank-2 node token tensors.")
            if input_ids.size(0) != num_nodes:
                raise ValueError("Node token count must match node feature count.")
            node_input_ids.append(input_ids)
            node_attention_masks.append(attention_mask)

        edge_index_td = graph["edge_index_td"].long()
        edge_index_bu = graph["edge_index_bu"].long()
        if edge_index_td.numel() > 0:
            edge_index_td = edge_index_td + node_offset
        if edge_index_bu.numel() > 0:
            edge_index_bu = edge_index_bu + node_offset
        edge_indices_td.append(edge_index_td)
        edge_indices_bu.append(edge_index_bu)
        edge_types_td.append(graph["edge_type_td"].long())
        edge_types_bu.append(graph["edge_type_bu"].long())

        root_index = int(graph["root_index"].item())
        root_indices.append(torch.tensor(root_index + node_offset, dtype=torch.long))
        node_offset += num_nodes
        graph_ptr.append(node_offset)

    batched_graph = {
        "node_features": torch.cat(node_features, dim=0),
        "node_time": torch.cat(node_times, dim=0),
        "edge_index_td": torch.cat(edge_indices_td, dim=1),
        "edge_index_bu": torch.cat(edge_indices_bu, dim=1),
        "edge_type_td": torch.cat(edge_types_td, dim=0),
        "edge_type_bu": torch.cat(edge_types_bu, dim=0),
        "root_indices": torch.stack(root_indices, dim=0),
        "graph_batch": torch.cat(graph_batch, dim=0),
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
        "num_graphs": len(graphs),
    }
    if has_node_tokens:
        batched_graph["node_input_ids"] = torch.cat(node_input_ids, dim=0)
        batched_graph["node_attention_mask"] = torch.cat(node_attention_masks, dim=0)
    if has_semantic_features:
        batched_graph["semantic_node_features"] = torch.cat(
            semantic_node_features,
            dim=0,
        )

    return {
        "ids": [item["id"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "graphs": graphs,
        "batched_graph": batched_graph,
    }

def build_pheme_raw_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int,
    max_seq_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    records = load_pheme_raw_records(dataset_root)
    labels = {record.sample_id: record.label for record in records}
    sample_ids = [record.sample_id for record in records]
    train_ids, val_ids, test_ids = stratified_split(
        sample_ids=sample_ids,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    records_by_id = {record.sample_id: record for record in records}
    train_records = [records_by_id[sample_id] for sample_id in train_ids]
    val_records = [records_by_id[sample_id] for sample_id in val_ids]
    test_records = [records_by_id[sample_id] for sample_id in test_ids]

    train_dataset = PhemeRawConversationDataset(
        records=train_records,
        max_graph_nodes=max_graph_nodes,
    )
    val_dataset = PhemeRawConversationDataset(
        records=val_records,
        max_graph_nodes=max_graph_nodes,
    )
    test_dataset = PhemeRawConversationDataset(
        records=test_records,
        max_graph_nodes=max_graph_nodes,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    event_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        event_counts[record.event_name] += 1

    metadata = DatasetMetadata(
        graph_input_dim=len(PHEME_RAW_GRAPH_FEATURE_NAMES),
        num_classes=infer_num_classes_from_labels(labels),
        graph_feature_key="pheme_raw_tweet_metadata",
        graph_feature_transposed=False,
        max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=False,
        uses_synthetic_graph=False,
        category_feature_size=len(PHEME_RAW_GRAPH_FEATURE_NAMES),
        category_to_index=dict(sorted(event_counts.items())),
    )
    return train_loader, val_loader, test_loader, metadata


def build_twitter_acl2017_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int,
    max_seq_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
    subset_name: str = "twitter15",
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    records = load_twitter_acl2017_records(dataset_root, subset_name=subset_name)
    labels = {record.sample_id: record.label for record in records}
    sample_ids = [record.sample_id for record in records]
    train_ids, val_ids, test_ids = stratified_split(
        sample_ids=sample_ids,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    records_by_id = {record.sample_id: record for record in records}
    train_records = [records_by_id[sample_id] for sample_id in train_ids]
    val_records = [records_by_id[sample_id] for sample_id in val_ids]
    test_records = [records_by_id[sample_id] for sample_id in test_ids]

    train_dataset = PhemeRawConversationDataset(
        records=train_records,
        max_graph_nodes=max_graph_nodes,
    )
    val_dataset = PhemeRawConversationDataset(
        records=val_records,
        max_graph_nodes=max_graph_nodes,
    )
    test_dataset = PhemeRawConversationDataset(
        records=test_records,
        max_graph_nodes=max_graph_nodes,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    metadata = DatasetMetadata(
        graph_input_dim=len(TWITTER_ACL2017_GRAPH_FEATURE_NAMES),
        num_classes=infer_num_classes_from_labels(labels),
        label_to_index=TWITTER_LABEL_TO_INDEX,
        graph_feature_key="twitter_acl2017_tree_metadata",
        graph_feature_transposed=False,
        max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=False,
        uses_synthetic_graph=False,
        category_feature_size=len(TWITTER_ACL2017_GRAPH_FEATURE_NAMES),
        category_to_index={name: index for index, name in enumerate(TWITTER_ACL2017_GRAPH_FEATURE_NAMES)},
    )
    return train_loader, val_loader, test_loader, metadata


def build_twitter_icdm_dataloaders(
    dataset_root: Union[str, Path],
    subset_name: str,
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    records, split_counts = load_twitter_icdm_records(dataset_root, subset_name)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "dev"]
    test_records = [record for record in records if record.split == "test"]
    if not train_records or not val_records or not test_records:
        raise RuntimeError(f"{subset_name} ICDM split files are incomplete: {split_counts}")

    train_dataset = PhemeRawConversationDataset(
        records=train_records,
        max_graph_nodes=max_graph_nodes,
    )
    val_dataset = PhemeRawConversationDataset(
        records=val_records,
        max_graph_nodes=max_graph_nodes,
    )
    test_dataset = PhemeRawConversationDataset(
        records=test_records,
        max_graph_nodes=max_graph_nodes,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    missing_graph_count = sum(1 for record in records if record.node_features.shape[0] == 1)
    metadata = DatasetMetadata(
        graph_input_dim=len(TWITTER15_ICDM_GRAPH_FEATURE_NAMES),
        num_classes=infer_num_classes_from_labels({record.sample_id: record.label for record in records}),
        label_to_index=TWITTER_LABEL_TO_INDEX,
        graph_feature_key=f"{subset_name}_icdm_source_user_weight_graph",
        graph_feature_transposed=False,
        max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=True,
        uses_synthetic_graph=False,
        category_feature_size=len(TWITTER15_ICDM_GRAPH_FEATURE_NAMES),
        category_to_index={
            **{name: index for index, name in enumerate(TWITTER15_ICDM_GRAPH_FEATURE_NAMES)},
            "split_train_size": split_counts.get("train", 0),
            "split_dev_size": split_counts.get("dev", 0),
            "split_test_size": split_counts.get("test", 0),
            "root_only_graph_count": missing_graph_count,
        },
    )
    return train_loader, val_loader, test_loader, metadata


def build_twitter15_icdm_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    return build_twitter_icdm_dataloaders(
        dataset_root=dataset_root,
        subset_name="twitter15",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        max_graph_nodes=max_graph_nodes,
    )


def build_twitter16_icdm_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    return build_twitter_icdm_dataloaders(
        dataset_root=dataset_root,
        subset_name="twitter16",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        max_graph_nodes=max_graph_nodes,
    )


def build_twitter_sparse_tree_dataloaders(
    dataset_root: Union[str, Path],
    dataset_name: str,
    batch_size: int,
    max_seq_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
    graph_feature_dim: int = 5000,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    records = load_twitter_sparse_tree_records(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
    )
    labels = {record.sample_id: record.label for record in records}
    sample_ids = [record.sample_id for record in records]
    train_ids, val_ids, test_ids = stratified_split(
        sample_ids=sample_ids,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    records_by_id = {record.sample_id: record for record in records}
    train_records = [records_by_id[sample_id] for sample_id in train_ids]
    val_records = [records_by_id[sample_id] for sample_id in val_ids]
    test_records = [records_by_id[sample_id] for sample_id in test_ids]

    train_dataset = TwitterSparseTreeDataset(
        records=train_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )
    val_dataset = TwitterSparseTreeDataset(
        records=val_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )
    test_dataset = TwitterSparseTreeDataset(
        records=test_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    metadata = DatasetMetadata(
        graph_input_dim=graph_feature_dim,
        num_classes=infer_num_classes_from_labels(labels),
        label_to_index=TWITTER_LABEL_TO_INDEX,
        graph_feature_key="td_rvnn_sparse_index_counts",
        graph_feature_transposed=False,
        max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=False,
        uses_synthetic_graph=False,
        category_feature_size=None,
        category_to_index={
            "records_loaded_from_td_file": len(records),
            "graph_feature_dim": graph_feature_dim,
            "format": "root-id,parent-index,current-index,parent-number,text-length,index-counts",
        },
    )
    return train_loader, val_loader, test_loader, metadata


def build_ragcl_tfidf_dataloaders(
    dataset_root: Union[str, Path],
    dataset_name: str,
    batch_size: int,
    max_seq_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
    graph_feature_dim: int = 5000,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    records = load_ragcl_tfidf_records(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_seq_len=max_seq_len,
    )
    labels = {record.sample_id: record.label for record in records}
    sample_ids = [record.sample_id for record in records]
    train_ids, val_ids, test_ids = stratified_split(
        sample_ids=sample_ids,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    records_by_id = {record.sample_id: record for record in records}
    train_records = [records_by_id[sample_id] for sample_id in train_ids]
    val_records = [records_by_id[sample_id] for sample_id in val_ids]
    test_records = [records_by_id[sample_id] for sample_id in test_ids]

    train_dataset = RagclTFIDFDataset(
        records=train_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )
    val_dataset = RagclTFIDFDataset(
        records=val_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )
    test_dataset = RagclTFIDFDataset(
        records=test_records,
        max_seq_len=max_seq_len,
        graph_feature_dim=graph_feature_dim,
        max_graph_nodes=max_graph_nodes,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    metadata = DatasetMetadata(
        graph_input_dim=graph_feature_dim,
        num_classes=infer_num_classes_from_labels(labels),
        label_to_index=TWITTER_LABEL_TO_INDEX,
        graph_feature_key="ragcl_tfidf_5000_index_counts",
        graph_feature_transposed=False,
        max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=False,
        uses_synthetic_graph=False,
        category_feature_size=None,
        category_to_index={
            "records_loaded_from_ragcl_source": len(records),
            "graph_feature_dim": graph_feature_dim,
            "format": "RAGCL JSON source/comment TF-IDF index-counts",
        },
    )
    return train_loader, val_loader, test_loader, metadata


def build_weibo21_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    dataset_dir = resolve_weibo21_dir(dataset_root)
    train_records = load_weibo21_records(dataset_dir, "train")
    val_records = load_weibo21_records(dataset_dir, "val")
    test_records = load_weibo21_records(dataset_dir, "test")
    all_records = train_records + val_records + test_records

    category_to_index = build_weibo21_category_vocab(all_records)

    train_dataset = Weibo21GraphDataset(
        records=train_records,
        category_to_index=category_to_index,
    )
    val_dataset = Weibo21GraphDataset(
        records=val_records,
        category_to_index=category_to_index,
    )
    test_dataset = Weibo21GraphDataset(
        records=test_records,
        category_to_index=category_to_index,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    metadata = DatasetMetadata(
        graph_input_dim=len(category_to_index),
        num_classes=infer_num_classes_from_labels({record.sample_id: record.label for record in all_records}),
        graph_feature_key="weibo21_category_one_hot",
        graph_feature_transposed=False,
        max_graph_nodes=1,
        num_edge_relations=NUM_EDGE_RELATIONS,
        source_split_files=True,
        uses_synthetic_graph=True,
        category_feature_size=len(category_to_index),
        category_to_index=category_to_index,
    )
    return train_loader, val_loader, test_loader, metadata


def build_dataloaders(
    dataset_root: Union[str, Path],
    dataset_name: str,
    batch_size: int,
    max_seq_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
    early_cutoff_minutes: Optional[float] = None,
    node_token_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    dataset_name = normalize_dataset_name(dataset_name)
    if early_cutoff_minutes is not None and dataset_name not in {"Pheme", "Weibo"}:
        raise ValueError(
            "early_cutoff_minutes is currently supported only for the Pheme and Weibo datasets."
        )
    if node_token_root is not None and dataset_name != "Pheme":
        raise ValueError("node_token_root is currently supported only for the Pheme dataset.")
    if dataset_name == "PhemeRaw":
        return build_pheme_raw_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
        )

    if dataset_name in {"Twitter15TD", "Twitter16TD"}:
        return build_twitter_sparse_tree_dataloaders(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
        )

    if dataset_name == "Twitter16TFIDF":
        return build_ragcl_tfidf_dataloaders(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
        )

    if dataset_name == "Twitter15Raw":
        return build_twitter_acl2017_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
            subset_name="twitter15",
        )

    if dataset_name == "Twitter15ICDM":
        return build_twitter15_icdm_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
        )

    if dataset_name == "Twitter16ICDM":
        return build_twitter16_icdm_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_workers=num_workers,
            max_graph_nodes=max_graph_nodes,
        )

    if dataset_name == "Weibo21":
        return build_weibo21_dataloaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_workers=num_workers,
        )

    dataset = RumorGraphDataset(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_graph_nodes=max_graph_nodes,
        early_cutoff_minutes=early_cutoff_minutes,
        node_token_root=node_token_root,
    )

    train_ids, val_ids, test_ids = stratified_split(
        sample_ids=dataset.sample_ids,
        labels=dataset.labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset.sample_ids)}
    train_subset = Subset(dataset, [id_to_index[sample_id] for sample_id in train_ids])
    val_subset = Subset(dataset, [id_to_index[sample_id] for sample_id in val_ids])
    test_subset = Subset(dataset, [id_to_index[sample_id] for sample_id in test_ids])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )
    return train_loader, val_loader, test_loader, dataset.metadata


def build_kfold_dataloaders(
    dataset_root: Union[str, Path],
    dataset_name: str,
    batch_size: int,
    max_seq_len: int,
    n_folds: int,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    max_graph_nodes: Optional[int] = None,
    graph_feature_dim: int = 5000,
    early_cutoff_minutes: Optional[float] = None,
    cv_protocol: str = "standard",
    node_token_root: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[DataLoader, DataLoader, DataLoader]], DatasetMetadata]:
    """Build stratified K-fold (train, val, test) DataLoaders for cross-validation.

    Returns a list of one (train_loader, val_loader, test_loader) tuple per fold,
    plus shared dataset metadata. ``cv_protocol='bigcn'`` reproduces BiGCN's
    class-wise five-fold test split. Its validation loader is empty when
    ``val_ratio`` is zero; otherwise it is sampled from the outer training pool.
    Supports the RAGCL TF-IDF datasets (Twitter15/16-tfidf) and the generic
    propagation-graph datasets.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    if cv_protocol not in {"standard", "bigcn"}:
        raise ValueError("cv_protocol must be 'standard' or 'bigcn'.")
    if early_cutoff_minutes is not None and dataset_name not in {"Pheme", "Weibo"}:
        raise ValueError(
            "early_cutoff_minutes is currently supported only for the Pheme and Weibo datasets."
        )
    if node_token_root is not None and dataset_name != "Pheme":
        raise ValueError("node_token_root is currently supported only for the Pheme dataset.")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_batch,
    }

    if dataset_name in {"Twitter15TFIDF", "Twitter16TFIDF"}:
        records = load_ragcl_tfidf_records(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            max_seq_len=max_seq_len,
        )
        labels = {record.sample_id: record.label for record in records}
        sample_ids = [record.sample_id for record in records]
        records_by_id = {record.sample_id: record for record in records}
        if cv_protocol == "bigcn":
            folds = stratified_bigcn_kfold(
                sample_ids=sample_ids,
                labels=labels,
                n_folds=n_folds,
                seed=seed,
                val_ratio=val_ratio,
            )
        else:
            folds = stratified_kfold(
                sample_ids=sample_ids,
                labels=labels,
                n_folds=n_folds,
                val_ratio=val_ratio,
                seed=seed,
            )

        def make_tfidf_loader(ids: List[str], shuffle: bool) -> DataLoader:
            fold_dataset = RagclTFIDFDataset(
                records=[records_by_id[sample_id] for sample_id in ids],
                max_seq_len=max_seq_len,
                graph_feature_dim=graph_feature_dim,
                max_graph_nodes=max_graph_nodes,
            )
            return DataLoader(fold_dataset, shuffle=shuffle, **loader_kwargs)

        fold_loaders = [
            (
                make_tfidf_loader(train_ids, True),
                make_tfidf_loader(val_ids, False),
                make_tfidf_loader(test_ids, False),
            )
            for train_ids, val_ids, test_ids in folds
        ]
        metadata = DatasetMetadata(
            graph_input_dim=graph_feature_dim,
            num_classes=infer_num_classes_from_labels(labels),
            label_to_index=TWITTER_LABEL_TO_INDEX,
            graph_feature_key="ragcl_tfidf_5000_index_counts",
            graph_feature_transposed=False,
            max_graph_nodes=max_graph_nodes if max_graph_nodes and max_graph_nodes > 0 else None,
            num_edge_relations=NUM_EDGE_RELATIONS,
            source_split_files=False,
            uses_synthetic_graph=False,
            category_feature_size=None,
            category_to_index={
                "records_loaded_from_ragcl_source": len(records),
                "graph_feature_dim": graph_feature_dim,
                "format": "RAGCL JSON source/comment TF-IDF index-counts",
                "cross_validation_folds": n_folds,
            },
        )
        return fold_loaders, metadata

    dataset = RumorGraphDataset(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_graph_nodes=max_graph_nodes,
        early_cutoff_minutes=early_cutoff_minutes,
        node_token_root=node_token_root,
    )
    if cv_protocol == "bigcn":
        folds = stratified_bigcn_kfold(
            sample_ids=dataset.sample_ids,
            labels=dataset.labels,
            n_folds=n_folds,
            seed=seed,
            val_ratio=val_ratio,
        )
    else:
        folds = stratified_kfold(
            sample_ids=dataset.sample_ids,
            labels=dataset.labels,
            n_folds=n_folds,
            val_ratio=val_ratio,
            seed=seed,
        )
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset.sample_ids)}

    def make_subset_loader(ids: List[str], shuffle: bool) -> DataLoader:
        subset = Subset(dataset, [id_to_index[sample_id] for sample_id in ids])
        return DataLoader(subset, shuffle=shuffle, **loader_kwargs)

    fold_loaders = [
        (
            make_subset_loader(train_ids, True),
            make_subset_loader(val_ids, False),
            make_subset_loader(test_ids, False),
        )
        for train_ids, val_ids, test_ids in folds
    ]
    return fold_loaders, dataset.metadata

