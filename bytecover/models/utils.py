import glob
import json
import os
import re
from typing import Dict, List, Tuple

import jsonlines
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from bytecover.models.data_loader import bytecover_dataloader
from bytecover.models.data_model import Postfix


def dataloader_factory(config: Dict, data_split: str) -> List[DataLoader]:
    seq_len_key = "max_seq_len"

    if data_split == "TRAIN":
        t_loaders = []
        for L in config["train"][seq_len_key]:
            t_loaders.append(
                bytecover_dataloader(
                    data_path=config["data_path"],
                    file_ext=config["file_extension"],
                    dataset_path=config["dataset_path"],
                    data_split=data_split,
                    debug=config["debug"],
                    max_len=L,
                    **config["train"],
                )
            )
        return t_loaders
    L = config[data_split.lower()][seq_len_key]
    return [
        bytecover_dataloader(
            data_path=config["data_path"],
            file_ext=config["file_extension"],
            dataset_path=config["dataset_path"],
            data_split=data_split,
            debug=config["debug"],
            max_len=L,
            **config[data_split.lower()],
        )
    ]


def validation_triplet_sampling(anchor_id: str, val_ids: List[str], df: pd.DataFrame) -> Dict[str, int]:
    np.random.shuffle(df.loc[anchor_id, "versions"])
    pos_list = np.setdiff1d(df.loc[anchor_id, "versions"], anchor_id)
    pos_id = np.random.choice(pos_list, 1)[0]
    pos_id = val_ids.index(pos_id)

    neg_id = df.loc[~df.index.isin([anchor_id] + list(pos_list))].sample(1).index[0]
    neg_id = val_ids.index(neg_id)

    return dict(pos_id=pos_id, neg_id=neg_id)


def calculate_ranking_metrics(embeddings: np.ndarray, cliques: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(embeddings)
    s_distances = np.argsort(distances, axis=1)
    cliques = np.array(cliques)
    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques = np.tile(query_cliques, (search_cliques.shape[-1], 1)).T
    mask = np.equal(search_cliques, query_cliques)

    ranks = mask.argmax(axis=1)

    cumsum = np.cumsum(mask, axis=1)
    mask2 = mask * cumsum
    mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
    average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

    return (ranks, average_precisions)


def dir_checker(output_dir: str) -> str:
    output_dir = re.sub(r"run-[0-9]+/*", "", output_dir)
    runs = glob.glob(os.path.join(output_dir, "run-*"))
    if runs != []:
        max_run = max(map(lambda x: int(x.split("-")[-1]), runs))
        run = max_run + 1
    else:
        run = 0
    outdir = os.path.join(output_dir, f"run-{run}")
    return outdir


def save_predictions(outputs: Dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])


def save_logs(outputs: dict, output_dir: str, name: str = "log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(log_file, "a") as f:
        f.write(outputs)


def save_best_log(outputs: Postfix, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "best-log.json")
    with open(log_file, "w") as f:
        json.dump(outputs, f, indent=2)
