from typing import TypedDict

import torch


class ValDict(TypedDict):
    anchor_id: str
    positive_id: str
    negative_id: str
    f_t: torch.Tensor
    f_c: torch.Tensor


class BatchDict(TypedDict):
    anchor_id: str
    anchor: torch.Tensor
    anchor_label: torch.Tensor
    positive_id: str
    positive: torch.Tensor
    negative_id: str
    negative: torch.Tensor


class Postfix(TypedDict):
    Epoch: int
    train_loss: float
    train_loss_step: float
    train_cls_loss: float
    train_cls_loss_step: float
    train_triplet_loss: float
    train_triplet_loss_step: float
    val_loss: float
    mr1: float
    mAP: float


class TestResults(TypedDict):
    test_mr1: float
    test_mAP: float
