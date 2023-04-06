import logging
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm, trange

from bytecover.models.data_model import BatchDict, Postfix, TestResults, ValDict
from bytecover.models.early_stopper import EarlyStopper
from bytecover.models.modules import Bottleneck, Resnet50
from bytecover.models.utils import (
    average_precision,
    dataloader_factory,
    dir_checker,
    rank_one,
    save_best_log,
    save_logs,
    save_predictions,
)

logger: logging.Logger = logging.getLogger()  # The logger used to log output


class TrainModule:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.state = "initializing"
        self.best_model_path: str = None
        self.num_classes = self.config["train"]["num_classes"]
        self.max_len = self.config["train"]["max_seq_len"][0]

        self.model = Resnet50(
            Bottleneck,
            num_channels=self.config["num_channels"],
            num_classes=self.num_classes,
            compress_ratio=self.config["train"]["compress_ratio"],
            tempo_factors=self.config["train"]["tempo_factors"],
        )
        self.model.to(self.config["device"])
        if self.config["wandb"]:
            wandb.watch(self.model)
        self.postfix: Postfix = {}

        self.triplet_loss = nn.TripletMarginLoss(margin=config["train"]["triplet_margin"])
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=config["train"]["smooth_factor"])

        self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])
        self.optimizer = self.configure_optimizers()
        if self.config["device"] != "cpu":
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["train"]["mixed_precision"])

    def pipeline(self) -> None:
        self.config["val"]["output_dir"] = dir_checker(self.config["val"]["output_dir"])

        if self.config["train"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["train"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["train"]["model_ckpt"]}')

        self.t_loaders = dataloader_factory(config=self.config, data_split="TRAIN")
        self.v_loader = dataloader_factory(config=self.config, data_split="VAL")[0]

        self.state = "running"

        self.pbar = trange(
            self.config["train"]["epochs"], disable=(not self.config["progress_bar"]), position=0, leave=True
        )
        for epoch in self.pbar:
            if self.state in ["early_stopped", "interrupted", "finished"]:
                return

            self.postfix["Epoch"] = epoch
            self.pbar.set_postfix(self.postfix)

            try:
                self.train_procedure()
            except KeyboardInterrupt:
                logger.warning("\nKeyboard Interrupt detected. Attempting gracefull shutdown...")
                self.state = "interrupted"
            except Exception as err:
                raise (err)

            if self.state == "interrupted":
                self.validation_procedure()
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
                )

        self.state = "finished"

    def validate(self) -> None:
        self.v_loader = dataloader_factory(config=self.config, data_split="VAL")[0]
        self.state = "running"
        self.validation_procedure()
        self.state = "finished"

    def test(self) -> None:
        self.test_loader = dataloader_factory(config=self.config, data_split="TEST")[0]
        self.test_results: TestResults = {}

        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path), strict=False)
            logger.info(f"Best model loaded from checkpoint: {self.best_model_path}")
        elif self.config["test"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["test"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["test"]["model_ckpt"]}')
        elif self.state == "initializing":
            logger.warning("Warning: Testing with random weights")

        self.state = "running"
        self.test_procedure()
        self.state = "finished"

    def train_procedure(self) -> None:
        self.model.train()
        pbar_loaders = tqdm(self.t_loaders, disable=(not self.config["progress_bar"]), position=1, leave=False)
        for _, t_loader in enumerate(pbar_loaders):
            train_loss_list = []
            train_cls_loss_list = []
            train_triplet_loss_list = []
            self.max_len = t_loader.dataset.max_len
            pbar_loaders.set_postfix_str(f"max_seq_len={self.max_len}")
            for step, batch in tqdm(
                enumerate(t_loader),
                total=len(t_loader),
                disable=(not self.config["progress_bar"]),
                position=2,
                leave=False,
            ):
                train_step = self.training_step(batch)
                self.postfix["train_loss_step"] = float(f"{train_step['train_loss_step']:.3f}")
                train_loss_list.append(train_step["train_loss_step"])
                self.postfix["train_cls_loss_step"] = float(f"{train_step['train_cls_loss']:.3f}")
                train_cls_loss_list.append(train_step["train_cls_loss"])
                self.postfix["train_triplet_loss_step"] = float(f"{train_step['train_triplet_loss']:.3f}")
                train_triplet_loss_list.append(train_step["train_triplet_loss"])
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
                )
                if self.config["wandb"]:
                    wandb.log(self.postfix)
                if step % self.config["train"]["log_steps"] == 0:
                    save_logs(
                        dict(
                            epoch=self.postfix["Epoch"],
                            seq_len=self.max_len,
                            step=step,
                            train_loss_step=f"{train_step['train_loss_step']:.3f}",
                            train_cls_loss_step=f"{train_step['train_cls_loss']:.3f}",
                            train_triplet_loss_step=f"{train_step['train_triplet_loss']:.3f}",
                        ),
                        output_dir=self.config["val"]["output_dir"],
                        name="log_steps",
                    )
            train_loss = torch.tensor(train_loss_list)
            train_cls_loss = torch.tensor(train_cls_loss_list)
            train_triplet_loss = torch.tensor(train_triplet_loss_list)
            self.postfix["train_loss"] = train_loss.mean().item()
            self.postfix["train_cls_loss"] = train_cls_loss.mean().item()
            self.postfix["train_triplet_loss"] = train_triplet_loss.mean().item()
            if self.config["wandb"]:
                wandb.log(self.postfix)
            self.validation_procedure()
            if self.config["wandb"]:
                wandb.log(self.postfix)
            self.overfit_check()
            self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}})

    def training_step(self, batch: BatchDict) -> Dict[str, float]:
        with torch.autocast(
            device_type=self.config["device"].split(":")[0], enabled=self.config["train"]["mixed_precision"]
        ):
            anchor = self.model.forward(batch["anchor"].to(self.config["device"]))
            positive = self.model.forward(batch["positive"].to(self.config["device"]))
            negative = self.model.forward(batch["negative"].to(self.config["device"]))
            l1 = self.triplet_loss(anchor["f_t"], positive["f_t"], negative["f_t"])
            labels = nn.functional.one_hot(batch["anchor_label"].long(), num_classes=self.num_classes)
            l2 = self.cls_loss(anchor["cls"], labels.float().to(self.config["device"]))
            loss = l1 + l2

        self.optimizer.zero_grad()
        if self.config["device"] != "cpu":
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"train_loss_step": loss.item(), "train_triplet_loss": l1.item(), "train_cls_loss": l2.item()}

    def validation_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[str, torch.Tensor] = {}
        for batch in tqdm(self.v_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            val_dict = self.validation_step(batch)
            if val_dict["f_t"].ndim == 1:
                val_dict["f_c"] = val_dict["f_c"].unsqueeze(0)
                val_dict["f_t"] = val_dict["f_t"].unsqueeze(0)
            for anchor_id, triplet_embedding, embedding in zip(val_dict["anchor_id"], val_dict["f_t"], val_dict["f_c"]):
                embeddings[anchor_id] = torch.stack([triplet_embedding, embedding])

        val_outputs = self.validation_epoch_end(embeddings)
        logger.info(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.postfix.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )

        if self.config["val"]["save_val_outputs"]:
            val_outputs["val_embeddings"] = torch.stack(list(embeddings.values()))[:, 1].numpy()
            save_predictions(val_outputs, output_dir=self.config["val"]["output_dir"])
            save_logs(self.postfix, output_dir=self.config["val"]["output_dir"])
        self.model.train()

    def validation_epoch_end(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        val_loss = torch.zeros(len(outputs))
        pos_ids = []
        neg_ids = []
        clique_ids = []
        for k, (anchor_id, embeddings) in enumerate(outputs.items()):
            clique_id, pos_id, neg_id = self.v_loader.dataset._triplet_sampling(anchor_id)
            val_loss[k] = self.triplet_loss(embeddings[0], outputs[pos_id][0], outputs[neg_id][0]).item()
            pos_ids.append(pos_id)
            neg_ids.append(neg_id)
            clique_ids.append(clique_id)
        anchor_ids = np.stack(list(outputs.keys()))
        preds = torch.stack(list(outputs.values()))[:, 1]
        self.postfix["val_loss"] = val_loss.mean().item()
        ranks = rank_one(embeddings=preds.numpy(), cliques=clique_ids)
        self.postfix["mr1"] = ranks.mean()
        average_precisions = average_precision(embeddings=preds.numpy(), cliques=clique_ids)
        self.postfix["mAP"] = average_precisions.mean()
        return {
            "triplet_ids": np.stack(list(zip(clique_ids, anchor_ids, pos_ids, neg_ids))),
            "ranks": ranks,
            "average_precisions": average_precisions,
        }

    def validation_step(self, batch: BatchDict) -> ValDict:
        anchor_id = batch["anchor_id"]
        positive_id = batch["positive_id"]
        negative_id = batch["negative_id"]

        features = self.model.forward(batch["anchor"].to(self.config["device"]))

        return {
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "f_t": features["f_t"].squeeze(0).detach().cpu(),
            "f_c": features["f_c"].squeeze(0).detach().cpu(),
        }

    def test_procedure(self) -> None:
        self.model.eval()
        clique_ids = []
        embeddings: Dict[str, torch.Tensor] = {}
        for batch in tqdm(self.test_loader, disable=(not self.config["progress_bar"])):
            clique_ids_batch = self.test_loader.dataset.labels.loc[batch["anchor_id"], "clique"]
            test_dict = self.validation_step(batch)
            if test_dict["f_c"].ndim == 1:
                test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
            for anchor_id, clique_id, embedding in zip(test_dict["anchor_id"], clique_ids_batch, test_dict["f_c"]):
                embeddings[anchor_id] = embedding
                clique_ids.append(clique_id)

        test_outputs = self.test_epoch_end(embeddings, clique_ids)
        logger.info(
            f"\n{' Test Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.test_results.items()])
            + f"\n{' End of Testing ':=^50}\n"
        )

        if self.config["test"]["save_test_outputs"]:
            test_outputs["test_embeddings"] = torch.stack(list(embeddings.values())).numpy()
            save_predictions(test_outputs, output_dir=self.config["test"]["output_dir"])
            save_logs(self.test_results, output_dir=self.config["test"]["output_dir"])

    def test_epoch_end(self, outputs: Dict[str, torch.Tensor], clique_ids: List[int]) -> Dict[str, np.ndarray]:
        anchor_ids = np.stack(list(outputs.keys()))
        preds = torch.stack(list(outputs.values()))
        ranks = rank_one(embeddings=preds.numpy(), cliques=clique_ids)
        average_precisions = average_precision(embeddings=preds.numpy(), cliques=clique_ids)
        self.test_results["test_mr1"] = ranks.mean()
        self.test_results["test_mAP"] = average_precisions.mean()
        return {
            "anchor_ids": np.stack(list(zip(clique_ids, anchor_ids))),
            "ranks": ranks,
            "average_precisions": average_precisions,
        }

    def overfit_check(self) -> None:
        if self.early_stop(self.postfix["val_loss"]):
            logger.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopped"

        if self.early_stop.counter > 0:
            logger.info("\nValidation loss was not improved")
        else:
            logger.info(f"\nMetric improved. New best score: {self.early_stop.min_validation_loss:.3f}")
            save_best_log(self.postfix, output_dir=self.config["val"]["output_dir"])

            logger.info("Saving model...")
            epoch = self.postfix["Epoch"]
            max_secs = self.max_len
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.config["val"]["output_dir"], "model", f"best-model-{epoch=}-{max_secs=}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["train"]["learning_rate"])

        return optimizer
