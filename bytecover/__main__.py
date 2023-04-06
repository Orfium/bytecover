import wandb

from bytecover.models.train_module import TrainModule
from bytecover.utils import initialize_logging, load_config

config = load_config(config_path="config/config.yaml")
initialize_logging(config_path="config/logging_config.yaml", debug=False)
if config["wandb"]:
    wandb.init(
        # set the wandb project where this run will be logged
        project="ByteCover",
        # track hyperparameters and run metadata
        config=config["train"],
    )
trainer = TrainModule(config)
trainer.pipeline()
trainer.test()
