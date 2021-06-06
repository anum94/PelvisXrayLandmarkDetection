import argparse
import collections
import torch

# import data_loaders_old.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from torchsummary import summary
from data_loader.dataloader import XRayDataLoader
from torch.utils.data import DataLoader


def main(config):
    logger = config.get_logger("train")
    batch_size = config.config["data_loader"]["args"]["batch_size"]
    num_samples_to_load = config.config["data_loader"]["args"]["num_samples_load"]
    data_dir = config.config["data_loader"]["args"]["data_dir"]
    limit_samples = config.config["data_loader"]["args"]["limit_samples"]
    # setup data_torchloader instances
    # data_loader = config.initialize('data_loader', module_data)

    # Generators
    # training_set = Dataset(data_dir+"/train_overfit")
    data_loader = XRayDataLoader(
        data_dir + "/train_overfit",
        batch_size=batch_size,
        num_samples_to_load=num_samples_to_load,
        limit_samples=limit_samples,
    )
    # data_loader = DataLoader(training_set, batch_size = batch_size)

    # build model architecture, then print to console
    model = config.initialize("arch", module_arch)
    # model.double()
    logger.info(model)

    # summary(model, input_size=(1,480, 616))

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize("optimizer", torch.optim, trainable_params)

    lr_scheduler = config.initialize(
        "lr_scheduler", torch.optim.lr_scheduler, optimizer
    )

    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        # valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target=("optimizer", "args", "lr")
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target=("data_loader", "args", "batch_size"),
        ),
    ]
    config = ConfigParser(args, False, options)
    main(config)
