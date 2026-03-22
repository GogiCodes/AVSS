#!/usr/bin/env python3
"""
Training script for RAVSS model on LRS2-2Mix dataset
Usage: python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml
"""

import datetime
from pathlib import Path
from collections import defaultdict
import fire
from typing import Dict, List, Any
import logging

import torch
import numpy as np
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    CosineAnnealingScheduler,
    global_step_from_engine,
)
import ignite
import ignite.distributed as idist
from ignite.metrics import RunningAverage, Loss, EpochMetric
from ignite.utils import convert_tensor
from torch.cuda.amp import GradScaler, autocast
from ignite.utils import manual_seed, setup_logger

# Custom imports
import dataset_lrs2_2mix as dataset
import models
import utils
import criterion as losses

import os


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - {tag} time (seconds): {elapsed:.2f}\n{metrics_output}")


def transfer_to_device(batch):
    DEVICE = idist.device()
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def __setup(config: Path,
            default_args=utils.DEFAULT_ARGS,
            **override_kwargs) -> Dict[str, Any]:
    config_parameters = utils.parse_config_or_kwargs(
        config, default_args=default_args, **override_kwargs)
    return config_parameters


def log_basic_info(logger, config_parameters):
    logger.info(f"Training {config_parameters['model']} on LRS2-2Mix")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        from torch.backends import cudnn
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config_parameters.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def train(local_rank, config_parameters):
    """Train RAVSS model on LRS2-2Mix dataset"""
    rank = idist.get_rank()
    manual_seed(config_parameters["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name='lrs2_train')
    
    # Setup output directory
    outputpath = config_parameters['outputpath']
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config_parameters['model']}_lrs2_{now}"
        output_dir = Path(outputpath) / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        config_parameters["outputpath"] = output_dir.as_posix()
        
        log_file = output_dir / 'train.log'
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        logger.info(f"Output dir: {config_parameters['outputpath']}")
        log_basic_info(logger, config_parameters)

        if "cuda" in device.type:
            config_parameters["cuda device name"] = torch.cuda.get_device_name(local_rank)

    # Load datasets
    train_df = config_parameters.get('train_scp')
    val_df = config_parameters.get('val_scp')
    ds_root = config_parameters.get('dataset_root_path', 'lrs2_rebuild')
    
    # Make paths absolute if relative
    if not os.path.isabs(ds_root):
        ds_root = os.path.join(os.getcwd(), ds_root)
    
    logger.info(f"Loading datasets from {ds_root}")

    # Create datasets
    train_ds = dataset.LRS2_2Mix_Dataset(
        mix_num=config_parameters.get('mix_num', 2),
        scp_file=train_df,
        ds_root=ds_root,
        dstype='train',
        batch_size=config_parameters.get('batch_size', 4)
    )

    val_ds = dataset.LRS2_2Mix_Dataset(
        mix_num=config_parameters.get('mix_num', 2),
        scp_file=val_df,
        ds_root=ds_root,
        dstype='val',
        batch_size=config_parameters.get('batch_size', 4)
    )

    # Create dataloaders
    nproc = idist.get_nproc_per_node()
    train_loader = idist.auto_dataloader(
        train_ds,
        batch_size=1,
        num_workers=config_parameters.get('num_workers', 2),
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.dummy_collate_fn
    )

    val_loader = idist.auto_dataloader(
        val_ds,
        batch_size=1,
        num_workers=config_parameters.get('num_workers', 2),
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.dummy_collate_fn
    )

    # Create model
    model = getattr(models, config_parameters['model'])(
        num_spks=config_parameters.get('mix_num', 2)
    )

    # Load checkpoint if resuming
    resume_from = config_parameters.get("resume_from", None)
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

    # Setup distributed model
    model = idist.auto_model(model, sync_bn=True)

    # Setup optimizer
    optimizer = getattr(torch.optim, config_parameters["optimizer"])(
        model.parameters(),
        **config_parameters.get("optimizer_args", {})
    )

    # Setup loss functions
    loss_func = getattr(losses, config_parameters['loss'])().to(device)
    sisnr_func = getattr(losses, config_parameters['val_loss'])().to(device)
    pesq_func = getattr(losses, config_parameters['pesq'])().to(device)

    # Setup metrics
    metrics = {
        "si-snr-loss": Loss(sisnr_func),
        "pesq": Loss(pesq_func)
    }

    scaler = GradScaler(enabled=config_parameters.get('with_amp', False))

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        mixture, source, condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])

        with autocast(enabled=config_parameters.get('with_amp', False)):
            pred_wav, compare_a, compare_v = model(mixture, condition, condition.shape[0])
            loss = loss_func(source, pred_wav, compare_a, compare_v)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    def val_step(engine, batch):
        model.eval()
        mixture, source, condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])

        with torch.no_grad():
            with autocast(enabled=config_parameters.get('with_amp', False)):
                pred_wav, compare_a, compare_v = model(mixture, condition, condition.shape[0])

        return source, pred_wav

    # Create engines
    trainer = Engine(train_step)
    evaluator = Engine(val_step)

    # Attach metrics
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # Attach progress bar
    if idist.get_rank() == 0:
        common.ProgressBar(desc="Training", persist=False).attach(trainer)
        common.ProgressBar(desc="Validation", persist=False).attach(evaluator)

    # Attach handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(trainer):
        eval_state = evaluator.run(val_loader)
        log_metrics(logger, trainer.state.epoch, eval_state.times["COMPLETED"], "Val", eval_state.metrics)

    # Learning rate scheduler
    scheduler = CosineAnnealingScheduler(
        optimizer,
        'lr',
        start_value=config_parameters['optimizer_args'].get('lr', 0.0001),
        end_value=1e-6,
        cycle_size=config_parameters['epochs']
    )
    trainer.add_event_handler(Events.ITERATION_START, scheduler)

    # Checkpoint handler
    to_save = {"model": model}
    checkpoint_handler = Checkpoint(
        to_save,
        get_save_handler(config_parameters),
        n_saved=config_parameters.get('n_saved', 1),
        filename_prefix='best',
        score_function=lambda engine: -engine.state.metrics.get('si-snr-loss', 0),
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

    # Training loop
    state = trainer.run(train_loader, max_epochs=config_parameters['epochs'])
    
    logger.info("Training completed!")
    logger.info(f"Final metrics: {state.metrics if hasattr(state, 'metrics') else 'N/A'}")


def create_evaluator(model, metrics, pit, config_parameters, tag='val'):
    with_amp = config_parameters.get('with_amp', False)

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        mixture, source, condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])
        
        with autocast(enabled=with_amp):
            pred_wav, compare_a, compare_v = model(mixture, condition, condition.shape[0])

        return source, pred_wav

    evaluator = Engine(evaluate_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Eval ({tag})", persist=False).attach(evaluator)
            
    return evaluator


def get_save_handler(config):
    return DiskSaver(config["outputpath"], require_empty=False)


def run(config, **kwargs):
    setup_args = __setup(config, **kwargs)
    config_parameters = setup_args
    config_parameters['master_port'] = 3333
    spawn_kwargs = {
        "nproc_per_node": config_parameters.get('nproc_per_node', 1),
        "master_port": config_parameters['master_port']
    }
    with idist.Parallel(backend=config_parameters.get('backend', 'nccl'), **spawn_kwargs) as parallel:
        parallel.run(train, config_parameters)


if __name__ == "__main__":
    fire.Fire({"run": run})
