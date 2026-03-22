#!/usr/bin/env python3
"""
Test script for RAVSS model on LRS2-2Mix dataset
Usage: python run_lrs2_2mix_test.py run config/lrs2_2mix_test.yaml
"""

import datetime
from pathlib import Path
from collections import defaultdict
import fire
from typing import Dict, List, Any, Union, Sequence, Tuple
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
import soundfile as sf

# Import custom dataset and utilities
import dataset_lrs2_2mix as dataset
import models
import utils
import criterion as losses

import os
import pdb


def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\n Test time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


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
    logger.info(f"Test {config_parameters['model']} on LRS2-2Mix")
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


def test(local_rank, config_parameters):
    """
    Test a given model on LRS2-2Mix dataset.
    
    :param config: yaml config file
    :param **kwargs: parameters to overwrite yaml config
    """
    rank = idist.get_rank()
    manual_seed(config_parameters["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name='lrs2_test')
    
    outputpath = config_parameters['outputpath']
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config_parameters['model']}_lrs2_test_{now}"
        output_dir = Path(outputpath) / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        config_parameters["outputpath"] = output_dir.as_posix()
        log_file = output_dir / 'test.log'
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        logger.info(f"Output dir: {config_parameters['outputpath']}")
        log_basic_info(logger, config_parameters)

        if "cuda" in device.type:
            config_parameters["cuda device name"] = torch.cuda.get_device_name(local_rank)
  
    test_df = config_parameters.get('test_scp')
    ds_root = config_parameters.get('dataset_root_path', 'lrs2_rebuild')
    
    # Create full paths if they're relative
    if not os.path.isabs(ds_root):
        ds_root = os.path.join(os.getcwd(), ds_root)
    
    if test_df and not os.path.isabs(test_df):
        test_df = os.path.join(os.getcwd(), test_df)
    
    logger.info(f"Loading test dataset from {ds_root}")
    logger.info(f"Test list: {test_df}")
    
    # Create test dataset
    test_ds = dataset.LRS2_2Mix_Dataset(
        mix_num=config_parameters.get('mix_num', 2),
        scp_file=test_df,
        ds_root=ds_root,
        dstype='test',
        batch_size=config_parameters.get('batch_size', 2)
    )
    
    # Create dataloader
    nproc = idist.get_nproc_per_node()
    test_loader = idist.auto_dataloader(
        test_ds, 
        batch_size=1,  # Already batched in dataset
        num_workers=config_parameters.get('num_workers', 0),
        shuffle=False, 
        drop_last=False, 
        collate_fn=dataset.dummy_collate_fn
    )

    # Create model
    model = getattr(models, config_parameters['model'])(num_spks=config_parameters.get('mix_num', 2))
    
    # Load checkpoint if provided
    checkpoint_path = config_parameters.get("test_cdp", None)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning("No checkpoint provided or checkpoint not found. Testing with random weights.")
    
    # Setup distributed model
    model = idist.auto_model(model, sync_bn=False)

    # Setup loss functions
    sisnr_func = getattr(losses, config_parameters.get('val_loss', 'SI_SNR'))().to(idist.device())
    loss_func = getattr(losses, config_parameters.get('loss', 'SI_SNR_PIT'))().to(idist.device())
    pesq_func = getattr(losses, config_parameters.get('pesq', 'PESQ'))().to(idist.device())
    
    # Setup metrics
    metrics = {
        "si-snr-loss": Loss(sisnr_func),
        "pesq": Loss(pesq_func)
    }

    # Create evaluator
    evaluator = create_evaluator(model, metrics=metrics, pit=loss_func, config_parameters=config_parameters, tag='test')

    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        state = evaluator.run(test_loader)
        log_metrics(logger, state.times["COMPLETED"], "Test", state.metrics)
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.exception("Error during evaluation:")
        raise e


def create_evaluator(model, metrics, pit, config_parameters, tag='val'):
    with_amp = config_parameters.get('with_amp', False)

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        mixture, source, condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])
        
        with autocast(enabled=with_amp):
            pred_wav, compare_a, compare_v = model(mixture, condition, condition.shape[0])

        # For evaluation, use predicted as-is (no source reordering since we don't have ground truth)
        select_source = source
        re_pred_wav = pred_wav

        return select_source, re_pred_wav

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Test ({tag})", persist=False).attach(evaluator)
            
    return evaluator


def run(config, **kwargs):
    setup_args = __setup(config, **kwargs)
    config_parameters = setup_args
    config_parameters['master_port'] = 3333
    spawn_kwargs = {
        "nproc_per_node": config_parameters.get('nproc_per_node', 1), 
        "master_port": config_parameters['master_port']
    }
    with idist.Parallel(backend=config_parameters.get('backend', 'nccl'), **spawn_kwargs) as parallel:
        parallel.run(test, config_parameters)


def get_save_handler(config):
    return DiskSaver(config["outputpath"], require_empty=False)


if __name__ == "__main__":
    fire.Fire({"run": run})
