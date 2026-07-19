import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from rift_svc import DiT, RF
from rift_svc.dataset import SVCDataset, collate_fn, LengthBucketedRandomBatchSampler
from rift_svc.lightning_module import RIFTSVCLightningModule
from rift_svc.utils import CustomProgressBar, load_state_dict
from rift_svc.optim import get_optimizer

# PyTorch 2.9 deprecation-warns on this legacy TF32 call (once per process, incl.
# each Windows dataloader worker), but the new per-backend API breaks
# torch.compile's internal precision check — so keep the legacy call and just
# silence that specific warning.
import warnings
warnings.filterwarnings('ignore', message='.*control TF32 behavior.*')
# Cosmetic: the Lightning model summary can't estimate size under bf16-mixed and
# falls back to 32-bit numbers; training itself is unaffected
warnings.filterwarnings('ignore', message='.*not supported by the model summary.*')
torch.set_float32_matmul_precision('high')


class SlimInferenceCheckpoint(pl.Callback):
    """Saves small inference-only checkpoints: a single fp16 copy of the weights
    (EMA already merged in) plus the config — no optimizer state, no separate
    raw/EMA copies. Loadable by infer.py and the GUI like any checkpoint, but
    NOT resumable; resume from the full 'last.ckpt' kept alongside.
    """

    def __init__(self, dirpath, every_n_train_steps):
        self.dirpath = dirpath
        self.every_n_train_steps = every_n_train_steps
        self.best_mcd = float('inf')
        self._last_saved_step = -1

    def _save(self, pl_module, filename):
        weights = {k: v.detach().cpu() for k, v in pl_module.model.state_dict().items()}
        if pl_module.ema_shadow is not None:
            # Inference prefers EMA weights; merge them in so only one copy is stored
            for k, v in pl_module.ema_shadow.items():
                weights[k] = v.detach().cpu()
        state_dict = {
            'model.' + k: (v.half() if v.is_floating_point() else v)
            for k, v in weights.items()
        }
        ckpt = {'state_dict': state_dict, 'hyper_parameters': {'cfg': pl_module.cfg}}
        os.makedirs(self.dirpath, exist_ok=True)
        torch.save(ckpt, os.path.join(self.dirpath, filename))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if (step > 0 and step % self.every_n_train_steps == 0
                and step != self._last_saved_step and trainer.is_global_zero):
            self._last_saved_step = step
            self._save(pl_module, f'model-step={step}.ckpt')

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        mcd = trainer.callback_metrics.get('val/mcd')
        if mcd is not None and float(mcd) < self.best_mcd:
            self.best_mcd = float(mcd)
            self._save(pl_module, 'model-best.ckpt')


def load_pretrained_weights(rf, pretrained_path, init_spk_embed=True):
    """Load pretrain weights into the RF model, allowing mismatched speaker tables.

    When the checkpoint has no usable speaker table (the distributed pretrains ship
    without one, and a user pretrain's table has a different row count), the new
    speaker embedding(s) optionally start from the pretrain's "average voice" —
    the mean of its speaker table if present, else the null-speaker (CFG
    unconditional) embedding — instead of random init, which speeds up timbre
    convergence on finetune.
    """
    state_dict = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    spk_key = 'transformer.spk_embed.weight'
    pretrain_spk_embed = state_dict.get(spk_key)
    if pretrain_spk_embed is not None and pretrain_spk_embed.shape != rf.transformer.spk_embed.weight.shape:
        # Different speaker count: keep the table aside for the mean init below
        del state_dict[spk_key]

    missing_keys, unexpected_keys = load_state_dict(rf, state_dict)

    if init_spk_embed and spk_key in missing_keys:
        if pretrain_spk_embed is not None:
            init = pretrain_spk_embed.mean(dim=0, keepdim=True)
        else:
            init = state_dict.get('transformer.null_spk_embed.weight')
        if init is not None:
            with torch.no_grad():
                rf.transformer.spk_embed.weight.copy_(
                    init.to(rf.transformer.spk_embed.weight.dtype).expand_as(rf.transformer.spk_embed.weight))
            print("Initialized speaker embedding(s) from the pretrain average voice")

    return missing_keys, unexpected_keys


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    train_dataset = SVCDataset(
        **cfg.dataset,
        split="train"
    )
    
    val_dataset = SVCDataset(
        **cfg.dataset,
        split="test"
    )

    transformer = DiT(
        **cfg.model,
        num_speaker=train_dataset.num_speakers,
    )

    rf = RF(
        transformer=transformer,
        time_schedule=cfg.training.time_schedule,
    )

    # Load pretrained weights if specified
    if cfg.training.get('pretrained_path', None) is not None:
        missing_keys, unexpected_keys = load_pretrained_weights(
            rf, cfg.training.pretrained_path,
            init_spk_embed=cfg.training.get('init_spk_embed_from_pretrain', True),
        )
        print(f"Loaded pretrained model from {cfg.training.pretrained_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    if cfg.training.get('lora_training', False):
        rf.transformer.apply_lora(cfg.training.lora_rank, cfg.training.lora_alpha)
    
    if cfg.training.get('freeze_adaln_and_tembed', False):
        rf.transformer.freeze_adaln_and_tembed()

    if cfg.training.get('gradient_checkpointing', False):
        rf.transformer.gradient_checkpointing = True
        print("Gradient checkpointing enabled for the transformer blocks")

    if cfg.training.get('compile_model', False):
        # Compile only the DiT transformer, in place: Module.compile keeps parameter
        # names unchanged, so EMA, checkpoints and inference loading are unaffected.
        # inductor needs triton (often missing on Windows); aot_eager works everywhere.
        try:
            import triton  # noqa: F401
            compile_backend = 'inductor'
        except ImportError:
            compile_backend = 'aot_eager'
        import torch._dynamo.config as dynamo_cfg
        if hasattr(dynamo_cfg, 'enable_cpp_symbolic_shape_guards'):
            # C++ shape guards need a host compiler (MSVC on Windows); python guards work fine
            dynamo_cfg.enable_cpp_symbolic_shape_guards = False
        rf.transformer.compile(backend=compile_backend, dynamic=True)
        print(f"torch.compile enabled for the transformer (backend={compile_backend})")

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    optimizer, lr_scheduler = get_optimizer(
        cfg.training.optimizer_type,
        rf, 
        cfg.training.learning_rate, 
        eval(cfg.training.betas), 
        cfg.training.weight_decay, 
        warmup_steps,
        max_steps=cfg.training.max_steps,
        min_lr=cfg.training.get('min_lr', 0.0),
        lora_training=cfg.training.get('lora_training', False),
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['spk2idx'] = train_dataset.spk2idx
    model = RIFTSVCLightningModule(
        model=rf,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        cfg=cfg_dict
    )

    run_name = cfg.training.run_name
    ckpt_dir = os.path.join('ckpts', run_name)

    if cfg.training.get('slim_checkpoints', False):
        # Disk-friendly mode: one full checkpoint ('last.ckpt', with optimizer
        # state, overwritten in place) to resume training from, plus small fp16
        # inference-only checkpoints (model-step=N / model-best) at each interval
        resume_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            save_top_k=1,
            every_n_train_steps=cfg.training.save_per_steps,
            save_weights_only=False,
            enable_version_counter=False,
        )
        checkpoint_callbacks = [
            resume_checkpoint_callback,
            SlimInferenceCheckpoint(ckpt_dir, cfg.training.save_per_steps),
        ]
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='model-{step}',
            save_top_k=-1,
            save_last='link',
            every_n_train_steps=cfg.training.save_per_steps,
            save_weights_only=cfg.training.save_weights_only,
        )

        # Keep the checkpoint with the best validation MCD (updated at each validation)
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='model-best',
            monitor='val/mcd',
            mode='min',
            save_top_k=1,
            auto_insert_metric_name=False,
            save_weights_only=cfg.training.save_weights_only,
        )
        checkpoint_callbacks = [checkpoint_callback, best_checkpoint_callback]

    # Logger selection based on config
    logger_type = cfg.training.get('logger', 'wandb').lower()

    if logger_type == 'wandb':
        # Use Weights & Biases logger
        logger = WandbLogger(
            project=cfg.training.wandb_project,
            name=run_name,
            id=cfg.training.get('wandb_resume_id', None),
            resume='allow',
        )
        if logger.experiment.config:
            # Merge with existing config, giving priority to existing values
            logger.experiment.config.update(cfg_dict, allow_val_change=True)
        else:
            # If no existing config, set it directly
            logger.experiment.config.update(cfg_dict)
    elif logger_type == 'tensorboard':
        # Use TensorBoard logger
        tensorboard_log_dir = os.path.join('logs', run_name)
        logger = TensorBoardLogger(
            save_dir=tensorboard_log_dir,
            name=None,  # Use the directory as is without adding another subfolder
            version='',  # Don't add version subdirectory
        )
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")

    callbacks = checkpoint_callbacks + [CustomProgressBar()]
    if lr_scheduler is not None:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    accum = cfg.training.grad_accumulation_steps

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=accum,
        callbacks=callbacks,
        logger=logger,
        # val_check_interval counts batches, so scale by accum to validate every
        # test_per_steps optimizer steps
        val_check_interval=cfg.training.test_per_steps * accum,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=1,
    )

    if hasattr(optimizer, 'train'):
        optimizer.train()

    # Sample with replacement sized (in batches) for the whole run so training is
    # one single "epoch": Lightning otherwise flushes gradient accumulation at every
    # epoch boundary, which on small finetune datasets (few batches per epoch) makes
    # accumulation nearly a no-op. Also avoids per-epoch dataloader worker restarts.
    # Batches are bucketed by frame length to minimize padding waste.
    train_batch_sampler = LengthBucketedRandomBatchSampler(
        train_dataset,
        batch_size=cfg.training.batch_size_per_gpu,
        num_batches=cfg.training.max_steps * accum,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_fn,
        ),
        ckpt_path=cfg.training.get('resume_from_checkpoint', None),
    )

if __name__ == "__main__":
    main()