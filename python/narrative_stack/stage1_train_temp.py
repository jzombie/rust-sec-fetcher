def main():
    # %%
    from config import simd_r_drive_server_config
    from us_gaap_store import UsGaapStore

    from simd_r_drive_ws_client import DataStoreWsClient

    data_store = DataStoreWsClient(simd_r_drive_server_config.host, simd_r_drive_server_config.port)
    us_gaap_store = UsGaapStore(data_store)

    # %%
    # Obtain input dimension

    sample_embedding = us_gaap_store.lookup_by_index(0).embedding
    input_dim = sample_embedding.shape[0]

    # No longer needed on this thread
    del us_gaap_store

    # %%
    from models.pytorch.narrative_stack.stage1.dataset import IterableConceptValueDataset, collate_with_scaler
    from models.pytorch.narrative_stack.stage1 import Stage1Autoencoder

    # %%
    # Training

    import os
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader
    from utils.pytorch import get_device
    from config import project_paths
    from pathlib import Path
    import pytorch_lightning as pl

    device = get_device()

    # === CONFIG ===
    OUTPUT_PATH = Path(project_paths.python_data / "stage1_23_(no_pre_dedupe)")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    OPTUNA_DB_PATH = os.path.join(OUTPUT_PATH, "optuna_study.db")
    EPOCHS = 1000
    PATIENCE = 20  # Compensate for long annealing period + some

    ckpt_path = f"{OUTPUT_PATH}/stage1_resume-v10.ckpt"
    # ckpt_path = f"{OUTPUT_PATH}/manual_resumed_checkpoint.ckpt"
    # ckpt_path = None

    model = Stage1Autoencoder.load_from_checkpoint(ckpt_path,
        # lr=5e-5, # New
        lr=1e-5, # Medium
        # lr=2.5e-6, # Fine Tune
        # min_lr=1e-6,
        min_lr=1e-7
    )
    model = Stage1Autoencoder()

    batch_size = model.hparams.batch_size
    gradient_clip = model.hparams.gradient_clip

    train_loader = DataLoader(
        IterableConceptValueDataset(simd_r_drive_server_config, internal_batch_size=64, return_scaler=True, shuffle=True),
        batch_size=batch_size,
        # shuffle=True, # Moved to dataset
        collate_fn=collate_with_scaler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4, # TODO: Experiment with this. `prefetch_factor` tells each worker process how many batches it should always keep queued ahead of the trainer loop.
        num_workers=2,
    )

    val_loader = DataLoader(
        IterableConceptValueDataset(simd_r_drive_server_config, internal_batch_size=64, return_scaler=True, shuffle=False),
        batch_size=batch_size,
        # shuffle=False, # Moved to dataset
        collate_fn=collate_with_scaler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4, # TODO: Experiment with this
        num_workers=2,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss_epoch", patience=PATIENCE, verbose=True, mode="min"
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=OUTPUT_PATH,
        filename="stage1_resume",
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=TensorBoardLogger(OUTPUT_PATH, name="stage1_autoencoder"),
        callbacks=[early_stop_callback, model_checkpoint],
        accelerator="auto",
        devices=1,
        gradient_clip_val=gradient_clip,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        #
        ckpt_path=ckpt_path # TODO: Uncomment if resuming training AND wanting to restore existing model configuration
    )

    # %%



if __name__ == "__main__":
    main()