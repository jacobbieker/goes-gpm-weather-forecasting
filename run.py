if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    import torch

    torch.set_float32_matmul_precision("medium")

    from dotenv import load_dotenv
    from vit import ViT, LitViT
    from geospatial_dataset import PreprocessedGeospatialDataset
    from torch.utils.data import DataLoader
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch import loggers as pl_loggers
    import argparse

    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument(
        "--data-location", type=str, default="data/goes-imreg"
    )
    parser.add_argument("--model-name", type=str, default="goes_imerg_6hour")
    parser.add_argument("--test", action="store_true", help="Run only the test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to load from",
        default=None,
    )
    parser.add_argument("--no-goes", action="store_true", help="Don't use GOES data")
    parser.add_argument("--no-imerg", action="store_true", help="Don't use IMERG data")
    parser.add_argument("--no-fourier", action="store_true", help="Don't use Fourier features")
    parser.add_argument("--no-fix-goes-scaling", action="store_true", help="Don't fix GOES scaling")
    parser.add_argument("--use-era5", action="store_true", help="Use ERA5 as input")
    args = parser.parse_args()

    total_channels = 61
    if args.no_goes:
        total_channels -= 16
    if args.no_imerg:
        total_channels -= 1
    if args.no_fourier:
        total_channels -= 44
    if args.use_era5:
        total_channels = 3

    v = ViT(
        image_size=64 if args.use_era5 else 224,
        frames=4,  # number of frames of history, last 40min of satellite + IMERG
        image_patch_size=16,  # image patch size
        frame_patch_size=2,  # frame patch size
        output_image_size=16,  # Output image size in pixels, for ERA5 roughly 25km per pixel, 4km for GOES, 3 channels, 7 timesteps
        output_channels=3,
        output_frames=7,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=total_channels,
    )
    if args.checkpoint is not None:
        model = LitViT.load_from_checkpoint(args.checkpoint, model=v)
    else:
        model = LitViT(v)

    dataloader = DataLoader(
        PreprocessedGeospatialDataset(
            f"{args.data_location}/data/",
            use_goes=not args.no_goes,
            use_imerg=not args.no_imerg,
            use_fourier=not args.no_fourier,
            fix_goes_scaling=not args.no_fix_goes_scaling,
            use_era5=args.use_era5,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        PreprocessedGeospatialDataset(
            f"{args.data_location}-val/data/",
            train=False,
            use_goes=not args.no_goes,
            use_imerg=not args.no_imerg,
            use_fourier=not args.no_fourier,
            fix_goes_scaling=not args.no_fix_goes_scaling,
            use_era5=args.use_era5,
        ),
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        PreprocessedGeospatialDataset(
            f"{args.data_location}-test/data/",
            train=False,
            use_goes=not args.no_goes,
            use_imerg=not args.no_imerg,
            use_fourier=not args.no_fourier,
            fix_goes_scaling=not args.no_fix_goes_scaling,
            use_era5=args.use_era5,
        ),
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    callback = EarlyStopping(monitor="val_rmse", patience=5)
    trainer = L.Trainer(
        max_epochs=100,
        precision="16-mixed",
        accelerator="gpu",
        logger=[pl_loggers.TensorBoardLogger(save_dir=args.model_name)],
        accumulate_grad_batches=4,
        callbacks=[
            callback,
            ModelCheckpoint(
                monitor="val_rmse",
                dirpath="checkpoints",
                filename=f"{args.model_name}" + "_{epoch}_{val_rmse:.2f}",
                save_top_k=2,
                mode="min",
            ),
        ],
    )
    if not args.test:
        trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)
