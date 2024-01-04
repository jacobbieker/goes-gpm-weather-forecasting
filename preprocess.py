import subprocess

def install(package):
    subprocess.check_call(["/srv/conda/envs/notebook/bin/python", "-m", "pip", "install", package])


try:
    import torch
except ImportError:
    # Special call to make sure to only install the CPU version of PyTorch
    subprocess.check_call(
        [
            "/srv/conda/envs/notebook/bin/python",
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )
try:
    import gcsfs
except ImportError:
    install("gcsfs")

try:
    import datasets
except ImportError:
    install("datasets")

try:
    import adlfs
except ImportError:
    install("adlfs")

from geospatial_dataset import GeospatialDataset
import torch
from torch.utils.data import DataLoader

from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem

import os
import fsspec
import uuid
import argparse
import datetime as dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="jacobbieker/era5-42hour")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--goes", action="store_true")
    parser.add_argument("--era5", action="store_false")
    parser.add_argument("--imerg", action="store_true")
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--forecast-hours", type=int, default=42)
    parser.add_argument("--forecast-frequency-hours", type=int, default=6)
    parser.add_argument("--forecast-timesteps", type=int, default=7)
    args = parser.parse_args()

    fs = HfFileSystem(token=os.environ["HF_TOKEN"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    files = api.list_repo_files(args.dataset_name, repo_type="dataset")

    last_done_batch = 0
    if args.test:
        files = api.list_repo_files(args.dataset_name, repo_type="dataset")
        for fi in files:
            if "data/" not in fi:
                continue
            fi = fi.split("/")[-1]
            fi = fi.split(".")[0]  # Should just be the number
            fi = int(fi)
            # Remove the number from total_time_numbers if it is in there
            if fi > last_done_batch:
                last_done_batch = fi

    if args.test:
        start_datetime = dt.datetime(2020, 1, 1)
        end_datetime = dt.datetime(2020, 12, 31)
    elif args.val:
        start_datetime = dt.datetime(2021, 1, 3)
        end_datetime = dt.datetime(2021, 5, 31)
    else:
        start_datetime = dt.datetime(2017, 3, 1)
        end_datetime = dt.datetime(2019, 12, 29)

    dataloader = DataLoader(
        GeospatialDataset(
            image_size=args.input_size,
            num_frames=4,
            use_goes=args.goes,
            use_era5_input=args.era5,
            use_imerg=args.imerg,
            start_val_idx=last_done_batch,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            test=args.test,
            forecast_hours=args.forecast_hours,
            forecast_frequency_hours=args.forecast_frequency_hours,
            forecast_timesteps=args.forecast_timesteps,
        ),
        batch_size=1 if args.test else 8,
        num_workers=0,
    )

    # Save out the batches
    batches_to_upload = []
    for i, batch in enumerate(dataloader):
        tmp_folder_num = 0
        if args.test:
            inputs, targets, val_idx = batch
            val_idx = val_idx.numpy()[0]
            out_filename = val_idx
            if fs.exists(f"datasets/{args.dataset_name}/data/{tmp_folder_num}/{out_filename}.pt"):
                continue
            batch = inputs, targets
        else:
            out_filename = uuid.uuid4()

        torch.save(batch, f"{out_filename}.pt")
        with fs.open(
            f"datasets/{args.dataset_name}/data/{tmp_folder_num}/{out_filename}.pt", "wb"
        ) as f:
            with fsspec.open(f"{out_filename}.pt", "rb") as rf:
                f.write(rf.read())
        os.remove(f"{out_filename}.pt")
