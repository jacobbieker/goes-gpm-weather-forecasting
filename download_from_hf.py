from huggingface_hub import snapshot_download
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="jacobbieker/goes-imerg-6hour")
    parser.add_argument("--download_location", type=str, default="data/goes-imerg-6hour")
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        local_dir_use_symlinks=False,
        local_dir=args.download_location,
    )
