from __future__ import annotations

import argparse
import os
from pathlib import Path

import kagglehub

from service.env import load_repo_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and locate the Chest X-Ray Pneumonia dataset via KaggleHub.",
    )
    parser.add_argument(
        "--dataset",
        default="paultimothymooney/chest-xray-pneumonia",
        help="KaggleHub dataset slug (default: %(default)s)",
    )
    parser.add_argument(
        "--env-file",
        default=".env.local",
        help="Env filename to load from repo root/parents (default: %(default)s)",
    )
    args = parser.parse_args()

    dotenv_loaded = load_repo_dotenv(args.env_file)
    token_set = bool(os.environ.get("KAGGLE_API_TOKEN"))

    print({"dotenv_loaded": dotenv_loaded, "KAGGLE_API_TOKEN_set": token_set})

    # KaggleHub will download the dataset on first run and then reuse
    # the cached copy on subsequent runs (no repeat 2.29GB download).
    # the data lives under .venv/lib/python3.12/site-packages/kagglehub/..., not inside the repo.
    path = kagglehub.dataset_download(args.dataset)
    print("Downloaded to:", path)

    base = Path(path)
    candidates = [p for p in base.rglob("chest_xray") if p.is_dir()]
    chest_xray_root = candidates[0] if candidates else base

    print("Using chest_xray_root:", chest_xray_root)
    for split in ["train", "val", "test"]:
        split_dir = chest_xray_root / split
        print(f"{split}:", split_dir)


if __name__ == "__main__":
    main()
