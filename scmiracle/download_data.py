import os
import sys
import tarfile

import requests

_DATASETS = {
    "DOTEA_mudata": {
        "archive_name": "DOTEA_mudata.tar.gz",
        "extract_dir": "DOTEA_mudata",
        "url": "https://pub-c608fed55b484fd79e24335cfb8e2509.r2.dev/DOTEA_mudata.tar.gz",
    },
    "atlas_transfer_mudata": {
        "archive_name": "atlas_transfer_mudata.tar.gz",
        "extract_dir": "atlas_transfer_mudata",
        "url": "https://pub-c608fed55b484fd79e24335cfb8e2509.r2.dev/atlas_transfer_mudata.tar.gz",
    },
}

# Backward-compatible alias for older examples.
_ALIASES = {
    "DOTEA": "DOTEA_mudata",
}


def _canonical_dataset_name(data):
    return _ALIASES.get(data, data)


def _download_url(dataset_name):
    return _DATASETS[dataset_name]["url"]


def list_available_datasets():
    """
    Displays a list of all datasets that can be downloaded.
    """
    print("Available datasets for download:")
    for dataset_name in _DATASETS.keys():
        print(f"- {dataset_name}")
    print("\nOptional aliases:")
    for alias_name, target_name in _ALIASES.items():
        print(f"- {alias_name} -> {target_name}")
    print("\nPlease choose a dataset name exactly as listed above.")


def _extract_archive(archive_path, destination_dir):
    with tarfile.open(archive_path, "r:gz") as tar_ref:
        tar_ref.extractall(destination_dir)


def download(data, data_path):
    """
    Downloads and extracts a specified dataset to the given data_path.
    Includes a progress bar for downloads.

    Args:
        data (str): The dataset name, e.g. 'DOTEA_mudata'.
        data_path (str): The base directory where the data should be stored.
    """
    dataset_name = _canonical_dataset_name(data)
    if dataset_name not in _DATASETS:
        print(f"Error: Dataset '{data}' not recognized.")
        list_available_datasets()
        return

    dataset_info = _DATASETS[dataset_name]
    target_dir_path = os.path.join(data_path, dataset_info["extract_dir"])
    archive_path = os.path.join(data_path, dataset_info["archive_name"])

    if os.path.exists(target_dir_path):
        print(f"Directory {target_dir_path} already exists. Skipping download and extraction.")
        print(f"'{dataset_name}' data path is set to: {target_dir_path}")
        return

    os.makedirs(data_path, exist_ok=True)

    try:
        download_url = _download_url(dataset_name)
        print(f"Downloading '{dataset_name}' from {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded_size = 0

        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    sys.stdout.write(
                        f"\rDownloading: {downloaded_size / (1024 * 1024):.2f}MB / "
                        f"{total_size / (1024 * 1024):.2f}MB ({progress:.2f}%)"
                    )
                    sys.stdout.flush()
        if total_size > 0:
            sys.stdout.write("\n")
        print(f"Download complete: {archive_path}")

        print(f"Extracting {archive_path} to {data_path}...")
        _extract_archive(archive_path, data_path)
        print("Extraction complete.")

        if not os.path.exists(target_dir_path):
            raise FileNotFoundError(
                f"Expected extracted directory was not found: {target_dir_path}"
            )

        os.remove(archive_path)
        print(f"Removed temporary archive: {archive_path}")

    except requests.exceptions.RequestException as e:
        print(f"\nError during download: {e}")
    except (tarfile.TarError, OSError) as e:
        print(f"\nError during extraction: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print(f"'{dataset_name}' data path is set to: {target_dir_path}")
