from datasets import load_dataset
import argparse
import os

def download_wikipedia_dataset(version: str, save_path: str):
    """
    Download the Wikipedia dataset for the specified version and save it to the given path.
    Args:
        version (str): The version of the Wikipedia dataset to download.
        save_path (str): The path where the dataset will be saved.
    """
    print(f"Downloading Wikipedia dataset version {version}...")
    # Download the Wikipedia dataset
    dataset = load_dataset("wikipedia", version, trust_remote_code=True)
    os.makedirs(save_path, exist_ok=True)
    # Save the dataset to the specified path
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia dataset.")
    parser.add_argument("--version", type=str, default="20220301.simple", required=True, help="Version of the Wikipedia dataset to download.")
    parser.add_argument("--save_path", type=str, default="data/wikipedia_simple_en.hf", required=True, help="Path to save the downloaded dataset.")
    args = parser.parse_args()

    download_wikipedia_dataset(args.version, args.save_path)
