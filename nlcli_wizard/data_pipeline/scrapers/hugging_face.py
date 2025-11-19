"""
Generic utility for downloading raw files from a Hugging Face dataset repo.
"""

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from pathlib import Path
import shutil
import os 


class HuggingFaceScraper:
    """
    A helper class to download raw files from a Hugging Face dataset repository.

    Example:
        scraper = HuggingFaceScraper(repo="MattCoddity", dataset="dockerNLcommands")
        scraper.download_raw_files(target_dir="assets/data/base/dockerNLcommands")
    """

    def __init__(self, repo: str, dataset: str):
        self.repo = repo
        self.dataset = dataset
        self.repo_id = f"{repo}/{dataset}"
        self.api = HfApi()

    def list_files(self):
        """List all non-hidden files in the repository."""
        all_files = list_repo_files(self.repo_id, repo_type="dataset")
        visible_files = [f for f in all_files if not f.startswith(".")]
        return visible_files

    def _clean_huggingface_cache(self, target_dir: Path):
            """
            Performs a two-part cleanup:
            1. Deletes the specific repository's global cache directory (~/.cache/huggingface/hub/...)
            2. Deletes the temporary .cache folder created inside the target_dir.
            """
            
            # Clean the GLOBAL Hugging Face cache
            try:
                cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
                repo_folder_name_segment = self.repo_id.replace("/", "--")
                cache_dir_to_delete = cache_root / f"datasets--{repo_folder_name_segment}"
                
                if cache_dir_to_delete.is_dir():
                    print(f"Cleanup (Global): Deleting global cache: {cache_dir_to_delete.resolve()}")
                    shutil.rmtree(cache_dir_to_delete)
                    print("Global cache cleanup complete.")
                else:
                    print(f"Cleanup (Global): Global cache directory not found: {cache_dir_to_delete.resolve()}")

            except Exception as e:
                print(f"Warning: Global cache cleanup failed. Error: {e}")

            # Clean the TEMPORARY .cache folder inside the local download dir
            try:
                # Look for the .cache folder inside the target download directory
                local_cache_dir_to_delete = target_dir / ".cache"

                if local_cache_dir_to_delete.is_dir():
                    print(f"Cleanup (Local): Deleting local temporary cache: {local_cache_dir_to_delete.resolve()}")
                    shutil.rmtree(local_cache_dir_to_delete)
                    print("Local temporary cache cleanup complete.")
                else:
                    print(f"Cleanup (Local): Local temporary cache not found: {local_cache_dir_to_delete.resolve()}")

            except Exception as e:
                print(f"Warning: Local cache cleanup failed. Error: {e}")


    def download_raw_files(self, target_dir: str | Path):
        """
        Download all visible (non-hidden) files from the Hugging Face dataset repo,
        then calls the cleanup method.

        Args:
            target_dir: Directory to store the downloaded files.
        """
        target_dir = Path(target_dir) # Convert to Path object once
        target_dir.mkdir(parents=True, exist_ok=True)

        files = self.list_files()
        print(f"Found {len(files)} visible files in {self.repo_id}")

        for file_name in files:
            print(f"Downloading {file_name} ...")
            try:
                # Use force_download to ensure a fresh copy is made to target_dir
                file_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_name,
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    force_download=True,
                )
                print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Skipped {file_name}: {e}")

        print(f"Download complete. Files saved to: {target_dir.resolve()}")
        
        # Pass the target_dir Path object to the cleanup method
        self._clean_huggingface_cache(target_dir)