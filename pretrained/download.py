import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download


PC_NSF_HIFIGAN_URL = (
    "https://github.com/openvpi/vocoders/releases/download/"
    "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
)


def download_pc_nsf_hifigan(dest_dir="pretrained"):
    """Download the optional PC-NSF-HiFiGAN vocoder from the OpenVPI release.

    Note: distributed under CC BY-NC-SA 4.0 (non-commercial).
    """
    dest_dir = Path(dest_dir)
    model_dir = dest_dir / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    if (model_dir / "model.ckpt").exists():
        print(f"PC-NSF-HiFiGAN already present at {model_dir}, skipping.")
        return

    zip_path = dest_dir / "pc_nsf_hifigan.zip"
    print(f"Downloading PC-NSF-HiFiGAN from {PC_NSF_HIFIGAN_URL} ...")
    urllib.request.urlretrieve(PC_NSF_HIFIGAN_URL, zip_path)
    # The zip already contains the pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/ folder
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print(f"PC-NSF-HiFiGAN extracted to {model_dir}")


if __name__ == "__main__":
    model_path = snapshot_download(
        repo_id="Pur1zumu/RIFT-SVC-modules",
        local_dir='pretrained',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,        # Allow downloading new files
        ignore_patterns=["*.git*"],    # Ignore git-related files
        resume_download=True           # Resume interrupted downloads
    )
    download_pc_nsf_hifigan()
