"""
Download the Twitter Customer Support dataset from Kaggle.
"""
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(dest_dir: Path | None = None) -> Path:
    """Download twcs.csv from Kaggle and copy to data/raw/."""
    from src.config import DATA_RAW_DIR

    dest_dir = dest_dir or DATA_RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "twcs.csv"

    if dest_path.exists():
        logger.info("Dataset already exists at %s — skipping download.", dest_path)
        return dest_path

    try:
        import kagglehub
        logger.info("Downloading dataset from Kaggle (this may take a few minutes)…")
        downloaded_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        downloaded_path = Path(downloaded_path)

        # kagglehub returns a directory; find the CSV inside it
        csv_candidates = list(downloaded_path.rglob("twcs.csv"))
        if not csv_candidates:
            # try any CSV
            csv_candidates = list(downloaded_path.rglob("*.csv"))

        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV found in downloaded path: {downloaded_path}"
            )

        src_csv = csv_candidates[0]
        logger.info("Copying %s → %s", src_csv, dest_path)
        shutil.copy2(src_csv, dest_path)
        logger.info("Dataset ready at %s (%.1f MB)", dest_path, dest_path.stat().st_size / 1e6)
        return dest_path

    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info(
            "Manual fallback: download from "
            "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
            "and place twcs.csv in %s",
            dest_dir,
        )
        raise


if __name__ == "__main__":
    download_dataset()
