import os
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

RAID_DATA_URL_BASE = "https://dataset.raid-bench.xyz"
RAID_CACHE_DIR = Path(os.getenv("RAID_CACHE_DIR", "~/.cache/raid/")).expanduser()
RAID_CACHE_DIR.mkdir(exist_ok=True, parents=True)


def download_file(url: str, out: Path, cache=True) -> Path:
    """Download file from the remote to the given path, show progress bar for files >1MB.

    Returns the Path to the downloaded file.
    """
    # check filesize for cache
    response = urlopen(url)
    filesize = int(response.headers["Content-Length"])

    if cache and out.is_file() and out.stat().st_size == filesize:
        return out

    # download the file in 16KiB chunks
    print(f"Downloading {url} ({filesize}B) to {out.resolve()}")
    t = tqdm(total=filesize, unit="B", unit_scale=True)
    with open(out, "wb") as f:
        while chunk := response.read(16 * 1024):
            f.write(chunk)
            t.update(len(chunk))
    t.close()

    return out


def load_data(split: Literal["train", "test", "extra"], include_adversarial: bool = True, fp: str = None):
    """Load the given split of RAID into memory from the given filepath, downloading it if it does not exist.
    Returns a DataFrame.
    """
    if split not in ("train", "test", "extra"):
        raise ValueError('`split` must be one of ("train", "test", "extra")')

    fname = f"{split}.csv" if include_adversarial else f"{split}_none.csv"

    if fp is None:
        fp = RAID_CACHE_DIR / fname
    else:
        fp = Path(fp)

    fp = download_file(f"{RAID_DATA_URL_BASE}/{fname}", fp)
    return pd.read_csv(fp)
