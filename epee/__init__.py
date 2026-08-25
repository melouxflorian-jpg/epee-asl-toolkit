"""epee - toolkit for the CLERC Épée ASL keypoint corpus.

Dataset: https://huggingface.co/datasets/CLERC-DATA/epee
"""

from .loader import (
    Clip,
    GRID_SIGNERS,
    SIGNERS,
    base_gloss,
    download_all,
    grid_coverage,
    load_clip,
    load_clips,
    load_metadata,
    normalize_by_shoulders,
    parallel_group,
    resample,
)

__version__ = "0.2.0"
__all__ = [
    "Clip", "GRID_SIGNERS", "SIGNERS", "base_gloss", "download_all",
    "grid_coverage", "load_clip", "load_clips", "load_metadata",
    "normalize_by_shoulders", "parallel_group", "resample",
]
