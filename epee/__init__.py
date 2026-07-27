"""epee — toolkit for the CLERC Épée ASL keypoint corpus.

Dataset: https://huggingface.co/datasets/CLERC-DATA/epee
"""

from .loader import (
    Clip,
    SIGNERS,
    base_gloss,
    download_all,
    load_clip,
    load_clips,
    load_metadata,
    normalize_by_shoulders,
    parallel_group,
    resample,
)

__version__ = "0.1.0"
__all__ = [
    "Clip", "SIGNERS", "base_gloss", "download_all", "load_clip", "load_clips",
    "load_metadata", "normalize_by_shoulders", "parallel_group", "resample",
]
