"""Loading utilities for the CLERC Épée ASL keypoint corpus.

Dataset: https://huggingface.co/datasets/CLERC-DATA/epee
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ID = "CLERC-DATA/epee"

# Keypoint layout: 128 landmarks per frame, (n_frames, 128, 3)
LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
UPPER_BODY = slice(42, 54)
LOWER_BODY = slice(54, 64)
EYES_MOUTH = slice(64, 92)
HEAD_OVAL = slice(92, 128)

# All six signers in the release. GRID_SIGNERS are the five on the parallel
# grid; FOXTROT was recorded on a disjoint prompt set (a second, declarative
# register) so he has no rendering of the grid phrases.
SIGNERS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT")
GRID_SIGNERS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO")


@dataclass
class Clip:
    """One annotated ASL clip: keypoints plus gloss segmentation."""

    clip_id: str
    signer_id: str
    text_en: str
    fps: float
    n_frames: int
    segments: list[dict]
    keypoints: np.ndarray = field(repr=False)
    phrase_id: int | None = None
    parallel: bool = True

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    @property
    def glosses(self) -> list[str]:
        return [s["gloss"] for s in self.segments]

    def slr_keypoints(self) -> np.ndarray:
        """Drop lower-body landmarks. -> (n_frames, 118, 3)

        Source clips are framed waist-up, so indices 54-63 are MediaPipe
        extrapolations rather than observations. Most SLR pipelines drop them.
        """
        return np.concatenate([self.keypoints[:, :54], self.keypoints[:, 64:]], axis=1)

    def hands(self) -> np.ndarray:
        """Both hands only. -> (n_frames, 42, 3)"""
        return self.keypoints[:, 0:42]

    def segment_frames(self, segment: dict) -> tuple[int, int]:
        """Frame index range [start, end) for a gloss segment."""
        return int(segment["start"] * self.fps), int(segment["end"] * self.fps)

    def segment_keypoints(self, segment: dict) -> np.ndarray:
        a, b = self.segment_frames(segment)
        return self.keypoints[a:b]


def base_gloss(gloss: str) -> str:
    """Collapse variant suffixes: SIGN_2 -> SIGN.

    Variants are alternative ways to sign the same concept (different
    handshape, location or movement), not intensity markers. Directional
    suffixes (GO_LEFT, POINTER_RIGHT) carry meaning and are preserved.
    """
    return re.sub(r"_\d+$", "", gloss)


@lru_cache(maxsize=2048)
def _fetch(filename: str) -> Path:
    """Download one file from the Hub, cached locally after the first call."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))


def download_all() -> Path:
    """Fetch the whole dataset at once (~2400 files). Returns the local root.

    Only needed for full-corpus work; load_clip() fetches lazily per clip.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))


def load_metadata():
    """Master index as a pandas DataFrame, one row per clip."""
    import pandas as pd

    return pd.read_csv(_fetch("metadata.csv"))


def load_clip(clip_id: str) -> Clip:
    """Load a single clip by id, e.g. 'clerc_v03_0001'."""
    with open(_fetch(f"annotations/{clip_id}.json")) as fh:
        ann = json.load(fh)
    kp = np.load(_fetch(f"keypoints/{clip_id}.npy"))
    return Clip(
        clip_id=ann["clip_id"],
        signer_id=ann["signer_id"],
        text_en=ann["text_en"],
        fps=float(ann["fps"]),
        n_frames=int(ann["n_frames"]),
        segments=ann["segments"],
        keypoints=kp,
        phrase_id=ann.get("phrase_id"),
        parallel=bool(ann.get("parallel", True)),
    )


def load_clips(signer: str | None = None, limit: int | None = None) -> list[Clip]:
    """Load clips, optionally filtered to one signer."""
    meta = load_metadata()
    if signer is not None:
        meta = meta[meta.signer_id == signer.upper()]
    if limit is not None:
        meta = meta.head(limit)
    return [load_clip(cid) for cid in meta.clip_id]


def parallel_group(phrase_id: int) -> list[Clip]:
    """Every rendering of one phrase, one clip per signer on the parallel grid.

    Pairing goes through the ``phrase_id`` column of ``metadata.csv``, not
    through arithmetic on clip ids. Block offsets change between releases and
    the grid is not perfectly rectangular, so arithmetic silently returns the
    wrong clips; the column does not.

    Returns clips ordered by signer. Most phrases come back with five, a few
    with four: see ``grid_coverage()``.
    """
    meta = load_metadata()
    rows = meta[(meta.parallel == "yes") & (meta.phrase_id == phrase_id)]
    if rows.empty:
        raise ValueError(
            f"no parallel-grid phrase with phrase_id={phrase_id}; "
            f"valid ids: {sorted(meta[meta.parallel == 'yes'].phrase_id.unique())[:5]}..."
        )
    rows = rows.sort_values("signer_id")
    return [load_clip(cid) for cid in rows.clip_id]


def grid_coverage():
    """How many signers rendered each grid phrase, as a DataFrame.

    The grid is near-rectangular, not rectangular: one legacy phrase is missing
    a signer, and one was recorded by a single signer. Check coverage before
    assuming a fixed group size.
    """
    meta = load_metadata()
    grid = meta[meta.parallel == "yes"]
    return (grid.groupby("phrase_id")
                .agg(n_signers=("signer_id", "nunique"),
                     text_en=("text_en", "first"))
                .reset_index())


def normalize_by_shoulders(kp: np.ndarray) -> np.ndarray:
    """Translate to shoulder midpoint, scale by shoulder width.

    Removes camera framing and body-size differences so that what remains is
    signing behaviour. This is the preprocessing used in the released
    cross-signer benchmark.
    """
    left_sh, right_sh = kp[:, 42, :2], kp[:, 43, :2]
    center = (left_sh + right_sh) / 2.0
    width = np.linalg.norm(right_sh - left_sh, axis=-1, keepdims=True)
    width = np.where(width < 1e-6, 1.0, width)
    out = kp.copy().astype(np.float32)
    out[:, :, :2] = (out[:, :, :2] - center[:, None, :]) / width[:, None, :]
    return out


def resample(seq: np.ndarray, n: int = 24) -> np.ndarray:
    """Resample a variable-length sequence to n frames by linear interpolation."""
    if len(seq) == 0:
        return np.zeros((n, *seq.shape[1:]), dtype=np.float32)
    if len(seq) == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0, 1, len(seq))
    dst = np.linspace(0, 1, n)
    flat = seq.reshape(len(seq), -1)
    out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape(n, *seq.shape[1:]).astype(np.float32)
