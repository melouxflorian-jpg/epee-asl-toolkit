"""Rendering utilities for Épée keypoint sequences."""

from __future__ import annotations

import numpy as np

from .loader import Clip

# MediaPipe hand topology, applied to both hands (0-20 and 21-41).
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# Upper body, in dataset index space (42-53 = MediaPipe Pose 11-22).
_BODY_EDGES = [(42, 43), (42, 44), (44, 46), (43, 45), (45, 47)]


def _edges() -> list[tuple[int, int]]:
    edges = [(a, b) for a, b in _HAND_EDGES]
    edges += [(a + 21, b + 21) for a, b in _HAND_EDGES]
    return edges + _BODY_EDGES


def draw_frame(
    ax,
    kp_frame: np.ndarray,
    color: str = "#1f77b4",
    label: str | None = None,
    lower_body: bool = False,
):
    """Draw one frame's skeleton onto a matplotlib axis.

    kp_frame: (128, 3) array for a single frame.
    lower_body: source clips are framed waist-up, so indices 54-63 are
        extrapolations. Hidden by default.
    """
    visible = ~np.all(kp_frame[:, :2] == 0, axis=1)
    if not lower_body:
        visible[54:64] = False
    for a, b in _edges():
        if visible[a] and visible[b]:
            ax.plot(
                [kp_frame[a, 0], kp_frame[b, 0]],
                [kp_frame[a, 1], kp_frame[b, 1]],
                color=color, linewidth=1.4, alpha=0.85, solid_capstyle="round",
            )
    head = kp_frame[92:128]
    head = head[~np.all(head[:, :2] == 0, axis=1)]
    if len(head):
        ax.plot(
            np.append(head[:, 0], head[0, 0]),
            np.append(head[:, 1], head[0, 1]),
            color=color, linewidth=1.0, alpha=0.5,
        )
    pts = kp_frame[visible]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, color=color, alpha=0.6, linewidths=0)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=9)


def plot_clip_strip(clip: Clip, n: int = 6, path: str = "clip_strip.png"):
    """Render n evenly spaced frames of a clip as a filmstrip."""
    import matplotlib.pyplot as plt

    idx = np.linspace(0, clip.n_frames - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6))
    for ax, i in zip(np.atleast_1d(axes), idx):
        draw_frame(ax, clip.keypoints[i], label=f"f{i}")
    fig.suptitle(f"{clip.clip_id} · {clip.signer_id} · “{clip.text_en}”", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_signer_comparison(clips: list[Clip], at: float = 0.5, path: str = "signers.png"):
    """Same phrase, all signers, at the same relative point in the clip.

    This is the picture that makes inter-signer variability legible: identical
    phrase, four bodies, four different realisations.
    """
    import matplotlib.pyplot as plt

    from .loader import normalize_by_shoulders

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, len(clips), figsize=(2.4 * len(clips), 3.0))
    for ax, clip, color in zip(np.atleast_1d(axes), clips, palette):
        i = int(at * (clip.n_frames - 1))
        # Shoulder-normalized so body size and camera framing cancel out and
        # what remains on screen is the signing itself.
        kp = normalize_by_shoulders(clip.keypoints)
        draw_frame(ax, kp[i], color=color, label=clip.signer_id)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(2.9, -2.4)
    fig.suptitle(f"“{clips[0].text_en}” - same phrase, {len(clips)} Deaf signers", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
