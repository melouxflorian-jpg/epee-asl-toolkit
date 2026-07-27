"""Quickstart — load the corpus, inspect a clip, render it.

    python examples/01_quickstart.py
"""

import epee
from epee.visualize import plot_clip_strip

meta = epee.load_metadata()
print(f"{len(meta)} clips · {meta.signer_id.nunique()} signers · "
      f"{meta.n_frames.sum():,} frames · "
      f"{meta.n_frames.sum() / meta.fps.mean() / 60:.1f} min signed\n")

clip = epee.load_clip("clerc_v02_002")
print(f"{clip.clip_id}  [{clip.signer_id}]  “{clip.text_en}”")
print(f"  {clip.n_frames} frames @ {clip.fps} fps  ({clip.duration_s:.1f}s)")
print(f"  keypoints {clip.keypoints.shape}   SLR subset {clip.slr_keypoints().shape}\n")

for seg in clip.segments:
    a, b = clip.segment_frames(seg)
    print(f"  {seg['gloss']:<12} {seg['start']:.1f}s → {seg['end']:.1f}s   frames {a}-{b}")

out = plot_clip_strip(clip, path="quickstart_strip.png")
print(f"\nWrote {out}")
