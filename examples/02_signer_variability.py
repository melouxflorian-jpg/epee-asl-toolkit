"""Inter-signer variability - the thing this corpus exists to measure.

Five of the six Deaf signers sign a shared grid of 201 phrases. That structure
lets you isolate signer variation from phrase variation, which single-signer
datasets cannot do. Clips are paired through ``phrase_id``.

    python examples/02_signer_variability.py
"""

import numpy as np

import epee
from epee.visualize import plot_signer_comparison

PHRASE = 7

clips = epee.parallel_group(PHRASE)
print(f"Phrase #{PHRASE}: “{clips[0].text_en}” - {len(clips)} signers\n")

for c in clips:
    print(f"  {c.signer_id:<8} {c.duration_s:5.2f}s  "
          f"{len(c.segments)} segments  {' | '.join(c.glosses)}")

durations = np.array([c.duration_s for c in clips])
print(f"\nDuration spread: {durations.min():.2f}s → {durations.max():.2f}s "
      f"({durations.max() / durations.min():.1f}× between the fastest and slowest signer)")

# How much of the signing space each signer uses, after removing body size and
# camera framing. Signing space is one of the clearest inter-signer differences.
print("\nSigning-space extent (shoulder-normalized, hands only):")
for c in clips:
    kp = epee.normalize_by_shoulders(c.keypoints)[:, 0:42, :2]
    kp = kp[~np.all(kp == 0, axis=-1)]
    print(f"  {c.signer_id:<8} horizontal {np.ptp(kp[:, 0]):.2f}   vertical {np.ptp(kp[:, 1]):.2f}")

out = plot_signer_comparison(clips, at=0.5, path="signer_variability.png")
print(f"\nWrote {out}")
print("\nThis is why single-signer training does not transfer: the same phrase,")
print(f"signed by {len(clips)} native signers, produces {len(clips)} measurably different sequences.")
