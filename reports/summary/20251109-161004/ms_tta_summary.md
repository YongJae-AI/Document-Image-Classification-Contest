# Multiscale TTA Leaderboard Summary

- Model: tf_efficientnet_b7_ns + SwinIR noise15
- TTA: rotations [0,90,180,270] + HFlip; no jitter; VFlip/Transpose disabled

- 413px: LB 0.9425
- 528px: LB 0.9458
- 613px: LB 0.9408

Conclusion: 528px is best in this sweep.
