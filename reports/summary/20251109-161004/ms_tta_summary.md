# Multiscale TTA Leaderboard Summary

- Model: tf_efficientnet_b7_ns + SwinIR noise15
- TTA: rotations [0,90,180,270] + HFlip; no jitter; VFlip/Transpose disabled

- 413px: LB 0.9425
- 528px: LB 0.9458
- 613px: LB 0.9408

Conclusion: 528px is best in this sweep.


OOM/Safety Notes
- 528px TTA (SwinIR dn15 + [0,90,180,270] + HFlip) ran without OOM on RTX 3090 24GB.
- Recommend infer batch=2 for >=528px; avoid concurrent SwinIR+classifier jobs.
