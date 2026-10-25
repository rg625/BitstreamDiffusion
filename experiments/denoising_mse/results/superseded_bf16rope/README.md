# Superseded: eager bf16 RoPE

Everything in this folder (and runs/denoising_mse/{main,tok}_bf16rope) was computed with the model run
eagerly under bf16 autocast, where models/sdt.py RotaryEmbedding computes its angles in bf16 and 511 of
the 1,024 positions collapse onto their neighbour. The production model (training and paper evaluation)
runs under torch.compile, which computes the angles in float32, so these numbers come from a
train/test mismatch. The "position bug" in the old report was an artefact of this setup.

Superseded by runs made with experiments/denoising_mse/rope_fix.py (default in mse_curves.py and
trajectories.py), which reproduces the paper's GenPPL (S 28.6 / 5.28, M 20.3 / 5.40 on 128 samples at
gamma = 0.13 with CoBit-S's schedule; paper: 27.06 / 5.26 and 18.47 / 5.378 with own schedules).
