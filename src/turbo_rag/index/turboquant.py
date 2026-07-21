"""Stage 3 — the main event. Do NOT start this until Stages 1-2 are done and
your eval scoreboard runs with one command.

Implementation order (from PLAN.md, expanded):

1. Random rotation. A random orthogonal matrix R (QR-decompose a Gaussian
   matrix, or a fast Hadamard-based rotation later). After rotation, each
   coordinate of a unit vector is approximately Beta-distributed and
   near-independent — this is what makes a *fixed* per-coordinate scalar
   quantizer near-optimal with no training data (the "data-oblivious" claim).

2. Per-coordinate optimal scalar quantizer at b bits (start b=4, then 2, 1).
   Validate on synthetic data FIRST: quantize random unit Gaussian vectors,
   measure MSE distortion, compare against the paper's reported rates. Do not
   touch real embeddings until synthetic distortion matches the paper.

3. Inner-product correction: MSE quantizer alone gives biased inner-product
   estimates; add the 1-bit QJL transform on the residual to debias.

4. Bit-packing (np.packbits or manual) so memory_bytes() reports the honest
   compressed size, not float32-pretending-to-be-bits.

Must satisfy the VectorIndex protocol in base.py — then Stage 4 is a for-loop.

Paper: https://arxiv.org/abs/2504.19874  — expect 2-3 careful reads.
"""
