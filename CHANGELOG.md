# Changelog

## v2.1 — curated results release (2026-07)

- **RESULTS.md**: full multi-seed experimental record (supervision-density
  methodology, length generalization to 8x training length, interference
  analysis, DPFP capacity-axis win, honest negative-results ledger).
- Repo cleanup: v1-era diagnostic scripts (`debug_memory.py`, `eval_*.py`),
  Turkish-language internal review documents and superseded runners removed;
  experiment scripts use portable checkpoint paths (`HFP_CKPT_DIR`).
- New experiment scripts: `length_gen.py`, `interference_eval.py`.


## v2.0 — full architecture rewrite (2026-07)

Complete replacement of the v1 codebase. Old code is preserved under the
`v1-legacy` git tag. **Public claims were recalibrated**: v1 documentation
claimed "infinite context", "eliminates O(N²) KV-cache VRAM" (the KV-cache is
O(N)), "prevents hallucinations" and presented physics analogies (Ryu-Takayanagi
bound, Witten propagator, 5D curvature) as load-bearing mechanisms. None of
those claims survive scrutiny and they are withdrawn. What HFP actually is —
a windowed-attention + recurrent-linear-attention-memory LM with a selectable
retention law — is documented honestly in the README.

### Architecture / correctness

- **K2** — causal chunkwise linear attention: memory params now receive gradients
  in single-forward training; train/generate decay semantics consistent.
- **K1** — MQAR label alignment fixed (old double-shift produced NaN loss).
- **K7 (root cause)** — embedding×√d + PE×0.3: raw positional encoding was
  ~35× louder than token content, making content-based recall impossible.
- **K8** — binding conv (depthwise causal, Q/K path). Ablation shows it is *not*
  load-bearing at ≥2 layers; kept as a cheap standard component.
- **K5** — true sliding-window attention (`local_window`); ring-buffer zero-slot
  masking (D2) fixed.
- **K3** — TunnelingDropout removed (cross-batch leakage); standard dropout.
- Multi-scale decay init (λ 0.90–0.999 per channel); K4 ring-buffer capacity fix;
  D1 weight tying (with `_tied_weights_keys` + embedding accessors so
  `from_pretrained` re-ties correctly); D3 silent half() removed.

### New mechanisms (each opt-in, baseline stays clean)

- **`decay_mode="cubic_flux"`** — state-magnitude-dependent retention
  `λ_t = 1/√(1+2η·z²)`, an exact discretization of `dθ/dτ = −η·θ³`.
- **`key_feature_map="dpfp"`** — DPFP capacity axis (`key_dim = 2H·nu`).
- **`aux_ortho_weight`** — EntangledFFN orthogonality regularizer, previously
  dead code, now optionally wired to the loss (default 0.0 = off).
- Physics-inspired aux hooks (`hfp_config.py`) all default **off**.

### Verification

- `smoke_test.py`: gradient flow, MQAR alignment, chunk-consistency of both
  retention modes (and DPFP), cached generation.
- Independent review (`INCELEME_RAPORU.md`, `review_scripts/`): O(1) state,
  causal no-leakage, exact ODE discretization, chunkwise==naive recurrence all
  verified; the "cubic > exp recall" headline did **not** replicate under
  LR/seed controls and is treated as unproven.

### Licensing

License remains **AGPL-3.0** (unchanged from v1). v1 releases stay available
under AGPL-3.0 via the `v1-legacy` tag.
