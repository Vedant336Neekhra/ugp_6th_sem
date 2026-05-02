# DJSCC — ADJSCC-CSINet+ Reproduction and Variants

This directory contains the implementation of the ADJSCC-CSINet+ deep
joint source–channel coding pipeline for FDD massive-MIMO CSI feedback
(Xu *et al.*, *IEEE J. Sel. Areas Commun.*, 2023), reproduced and
extended for the undergraduate project report
[`../UGP_report___Vedant_Neekhra-2.pdf`](../UGP_report___Vedant_Neekhra-2.pdf).

The dataset is generated with QuaDRiGa under the 3GPP TR 38.901
open-indoor scenario (`f_d = 5.2 GHz, f_u = 5.4 GHz, N_t = 32, N_c = 256`).
Train / val / test splits contain 100 000 / 30 000 / 20 000 sample pairs.

## Pipeline at a glance

```
H_d  ──ATN──►  T  ──CSINet+ Encoder + AF──►  c ∈ R^M
                                              │
                                              ▼
                                    real → complex symbols + power norm
                                              │
                                              ▼
                                  OFDM AWGN channel (uplink CSI, MRC)
                                              │
                                              ▼
                                    complex → real (C2R)
                                              │
                                              ▼
H_d_hat ◄─STN─  T_hat  ◄─CSINet+ RefineNet Decoder + AF─  c_hat
```

Every variant below shares this skeleton; they differ only in the
parts each one is highlighted to study.

## Notebooks

The notebook flow used everywhere mirrors `4 feb/ADJSCC-CSInet+.ipynb`:

> dataset → AF module → ATN → encoder → real↔complex + power norm
> → wireless channel → C2R → decoder → STN → training loop.

The exploratory notebooks in the dated folders (`26 jan/`, `27 jan/`,
…, `6 feb/`) are the original scratch implementations that became the
final pipeline.  Every `checkpoints_*` folder now contains a
**standalone notebook** that loads the variant-specific code from the
matching `.py` script in cell-by-cell order.

## Folder index

### Reproduction baselines

| Folder | Notebook | What it is |
| --- | --- | --- |
| `checkpoints_paper_baseline/` | `adjscc_paper_baseline.ipynb` | Paper-matched recipe at `k=32` (CR=32). Plain MSE loss, ReduceLROnPlateau scheduler. The "ground truth" we tuned away from. |
| `checkpoints_tuned_k32/` | `adjscc_tuned_k32.ipynb` | Same pipeline as the paper baseline but with the tuned hyper-parameters (mixed MSE+NMSE loss, weight decay `1e-5`, gradient clipping). |
| `checkpoints_tuned_k64/` | `adjscc_tuned_k64.ipynb` | Doubles the feedback bandwidth to `k=64` (CR=16). This single change buys ≈ 4 dB and becomes the standard for every later variant. |
| `checkpoints_tuned_k64_08_02/` | `adjscc_tuned_k64_loss08_02.ipynb` | Tuned-`k64` with the loss weights shifted to 0.8 MSE / 0.2 NMSE — anchors more strongly on absolute reconstruction magnitude. |
| `checkpoints_tuned_k64_bs50/` | `adjscc_tuned_k64_bs50.ipynb` | Tuned-`k64` with batch size 50. Used to study how mini-batch size affects the train–val NMSE gap. |
| `checkpoints_tuned_k64_bs50_08_02/` | `adjscc_tuned_k64_bs50_loss08_02.ipynb` | Both changes together: BS 50 *and* loss weights 0.8/0.2. |
| `checkpoints_fixed/` | `resume_training.ipynb` | Resume / fix-up script.  Restarts training from the most recent good checkpoint with a simplified MSE-only loop and a built-in NMSE-vs-SNR evaluator. |

### Loss ablations

| Folder | Notebook | What it is |
| --- | --- | --- |
| `checkpoints_k64_nmse_only/` | `adjscc_k64_nmse_only.ipynb` | Trains with linear NMSE as the *only* objective. Confirms the report's finding that scale-free losses regress to worse optima. |
| `checkpoints_k64_correlation_loss/` | `adjscc_k64_correlation_loss.ipynb` | Replaces MSE with a differentiable spectral-correlation (cosine-similarity) loss combined with NMSE. Encourages directional alignment. |
| `checkpoints_k64_correlation_mse_loss/` | `adjscc_k64_correlation_mse_loss.ipynb` | Adds the spectral-correlation term on top of the standard MSE+NMSE mix — three loss signals jointly minimised. |

### Numbered variants (Var1–Var5: structured architectural sweeps)

| Folder | Notebook | What it changes vs tuned-`k64` |
| --- | --- | --- |
| `checkpoints_warmup_cosine_largebatch/` | `var1_warmup_cosine_largebatch.ipynb` | **Var1.** Replaces `ReduceLROnPlateau` with a 30-epoch linear warmup followed by `CosineAnnealingLR`, and raises the batch size to 400 — eliminates validation spikes. |
| `checkpoints_stratified_snr/` | `var2_stratified_snr.ipynb` | **Var2.** Stratified SNR sampling (one SNR per equal-width bucket) on training; deterministic SNR schedule on validation. Makes val NMSE a stable early-stopping signal. |
| `checkpoints_deeper_atn_stn/` | `var3_deeper_atn_stn.ipynb` | **Var3.** Wider, deeper analysis / synthesis transforms: 3 layers × 16 ch → 4 layers × 32 ch. Wins early but overfits later. |
| `checkpoints_snr_curriculum/` | `var4_snr_curriculum.ipynb` | **Var4.** Three-phase SNR curriculum (±5 → ±8 → ±10 dB). Discrete phase boundaries cause sharp NMSE spikes. |
| `checkpoints_deeper_decoder_regularized/` | `var5_deeper_decoder_regularized.ipynb` | **Var5 — best ADJSCC reproduction (−7.5 dB).**  RefineNet stack 5 → 8 blocks, `Dropout2d(p=0.10)` per block, weight decay 1e-5 → 5e-5. |

### Lettered variants (VarB–VarE: advanced regularisation / alternatives)

| Folder | Notebook | What it changes |
| --- | --- | --- |
| `checkpoints_mean_flow_corrected/` | `varB_mean_flow_corrected.ipynb` | **VarB.** Replaces the deterministic CSINet+ decoder with a one-step *MeanFlow* generative decoder (MeanFlow Identity + forward-mode JVP). Restores gradient flow through the STN and uses correct one-step sampling direction. |
| `checkpoints_spectral_atn_stn/` | `varC_spectral_atn_stn.ipynb` | **VarC.** Var3's deeper ATN/STN, plus spectral-norm on every conv / transposed-conv and `Dropout2d(p=0.1)` after every hidden activation. Per-group weight decay (1e-4 for ATN/STN, 1e-5 elsewhere). |
| `checkpoints_smooth_curriculum/` | `varD_smooth_curriculum.ipynb` | **VarD.** Continuous half-cosine SNR-range annealing from ±5 dB to ±10 dB over 300 epochs — eliminates Var4's discrete phase shocks. |
| `checkpoints_adaptive_gate/` | `varE_adaptive_gate.ipynb` | **VarE.** Adds a learned per-symbol soft gate after the encoder FC, conditioned on SNR.  At low SNR most gates close, concentrating power into surviving symbols; at high SNR all gates open. |
| `checkpoints_combined_v1v5_mse/` | `var_combined_v1v5_mse.ipynb` | **Var1 + Var5 with MSE-only.** Combines the warmup-cosine-largebatch recipe with the deeper-dropout decoder, dropping the NMSE term entirely. |

## Key results from the report

Reproduced ADJSCC-CSINet+ test NMSE (best run = `checkpoints_deeper_decoder_regularized`, k = 64, CR = 16):

| SNR (dB) | −10 | −5 | 0 | 5 | 10 |
| --- | --- | --- | --- | --- | --- |
| Paper | −7.5 | −10.5 | −12.0 | −13.0 | −13.5 |
| Ours (Var5) | −6.5 | −7.0 | −7.2 | −7.3 | −7.3 |

The remaining gap is attributed primarily to (i) QuaDRiGa-vs-paper
dataset generation differences, (ii) undisclosed paper hyper-parameters,
(iii) limited training budget, and (iv) a high-SNR ceiling set by
decoder capacity.

## Cross-cutting findings (Section 9 of the report)

1. **Normalisation gates everything else** — min-max normalisation
   diverges; per-element Gaussian normalisation lets the model
   converge.
2. **Scale-free losses regress to worse optima** — an absolute MSE
   term is required to anchor reconstruction magnitude.
3. **Cosine LR consistently beats `ReduceLROnPlateau`**, which drops
   the LR prematurely after early-training noise spikes.
4. **Best-checkpoint NMSE alone is misleading** — final-epoch NMSE
   must also be reported to verify stability.
5. **Capacity is the dominant lever** — once the front end / loss /
   optimiser are correctly configured, scaling width and depth gives
   the largest NMSE wins (especially clear in the CrossNet leg of
   the project, where doubling capacity gave order-of-magnitude jumps
   from −16 → −24 → −27 dB).

## How to run a variant

Each notebook is a faithful, simplified view of the matching `.py`.
For full training (with checkpointing, plotting, resume support, and
the SNR-sweep evaluator) prefer the script:

```bash
# Example: reproduce the best run (Var5)
cd checkpoints_deeper_decoder_regularized/
python var5_deeper_decoder_regularized.py --train --evaluate
```

The notebooks are intended as a self-contained reference for what
each variant *does*; the `.py` files are what was actually trained.
