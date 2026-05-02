# CrossNet on QuaDRiGa — Variant Notebooks

This directory contains 22 variants of the CrossNet auto-encoder for CSI feedback,
trained on the open-indoor QuaDRiGa dataset (`{train,val,test}_data.mat`,
HDF5 / `csi_dl` key, 32 × 32 angular-delay matrices).

Each `CrossNet-*` folder contains a **standalone Jupyter notebook**
(`<variant>.ipynb`) that mirrors the simple flow of `CrossNet-V.ipynb` and
`CrossNet-H.ipynb`: imports → seed → device → dataset → model modules →
optimiser/scheduler → training loop. The notebooks do not import from
`crossnet_quadriga.py`; everything they need is inlined. The original
`<variant>.py` scripts (which import the shared library) are kept for reference.

The notebooks are produced by `generate_notebooks.py`. Re-run that script to
regenerate them after editing the variant table.

---

## Architecture axes

Three orthogonal choices define a variant name:

| Axis | Values | Meaning |
| --- | --- | --- |
| **Orientation** | `V`, `H` | Token layout. **V**: 64 tokens × 32-dim along subcarrier-stacked rows; **H**: 32 tokens × 64-dim along antenna-stacked columns. The cross-attention exchanges information between subcarrier and antenna streams. |
| **Transform path** | `ATN`, `DFT` | **ATN**: learnable convolutional analysis / synthesis transforms (with batchnorm + PReLU, stride (2, 1)). **DFT**: fixed 2-D DFT into the angular-delay domain, keeping the top-32 dominant delay rows by training-set energy. |
| **Normalisation** | `Gaussian`, `MinMax` | **Gaussian**: divide by a single global std fitted on the training split (sigmoid output disabled). **MinMax**: per-channel min-max scaling to [0, 1] (sigmoid output enabled). |

The report shows that `MinMax` normalisation fails entirely on QuaDRiGa,
that learned `ATN` beats fixed `DFT`, and that `V`-orientation slightly beats
`H`. Once `ATN + Gaussian + V` is fixed, model capacity becomes the dominant
lever for further NMSE gains.

---

## Variant catalogue

### Baselines (8) — orientation × transform × normalisation

| Notebook | Orientation | Transform | Normalisation |
| --- | --- | --- | --- |
| `CrossNet-V_ATN_Gaussian` | V | learned ATN/STN | Gaussian |
| `CrossNet-V_ATN_MinMax` | V | learned ATN/STN | min-max + sigmoid |
| `CrossNet-V_DFT_Gaussian` | V | DFT (top-32 rows) | Gaussian |
| `CrossNet-V_DFT_MinMax` | V | DFT (top-32 rows) | min-max + sigmoid |
| `CrossNet-H_ATN_Gaussian` | H | learned ATN/STN | Gaussian |
| `CrossNet-H_ATN_MinMax` | H | learned ATN/STN | min-max + sigmoid |
| `CrossNet-H_DFT_Gaussian` | H | DFT (top-32 rows) | Gaussian |
| `CrossNet-H_DFT_MinMax` | H | DFT (top-32 rows) | min-max + sigmoid |

All baselines use the standard recipe: 1000 cosine epochs, AdamW
(β = (0.8, 0.98), wd = 1e-3), LR 2e-3 → η_min 5e-5, batch size 128, MSE loss.

### Optimiser / training-budget sweeps (7) — V-ATN-Gaussian only

| Notebook | What changes vs the baseline |
| --- | --- |
| `..._BS64_LR1e-3` | Smaller batch (64), lower LR (1e-3). |
| `..._BS128_LR2e-3` | Canonical batch / LR with extended 1500-epoch cosine schedule. |
| `..._BS256_LR3e-3` | Larger batch (256), higher LR (3e-3) — linear-scaling check. |
| `..._LR1e-3_WD3e-4` | Conservative LR (1e-3) with lighter weight decay (3e-4). |
| `..._WD1e-4` | Reduced weight decay (1e-4) over 1500 epochs. |
| `..._LongCosine1500` | Same recipe, 1500-epoch cosine, η_min = 1e-5. |
| `..._ReduceLROnPlateau` | Plateau scheduler (factor 0.5, patience 50) on validation loss instead of cosine. |

### Architecture variants (4) — V-ATN-Gaussian with positional embeddings

These wrap a `MiniCrossNetVPE` backbone (learnable subcarrier / antenna PE
on both encoder and decoder). The first row matches the baseline capacity;
the next three scale width / depth / codeword.

| Notebook | d\_model | heads | self / cross blocks | codeword | Note |
| --- | --- | --- | --- | --- | --- |
| `..._PE` | 16 | 2 | 1 / 1 | 512 | PE-only ablation; isolates the effect of positional information. |
| `..._Big` | 32 | 4 | 2 / 2 | 512 | One width + depth doubling. **Rank 3** in the report (best NMSE −16.05 dB). |
| `..._XL` | 64 | 8 | 2 / 2 | 512 | Pure width scaling. **Rank 2** (best NMSE −23.62 dB). |
| `..._XXL` | 64 | 8 | 3 / 3 | 1024 | Width + depth + doubled codeword. **Rank 1** (best NMSE −27.31 dB). |

### Loss and wrapper variants (2)

| Notebook | What changes |
| --- | --- |
| `..._NMSELoss` | Same backbone as `..._PE`; replaces MSE with per-sample NMSE. Tests scale-free vs absolute reconstruction objectives. The report finds scale-free losses regress to worse optima. |
| `..._Residual` | Long-skip residual: `STN(CrossNet(ATN(x))) + STN_skip(ATN(x))`. CrossNet learns to refine; an auxiliary MSE term in the ATN-transformed domain feeds it a bottleneck-free gradient. |

### DFT row-selection variant (1)

| Notebook | What changes |
| --- | --- |
| `CrossNet-V_DFT_Gaussian_FirstRows` | DFT path keeping the **first** 32 delay rows (paper-style truncation) instead of the globally dominant ones. Uses the `MiniCrossNetVPE` backbone and an auxiliary MSE in the angular-delay domain. |

---

## Key findings (from `UGP_report___Vedant_Neekhra-2.pdf`)

- **Normalisation gates everything else.** Min-max + sigmoid simply does not
  converge on QuaDRiGa; per-element Gaussian normalisation does.
- **Learned ATN beats fixed DFT** front-ends; V-orientation slightly beats H.
- **Capacity is the dominant lever** once the front-end and optimiser are
  tuned — `Big → XL → XXL` produces a 11+ dB NMSE swing while everything else
  is held fixed.
- **Cosine annealing beats ReduceLROnPlateau**; plateau drops the LR
  prematurely after noise spikes.
- **Scale-free losses (NMSE-only) regress** by 1–3 dB; absolute MSE is needed
  to anchor reconstruction magnitude.
- **Positional embeddings** show minor detriments on the baseline but are
  crucial for stability of the larger XL / XXL backbones.

The top final result is **CrossNet-V_ATN_Gaussian_XXL at −27.31 dB test NMSE
(epoch 911)**.

---

## How to run a notebook

1. Place the QuaDRiGa `train_data.mat`, `val_data.mat`, `test_data.mat` files
   alongside the variant folder, or set absolute paths in the
   `## Variant configuration` cell (variables `train_path`, `val_path`,
   `test_path`).
2. Open the notebook in Jupyter or VS Code.
3. Run cells top-to-bottom. Outputs (CSV histories and best-checkpoint
   `.pth`) are written to `outputs/<variant_name>/`.
