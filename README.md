# PaTFuse

A compact traffic-flow forecasting model that fuses **patch-level temporal attention** with an
**adaptive periodic (frequency-domain) branch**. Single-file implementation in PyTorch, with
built-in ablations and optional hyperparameter search.

## Method

Given a window of `Th` historical steps (5-minute sampling), the model predicts the next `Tf`
steps of traffic flow for every sensor.

**Input representation.** The target series is split into `S = Th / P` patches, each linearly
projected into a token. An extra learnable global token is appended, giving `S + 1` tokens per
sensor. The side-information sequence `V` contains every variate of every sensor (via a linear
projection over time) plus calendar features (minute, hour, day-of-week, day-of-month,
day-of-year) for the observed window.

**PTIM — patch temporal interaction module** (`PTIMLayer`, repeated `R` times):

- **PTA** — multi-head self-attention over the patch tokens, computed independently per sensor.
- **CGCA** — multi-head cross-attention in which the global token attends to `V`, letting each
  sensor read from all other sensors and from calendar context.
- A feed-forward block with pre-norm residuals.

**APEFM — adaptive periodic emphasis module** (`APEFM`). The target series is embedded
per-sensor, a soft selector reweights the embedding channels, and an rFFT is taken over time.
The dominant non-DC frequency `k*` is located and its share of the total power `c` is used to
set a per-sensor gate `alpha = sigmoid(w_a * c + b_a)`. A softmax kernel around `k*` weights a
learned per-frequency gain, which is applied to the spectrum before the inverse transform and a
learned mixing over channels. The result is projected back into `S + 1` tokens.

**Fusion.** The PTIM branch produces a candidate `tanh(g1(XT))`; APEFM produces a gate
`sigmoid(g2(XF))`; the two are multiplied elementwise and passed through a
**DSFFN** (dual-stream FFN: a per-position transform concatenated with the position-mean
context). A linear head flattens the `(S + 1) x d` grid into the `Tf` predictions.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run on synthetic data (no download required):

```bash
python patfuse.py --synthetic
```

Run on PEMS-style data — an `.npz` file containing a `data` array of shape `(T, N)` or
`(T, N, F)`, sampled every 5 minutes, with flow as channel 0:

```bash
python patfuse.py --dataset PEMS04 --data_path PEMS04.npz --epochs 100
```

Available datasets with known start timestamps: `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`.
Splits are chronological 60 / 20 / 20, and normalization statistics are fit on the training
portion only.

Select an ablation via `--variant`:

```
full  wo_apefm  wo_ptim  wo_pta  wo_cgca  concat_fusion  wo_period
wo_dsffn  ptim_backbone  shared_gate  static_correction
```

Enable the hyperparameter search over width and FFN size (a BCO-style population search):

```bash
python patfuse.py --data_path PEMS08.npz --bco
```

## Key options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--Th`, `--Tf` | 12, 12 | history / horizon length (5-min steps) |
| `--P` | 4 | patch length, must divide `Th` |
| `--d`, `--d_ff` | 64, 64 | model width, FFN width |
| `--R` | 2 | number of PTIM layers |
| `--heads` | 8 | attention heads |
| `--tau` | 1.0 | softmax temperature around the dominant frequency |
| `--lr`, `--wd` | 5e-3, 0.0 | Adam learning rate, weight decay |
| `--bco` | off | run the hyperparameter search before final training |

Metrics reported: MAE, RMSE, MAPE, plus per-step MAE. Model selection uses validation MAE.

## Notes

- The best checkpoint by validation MAE is restored before the test evaluation.
- `--device` defaults to CUDA when available.
- This repository is the model code only; benchmark datasets are not included.

## License

No license has been specified.
