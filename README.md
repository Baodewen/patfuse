# PaTFuse

PaTFuse is a traffic-flow forecasting model that combines **patch-level temporal attention**
with an **adaptive periodic branch** in the frequency domain. It forecasts the next `Tf` steps
for every sensor from `Th` historical steps of 5-minute data, using a patch-temporal
interaction module (PTA + CGCA), a learnable frequency-gain branch (APEFM), and a small
fusion FFN. The whole model lives in one file, `patfuse.py`.

## Quick start

```bash
pip install -r requirements.txt
```

Run on synthetic data — no download needed:

```bash
python patfuse.py --synthetic
```

Run on PEMS-style data. The `.npz` file must contain a `data` array of shape `(T, N)` or
`(T, N, F)`, sampled every 5 minutes, with flow as channel 0:

```bash
python patfuse.py --dataset PEMS04 --data_path PEMS04.npz --epochs 100
```

Supported `--dataset` names (for the calendar start timestamp): `PEMS03`, `PEMS04`, `PEMS07`,
`PEMS08`. Splits are chronological 60 / 20 / 20, and normalization is fit on the training split
only.

## Options

```bash
python patfuse.py --help
```

Common flags: `--variant` selects an ablation (`full`, `wo_apefm`, `wo_ptim`, `wo_pta`,
`wo_cgca`, `concat_fusion`, `wo_period`, `wo_dsffn`, `ptim_backbone`, `shared_gate`,
`static_correction`); `--bco` runs a hyperparameter search over model width before training;
`--Th` / `--Tf` / `--P` control window length and patch size; `--epochs`, `--lr`, `--batch` control
training.

Test output reports MAE, RMSE, MAPE and per-step MAE. The checkpoint with the best validation
MAE is restored before evaluation.

## License

MIT — see [LICENSE](LICENSE).
