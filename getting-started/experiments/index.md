# Local classification experiments

JSON submits the existing classification benchmark. The contract adds identity and lifecycle records. CLI `--benchmark` also measures inference.

```sh
uv sync --locked
uv run --no-sync python -m references.classification.experiment CONFIG.json runs/TRIAL
```

## Input v1

See [the example](https://github.com/frgfm/Holocron/blob/main/references/classification/experiment.example.json). Paths resolve from CONFIG. TRIAL resolves from the working directory and must not exist, including as a symlink. `runs/` is Git-ignored. Unknown fields/versions, invalid values and unsupported limits are rejected before training.

| Field               | Meaning                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `schema_version`    | Required; `1`                                                                                 |
| `recipe`            | Required; CLI names/defaults; required `data_path` contains ImageFolder `train/` and `val/`   |
| `initialization`    | Required; null for random weights or a local checkpoint path; strict weights-only loading     |
| `deployment_target` | Required; null or description; separate from training `recipe.device`, never measured         |
| `test_split`        | Optional held-out ImageFolder path; hashed only, never decoded or used for training/selection |

Recipe keys: `data_path`, `arch`, `device`, `seed`, `epochs`, `lr`, `batch_size`, `workers`, `grad_acc`, `opt`, `sched`, `weight_decay`, `norm_wd`, `label_smoothing`, `mixup_alpha`, `amp`, `train_crop_size`, `val_resize_size`, `val_crop_size`, `random_erase`, `wb`, `name`. Models use the classification catalog; preprocessing uses the existing Imagenette RGB/augment/crop/normalize recipe. Epochs are the only budget. Schedulers: `cosine` or `onecycle` (`div_factor=100`, `pct_start=0.1`; ten updates rejected). W&B needs optional training dependencies. Seed covers Python, NumPy and PyTorch; identical bits across runtimes are not promised.

## Output v1

| File              | Contents                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.json`     | Resolved inputs                                                                                                                                 |
| `provenance.json` | Source revision/dirty state before trial creation, packages, actual transforms, data/init hashes, optimizer defaults                            |
| `data.json`       | Matching class mapping, exact split membership and raw-byte SHA-256 hashes; content shared across splits rejected                               |
| `progress.jsonl`  | Append-only validation epoch events; no intra-epoch heartbeat                                                                                   |
| `result.json`     | Native benchmark runtime/recipe/data/models plus versioned state, timestamps, elapsed time, epoch, actual device, error/traceback and exit code |
| `checkpoint.pth`  | Selected weights; atomic saves preserve the prior file on failure; optimizer/scheduler/RNG state absent, not fully resumable                    |

`models[0]` separates `final_epoch`/`final_metrics` from `selected_epoch`/ `selected_metrics` and records `checkpoint`/`checkpoint_sha256`. Minimum validation loss selects; ties keep the earlier epoch. `metrics_split`/`metric_directions` label loss as minimize and acc1/acc5 as maximize (fractions; acc5 needs five classes). Non-finite validation loss fails. Available metrics/checkpoints survive failure.

| State         | Meaning / exit code                                             |
| ------------- | --------------------------------------------------------------- |
| `running`     | Setup/training; not proof of liveness                           |
| `completed`   | All requested epochs and a selected checkpoint; 0               |
| `failed`      | Setup/training/tracking exception re-raised; 1                  |
| `interrupted` | KeyboardInterrupt 130, SIGTERM 143, or Python's SystemExit code |

Split hashes use ordered sample JSON, UTF-8, sorted keys, separators `(',', ':')`; roots/mapping are separate. Initial weights hash sorted `[name, dtype, shape]` JSON and contiguous CPU tensor bytes. Re-encoded duplicates are not detected. Data/source are not snapshotted; absent Git metadata is null. Abrupt kills/storage failures may leave `running` or partial progress. Verify checkpoint hashes before use. No downloads, true resume, wall-time enforcement, search, export/latency or agent servers.

## Offline CPU example

Use a fresh trial name:

```sh
mkdir -p runs
cp references/classification/experiment.example.json runs/tiny.json
uv run --no-sync python - <<'PY'
from pathlib import Path
from PIL import Image
for i, split in enumerate(("train", "val", "test")):
    for j, label in enumerate(("dark", "light")):
        root = Path("runs/tiny-data") / split / label
        root.mkdir(parents=True, exist_ok=True)
        for sample in range(2):
            Image.new("RGB", (40, 40), (i * 60 + j * 20 + sample, 30, 90)).save(root / f"{sample}.png")
PY
OMP_NUM_THREADS=2 uv run --no-sync python -m references.classification.experiment runs/tiny.json runs/tiny-trial
uv run --no-sync python -m json.tool runs/tiny-trial/result.json
```
