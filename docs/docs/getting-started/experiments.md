# Local classification experiments

Submit JSON to the existing classification benchmark training flow. The contract
adds data identity, source/initialization provenance and lifecycle records; the
regular CLI `--benchmark` keeps its inference profiling.

```sh
uv sync --locked
uv run --no-sync python -m references.classification.experiment CONFIG.json runs/TRIAL
```

The trial path must not exist, including as a symlink. Use Git-ignored `runs/` or an
external directory. Input paths resolve from CONFIG; output paths from the working directory.

## Input v1

Use [the example](https://github.com/frgfm/Holocron/blob/main/references/classification/experiment.example.json).
Unknown fields/versions, invalid values and unsupported limits are rejected before training.

| Field | Values |
| --- | --- |
| Required | `schema_version: 1`, `model`, `dataset`, `training_device`, `deployment_target` |
| `model` | Catalog classification `name`; initialization `{"kind":"random"}` or `{"kind":"checkpoint","path":"local.pth"}` |
| `dataset` | `format: "imagefolder"`; `train`, `validation`, optional `test` directories |
| Devices / target | Shared `cpu`, `cuda`, `cuda:N`, `mps`, `auto` resolver; target is null or a description, never a measurement |
| Optional | `seed`; `training` and `preprocessing` use reference CLI defaults; `tracking` accepts `wb`, `name` |
| `training` | `epochs`, `lr`, `batch_size`, `workers`, `grad_acc`, `opt`, `sched`, `weight_decay`, `norm_wd`, `label_smoothing`, `mixup_alpha`, `amp` |
| `preprocessing` | `train_crop_size`, `val_resize_size`, `val_crop_size`, `random_erase`; the existing Imagenette RGB/crop/augment/normalize recipe |

Epochs are the only budget. `wall_time_seconds`, `max_steps`, `resume` and providers
are rejected. Optimizers use the reference factory; schedulers are `cosine` or
`onecycle` (`div_factor=100`, `pct_start=0.1`, exactly ten updates rejected).
Initialization loads local weights strictly with fresh optimizer/scheduler state.
No downloads. W&B needs the optional training dependencies. Seed is an integer
0–2³²−1 and seeds Python, NumPy and PyTorch; it does not promise identical bits across runtimes.

## Output v1

| File | Contents |
| --- | --- |
| `config.json` | Resolved JSON inputs |
| `provenance.json` | Git revision/dirty state before trial creation, packages, actual transforms, data/init hashes and optimizer defaults |
| `data.json` | Matching class mapping and exact split membership with file hashes |
| `progress.jsonl` | Append-only benchmark epoch events, validation split; no intra-epoch heartbeat |
| `result.json` | Native benchmark `runtime`, `recipe`, `dataset`, `models`, plus versioned state, timestamps, elapsed time, epoch, actual device, error and exit code |
| `checkpoint.pth` | Selected weights; atomic replacement preserves the prior checkpoint on failed saves |

`models[0]` holds `history`, `final_epoch`/`final_metrics`, `selected_epoch`/
`selected_metrics`, `checkpoint`/`checkpoint_sha256` and `fully_resumable: false`.
Selected metrics come from the saved epoch; ties retain the earlier checkpoint.
`metrics_split` and `metric_directions` label validation metrics: loss minimizes;
acc1/acc5 maximize (fractions; acc5 needs five classes). Non-finite validation loss
fails. Failure records retain available metrics/checkpoints. JSON trials stop after
training/result collection; they do not run fusion, export or latency profiling.

| State | Meaning / exit code |
| --- | --- |
| `running` | Setup/training; finish and exit are null; not proof of process liveness |
| `completed` | All requested epochs and a selected checkpoint; 0 |
| `failed` | Setup/training/tracking error; exception re-raised, traceback stored; 1 |
| `interrupted` | KeyboardInterrupt 130, SIGTERM 143, or Python's SystemExit code |

File hashes use raw-byte SHA-256. Split hashes use UTF-8 JSON samples with sorted
keys and separators `(',', ':')`; roots/mapping are stored separately. Initial-state
hashes use sorted `[name, dtype, shape]` JSON followed by contiguous CPU tensor bytes.
Copies/aliases across splits are rejected; re-encoded duplicates are not detected.
**Test is hashed only**, never decoded, trained, evaluated or used for selection.
Data is not locked/snapshotted; dirty source is not saved, and absent Git metadata is null.
Abrupt kills/storage failures can leave `running` or partial progress. Verify checkpoint
hashes before use. True resume, wall-time enforcement, search and agent servers remain deferred.

## Offline CPU example

After installation, generate local images and use a fresh trial name:

```sh
mkdir -p runs
cp references/classification/experiment.example.json runs/tiny.json
uv run --no-sync python - <<'PY'
from pathlib import Path
from PIL import Image

for split_index, split in enumerate(("train", "validation", "test")):
    for class_index, label in enumerate(("dark", "light")):
        root = Path("runs/tiny-data") / split / label
        root.mkdir(parents=True, exist_ok=True)
        for sample in range(2):
            color = (split_index * 60 + class_index * 20 + sample, 30, 90)
            Image.new("RGB", (40, 40), color).save(root / f"{sample}.png")
PY
OMP_NUM_THREADS=2 uv run --no-sync python -m references.classification.experiment runs/tiny.json runs/tiny-trial
uv run --no-sync python -m json.tool runs/tiny-trial/result.json
```
