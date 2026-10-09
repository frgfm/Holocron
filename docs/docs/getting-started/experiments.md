# Local classification experiments

From the checkout, submit JSON and read one new trial directory:

```sh
uv sync --locked
uv run --no-sync python -m references.classification.experiment CONFIG.json runs/TRIAL
```

The trial path must not exist, including as a symlink. Input paths are relative to
CONFIG; output paths to the working directory. Use Git-ignored `runs/` or an external directory.

## Input v1

Use [the example](https://github.com/frgfm/Holocron/blob/main/references/classification/experiment.example.json).
Required: `schema_version: 1`, `model`, `dataset`, `training_device`, `deployment_target`.
Optional settings resolve from the existing CLI defaults. Unknown fields/versions,
invalid values and unsupported limits are rejected. Config errors create no trial;
setup/training errors leave a failed record.

| Field | Values |
| --- | --- |
| `model` | Catalog classification `name`; `initialization` is `{"kind":"random"}` or `{"kind":"checkpoint","path":"local.pth"}` |
| `dataset` | `format: "imagefolder"`; `train`, `validation`, optional `test` directories |
| `training_device` | `"cpu"` or available `"cuda:N"`; AMP requires CUDA |
| `deployment_target` | null or nonempty description; intent only, never a measurement |
| `seed` | Integer 0 through 2³²−1; seeds Python, NumPy and PyTorch |
| `preprocessing` | `train_crop_size`, `val_resize_size`, `val_crop_size`, `random_erase` |
| `training` | `epochs`, `lr`, `batch_size`, `workers`, `grad_acc`, `opt`, `sched`, `weight_decay`, `norm_wd`, `label_smoothing`, `mixup_alpha`, `amp` |
| `tracking` | Optional `wb` boolean, `name` string/null; W&B requires `--extra training` at install |

Optimizers: `sgd`, `radam`, `adamw`, `adamp`, `adabelief`, `ademamix`. Schedulers:
`cosine` or `onecycle` with reference `div_factor=100`, `pct_start=0.1`. OneCycle rejects
exactly ten planned updates; change epochs or use cosine. Epochs are the only budget;
`wall_time_seconds`, `max_steps` and `resume` are rejected. Local initialization loads
weights strictly and starts fresh optimizer/scheduler/epoch state. No weights are downloaded.

Preprocessing is the shared Imagenette recipe: PIL RGB, bilinear random resized crop
(scale 0.3–1), horizontal flip, TrivialAugmentWide, float32 conversion, Imagenette
normalization and random erasing (scale 0.02–0.2). Validation uses resize/center crop
and the same conversion/normalization. Exact transforms and optimizer/scheduler defaults
are recorded. Training shuffles and drops incomplete batches; validation reads all samples.
At least one full training batch is required. Mixup and label smoothing remain configurable.

## Output v1

| File | Contents |
| --- | --- |
| `config.json` | Resolved inputs and absolute data/initialization paths |
| `provenance.json` | Git revision/dirty state, package/runtime versions, data/init hashes, actual preprocessing, optimizer/scheduler, threads/device |
| `data.json` | Class mapping and ordered membership: relative path, class index, file SHA-256 |
| `progress.jsonl` | Append-only epoch events; empty until the first reported epoch, no intra-epoch heartbeat |
| `result.json` | Atomic status, UTC start/finish timestamps, elapsed seconds, final metrics, selected checkpoint, traceback/exit code |
| `checkpoint.pth` | Lowest validation-loss checkpoint; ties retain the earlier epoch; atomic saves preserve the previous file on failure |

`selected_checkpoint` records the relative path, SHA-256, epoch, loss, metrics and
`fully_resumable: false`. Metric records contain split, epoch and direction:
validation `val_loss` minimizes; `acc1`/`acc5` maximize (fractions; acc5 needs five classes).
Loss is the trainer's mean of batch losses; non-finite validation loss fails the trial.
`final_epoch_metrics` is the last reported epoch, even on failure; selected metrics
belong to the chosen checkpoint. No training/test metrics are invented. Callbacks and W&B remain.

| State | Meaning / exit code |
| --- | --- |
| `running` | Preparing/training; epoch counts reports; exit/finish are null |
| `completed` | All requested epochs returned and a model was selected; 0 |
| `failed` | Setup/training/tracking error, traceback preserved and exception re-raised; 1 |
| `interrupted` | KeyboardInterrupt 130, SIGTERM 143, or Python's SystemExit code |

## Identity and limits

- File hashes use raw bytes. Split hashes use the UTF-8 JSON sample list with sorted
  keys and separators `(',', ':')`; roots/class mapping are stored separately.
- Initial-state hashes use sorted entries: Python JSON `[name, dtype, shape]` followed
  by contiguous CPU tensor bytes. Local initialization also records its file hash.
- All splits require matching classes and disjoint hashes. Copies/aliases are caught;
  re-encoded duplicates are not. **Test is hashed only**, never decoded, trained or evaluated.
- Keep data fixed: it is hashed before training, not locked/snapshotted. Seeds do not
  promise identical bits across runtimes. Missing Git metadata is null; dirty source is not snapshotted.
- Abrupt kills/power/storage failures can leave `running` or a partial progress line.
  Running is not liveness proof; verify the checkpoint hash before use. Setup failures may leave partial metadata.
- No true resume: optimizer/scheduler/RNG state is absent. No final checkpoint unless
  selected. Wall-time enforcement, search, providers, export/latency, dashboards and servers remain deferred.

## Offline CPU example

After installation above, generate PNGs and run random weights. Use a fresh trial name:

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
