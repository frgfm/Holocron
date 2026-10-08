# Local classification experiments

Submit JSON; read a trial directory. Run from the checkout:

```sh
uv sync --locked
uv run --no-sync python -m references.classification.experiment CONFIG.json runs/TRIAL
```

The trial path must not exist, even as an empty directory. Input paths are relative
to the config file; output paths are relative to the working directory. `runs/`
and checkpoints are ignored by Git; use that directory or an external location.

## Input v1

Use [the example](https://github.com/frgfm/Holocron/blob/main/references/classification/experiment.example.json).
Required fields: `schema_version: 1`, `model`, `dataset`, `training_device`,
`deployment_target`. Optional values resolve from the existing classification CLI
defaults. Unknown fields, invalid values, unsupported versions and limits fail
before training. Config errors create no trial; later setup errors record `failed`.

| Field | Supported values |
| --- | --- |
| `model.name` | Catalog classification factory; see `holocron.models.list_models("classification")` |
| `model.initialization` | `{"kind":"random"}` or `{"kind":"checkpoint","path":"local.pth"}`; no downloads |
| `dataset` | `format: "imagefolder"`, `train`, `validation`, optional `test` directory |
| `training_device` | `"cpu"` or available `"cuda:N"`; AMP requires CUDA |
| `deployment_target` | `null` or nonempty description; intent only, never measured deployment evidence |
| `seed` | Integer 0 through 2³²−1; sets Python, NumPy and PyTorch seeds |
| `preprocessing` | `train_crop_size`, `val_resize_size`, `val_crop_size`, `random_erase` |
| `training` | `epochs`, `lr`, `batch_size`, `workers`, `grad_acc`, `opt`, `sched`, `weight_decay`, `norm_wd`, `label_smoothing`, `mixup_alpha`, `amp` |
| `tracking` | Optional `wb` boolean and `name` string/null; W&B needs `uv sync --locked --extra training` |

Optimizers: `sgd`, `radam`, `adamw`, `adamp`, `adabelief`, `ademamix`. Schedulers:
`onecycle` (reference `div_factor=100`, `pct_start=0.1`) and `cosine` (native defaults).
Epochs are the only execution budget; `wall_time_seconds`, `max_steps` and `resume`
are rejected. Checkpoint initialization strictly loads a local state dictionary or
trainer `model` dictionary, then starts fresh optimizer, scheduler and epoch state.

Preprocessing reuses the Imagenette reference: PIL RGB, bilinear random resized crop
(scale 0.3–1), horizontal flip, TrivialAugmentWide, float32 conversion, Imagenette
normalization and random erasing (scale 0.02–0.2). Validation uses bilinear resize
and center crop with the same conversion/normalization. Provenance saves actual
transform representations, optimizer defaults and scheduler arguments. Training
shuffles and drops incomplete batches; validation reads every sample. At least one
full training batch is required. Mixup and label smoothing remain configurable.

## Read a trial

| File | Contents |
| --- | --- |
| `config.json` | Versioned resolved inputs and absolute data/checkpoint paths |
| `provenance.json` | Source revision/dirty state, package/Python/platform versions, data/init identity, actual transforms, optimizer/scheduler settings, threads and device |
| `data.json` | Class mapping; ordered split membership with relative path, class index and file SHA-256 |
| `progress.jsonl` | Append-only epoch-end events; empty before the first complete epoch, no intra-epoch heartbeat |
| `result.json` | Atomic status/result, UTC timestamps, elapsed seconds, metrics, selected checkpoint and exception traceback |
| `checkpoint.pth` | Checkpoint with lowest validation loss; ties keep the earlier epoch |

JSON records use `schema_version: 1`. Paths in `artifacts` and `selected_checkpoint`
are relative to the trial. The latter includes SHA-256, epoch, loss, associated
metrics and `fully_resumable: false`. Setup failures may leave partial metadata.

| State | Meaning | Exit code |
| --- | --- | --- |
| `running` | Preparing/training; `epoch` counts reported epochs, `finished_at` is null | null |
| `completed` | All requested epochs returned and a checkpoint was selected | 0 |
| `failed` | Setup/training/tracking exception; re-raised with traceback preserved | 1 |
| `interrupted` | KeyboardInterrupt, SIGTERM or SystemExit | 130, 143 or Python's SystemExit code |

Metrics carry split, epoch and direction: validation `val_loss` minimizes;
`acc1` and `acc5` maximize (fractions; `acc5` requires five classes). Loss retains
the trainer's mean of batch losses. `final_epoch_metrics` means last reported epoch,
even on failure; `selected_checkpoint.metrics` means the selected epoch. No training
or test metrics are invented. Existing callbacks and optional W&B logging remain.

## Identity and limits

- Files: SHA-256 of raw bytes. Split digest: SHA-256 of the UTF-8 JSON sample list,
  sorted object keys, separators `(',', ':')`; absolute root and class mapping are
  stored separately. Renaming, relabeling, membership or byte changes alter identity.
- Initial model: hash sorted state-dictionary entries, each UTF-8 JSON
  `[name, dtype, shape]` followed by contiguous CPU tensor bytes. Local initialization
  also records the checkpoint file hash.
- All declared splits need identical class mappings and disjoint content hashes.
  Copies, aliases and hard links are caught; re-encoded near-duplicates are not.
  **Test is enumerated and hashed only**, never decoded, trained on or evaluated.
- Keep data fixed during the trial: files are fingerprinted before training, not
  locked/snapshotted. Seeds do not promise bitwise reproducibility across runtimes.
- Abrupt kills, power/storage failures can leave `running`, a partial progress line
  or a partial checkpoint. `running` is not liveness proof; verify checkpoint hashes.
  Missing Git metadata is null; dirty source contents are not snapshotted.
- Checkpoints lack optimizer, scheduler and RNG state: **no true resume**. No final
  checkpoint unless selected. Wall-time enforcement, search, providers, export/latency,
  dashboards and agent/MCP servers remain deferred.

## Offline CPU example

Environment setup may install packages; the trial uses local PNGs and random weights.
Use a fresh trial name each time.

```sh
uv sync --locked
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
uv run --no-sync python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("runs/tiny-trial")
records = {name: json.loads((root / f"{name}.json").read_text())
           for name in ("config", "provenance", "data", "result")}
result = records["result"]
selected = result["selected_checkpoint"]
assert result["state"] == "completed" and result["exit_code"] == 0
assert hashlib.sha256((root / selected["path"]).read_bytes()).hexdigest() == selected["sha256"]
print(json.dumps(records, indent=2))
PY
```
