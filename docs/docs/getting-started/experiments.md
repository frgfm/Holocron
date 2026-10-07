# Local classification experiments

An owner's existing agent can submit JSON and read one trial directory. Run from
the repository checkout. No server, agent framework, or log parser is needed.

```sh
uv sync --locked
uv run --no-sync python -m references.classification.experiment CONFIG.json runs/TRIAL
```

`runs/TRIAL` must not exist, even as an empty directory. Paths in the configuration
are relative to its file; the trial path is relative to the working directory.
`runs/` and checkpoints are ignored by Git. Keep outputs there or outside the
checkout; custom output locations inside the checkout need their own ignore rule.

## Input version 1

See [the example configuration](https://github.com/frgfm/Holocron/blob/main/references/classification/experiment.example.json).
The required fields are `schema_version: 1`, `model`, `dataset`, `training_device`,
and `deployment_target`. All unknown fields, unsupported versions, invalid values,
and unsupported limits are rejected before training. Configuration errors create
no trial; setup errors after directory creation produce a failed trial.

| Field | Supported values |
| --- | --- |
| `model.name` | A classification factory from `holocron.models.list_models("classification")` |
| `model.initialization` | `{"kind": "random"}` or `{"kind": "checkpoint", "path": "local.pth"}` |
| `dataset` | `format: "imagefolder"`, `train` and `validation` directory paths, optional `test` path |
| `training_device` | `"cpu"` or an available `"cuda:N"`; CPU AMP and MPS are rejected |
| `deployment_target` | `null` or a nonempty description of intended deployment; never a measurement |
| `seed` | Integer from 0 through 2³²−1 |
| `preprocessing` | `train_crop_size`, `val_resize_size`, `val_crop_size`, `random_erase` |
| `training` | `epochs`, `lr`, `batch_size`, `workers`, `grad_acc`, `opt`, `sched`, `weight_decay`, `norm_wd`, `label_smoothing`, `mixup_alpha`, `amp` |
| `tracking` | Optional `wb` boolean and `name` string or null; requires `uv sync --locked --extra training` for W&B |

Optional settings use the existing classification CLI parser's defaults, resolved
and saved in `config.json`. Optimizers are the existing `sgd`, `radam`, `adamw`,
`adamp`, `adabelief`, and `ademamix`; schedulers are `onecycle` and `cosine`.
OneCycle uses the reference's `div_factor=100`, `pct_start=0.1`; cosine uses its
native defaults. Optimizer defaults and scheduler arguments are saved in provenance.
Epochs are the only supported execution budget. For example `wall_time_seconds`,
`max_steps`, and `resume` are rejected, wherever placed.

The ImageFolder recipe is the same as the existing Imagenette reference:
PIL RGB loading; bilinear random resized crop with scale `(0.3, 1.0)`, horizontal
flip, TrivialAugmentWide, float32 conversion, Imagenette normalization, and random
erasing with scale `(0.02, 0.2)`. Validation uses bilinear resize and center crop,
float32 conversion, and the same normalization. Provenance records the actual
transform representations, including normalization values and torchvision defaults.
Mixup and cross-entropy label smoothing are recorded in the resolved configuration.
Training randomly samples the declared train pool and drops the last incomplete
batch, as the reference does; validation reads every sample. A train split smaller
than one full batch is rejected.

Random initialization records a digest of the initial model state. Checkpoint
initialization accepts a local tensor state dictionary or the trainer's `model`
dictionary, loads strictly, records the file hash and initial state digest, and
starts a fresh optimizer, scheduler, and epoch count. No weights are downloaded.

## Data identity

`data.json` records the class-to-index mapping and each split's ordered ImageFolder
sample list: relative file path, class index, and SHA-256 of **raw file bytes**.
The split digest is SHA-256 of the UTF-8 JSON sample list with sorted object keys
and separators `(',', ':')`; absolute roots are recorded separately. Renaming,
relabeling, adding, removing, or changing sample bytes changes the split digest.
The initial-state digest hashes sorted state-dictionary entries: UTF-8 JSON
`[name, dtype, shape]` followed by each contiguous CPU tensor's raw bytes.

All declared splits must have identical class mappings and disjoint content
hashes. This rejects aliases, hard links, and byte-identical copies across splits,
including a declared test split. It does not detect re-encoded near-duplicate
images. The optional test split is enumerated and hashed only: its images are not
decoded, evaluated, trained on, or used for checkpoint selection. Keep data fixed
for the whole trial; this slice fingerprints before training and does not lock or
snapshot dataset files. The mapping is stored separately from each split digest.

## Read a trial

| File | Meaning |
| --- | --- |
| `config.json` | Versioned resolved inputs, including absolute data/initialization paths |
| `provenance.json` | Source revision/dirty state, package/Python/platform versions, data digests, initialization identity, actual transforms, optimizer/scheduler settings, threads and device |
| `data.json` | Exact class mappings, split membership, and file/split digests |
| `progress.jsonl` | Append-only epoch-end events from the existing trainer callback |
| `result.json` | Atomic current status or terminal result, timestamps, elapsed seconds, final epoch metrics, selected checkpoint, and exception details |
| `checkpoint.pth` | Existing trainer's checkpoint selected by the lowest validation loss; strict improvement wins, so ties keep the earlier epoch |

JSON records use `schema_version: 1`; timestamps use UTC ISO 8601. Metadata and
result replacements are atomic within the trial directory. `artifacts` lists
metadata paths relative to that directory; `selected_checkpoint` contains its
relative path, SHA-256, epoch, validation loss, associated metrics, and
`fully_resumable: false`. Early setup failures can leave only some metadata files.
Progress is empty until the first complete epoch; it has no intra-epoch heartbeat.

| State | Meaning |
| --- | --- |
| `running` | Preparation or training is in progress; `epoch` counts reported epochs, `exit_code` and `finished_at` are null |
| `completed` | All requested epochs returned and a checkpoint was selected; exit code 0 |
| `failed` | A setup/training/tracking exception escaped; traceback is preserved, exit code 1 |
| `interrupted` | KeyboardInterrupt (exit 130), SIGTERM (exit 143), or SystemExit; the integer SystemExit code is preserved |

Exceptions are re-raised and process exit codes remain nonzero on failure. An
abrupt kill (SIGKILL), power loss, or storage failure can leave an incomplete
`running` record, an incomplete last progress line, or a partly written checkpoint.
Check the selected checkpoint hash before using it. `running` is not proof of
process liveness. Git metadata is null if no Git checkout is available; dirty
source is identified as dirty, but its contents are not snapshotted.

Metrics include their split, epoch, and direction. Available metrics are validation
`val_loss` (minimize), `acc1` (maximize), and `acc5` (maximize, only for at least
five classes). Accuracy is a fraction. Validation loss retains the trainer's mean
of batch losses. No training or test metrics are invented. `final_epoch_metrics`
describes the last reported epoch, even on failure; `selected_checkpoint.metrics`
describes the selected epoch, which can be earlier. Existing callbacks and optional
W&B logging receive the usual trainer metrics; the local record is written first.

## Tiny offline CPU example

Run these exact commands from the checkout. They generate local PNGs and use random
weights; training needs no dataset or checkpoint downloads. Environment setup may
install packages. Use a new trial name on subsequent runs.

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
config, provenance, data, result = [json.loads((root / name).read_text()) for name in
    ("config.json", "provenance.json", "data.json", "result.json")]
selected = result["selected_checkpoint"]
assert result["state"] == "completed" and result["exit_code"] == 0
assert hashlib.sha256((root / selected["path"]).read_bytes()).hexdigest() == selected["sha256"]
print(config["model"], config["training"], config["preprocessing"])
print(provenance["source"], provenance["actual_device"], data["class_to_idx"])
print({split: value["sha256"] for split, value in data["splits"].items()})
print(result["state"], result["final_epoch_metrics"], selected)
PY
```

## Limits

Only local ImageFolder classification is integrated. Seeds are recorded and set
for Python, NumPy and PyTorch, but bitwise reproducibility across devices, package
versions and worker counts is not promised. The existing checkpoints contain model
weights and counters; they lack optimizer, scheduler and random-number states and
are **not fully resumable**. There is no last-epoch checkpoint unless that epoch
was selected. Deployment intent is not export, accuracy on a deployment runtime,
latency, throughput, or memory evidence. True resume, wall-time enforcement, search,
external providers, export/latency integration, dashboards and agent/MCP servers
remain outside this contract.
