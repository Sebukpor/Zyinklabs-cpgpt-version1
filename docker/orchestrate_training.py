#!/usr/bin/env python3
"""Orchestrates a full ZyinkLabs CpGPT finetuning run inside the training container:

    1. Run `cpgpt-train` for the requested experiment.
    2. Read the resulting metrics and check val/condition_loss against a threshold.
    3. If — and only if — the run passes the threshold, push the checkpoint +
       config + a generated model card to the Hugging Face Hub.
    4. If it fails, exit non-zero with a clear message and DO NOT push anything.

This is intentionally a gate, not a suggestion: a failing run must not reach the
Hub. Treat "successful" as "training completed AND validation loss beats the
configured threshold" -- nothing softer than that should result in a push.

Environment variables (set via docker-compose.yml / .env):
    EXPERIMENT_NAME        e.g. zyinklabs_bioage or zyinklabs_disease
    VAL_LOSS_THRESHOLD     float; run is "successful" if val/condition_loss <= this
    HF_TOKEN                Hugging Face write-access token
    HF_REPO_ID               e.g. zyinklabs/cpgpt-bioage-v1
    HF_PRIVATE               "true"/"false" -- whether the HF repo should be private
    SKIP_PUSH                "true" to run training + validation only, never push
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CPGPT_DIR = Path("/workspace/CpGPT")
LOG_DIR = CPGPT_DIR / "logs" / "experiments"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def run_training(experiment_name: str) -> None:
    log(f"Starting cpgpt-train experiment={experiment_name}")
    result = subprocess.run(
        ["cpgpt-train", f"experiment={experiment_name}"],
        cwd=CPGPT_DIR,
        check=False,
    )
    if result.returncode != 0:
        log(f"Training process exited with code {result.returncode}. Aborting -- no push will happen.")
        sys.exit(result.returncode)
    log("Training process completed.")


def find_latest_run_dir(experiment_name: str) -> Path:
    exp_dir = LOG_DIR / experiment_name
    if not exp_dir.exists():
        msg = f"No logs found at {exp_dir} -- training may not have produced output."
        raise FileNotFoundError(msg)
    run_dirs = sorted(exp_dir.iterdir(), key=lambda p: p.name)
    if not run_dirs:
        msg = f"{exp_dir} exists but has no run subdirectories."
        raise FileNotFoundError(msg)
    return run_dirs[-1]


def read_best_val_condition_loss(run_dir: Path) -> float:
    """Read the CSV logger output and return the best (minimum) val/condition_loss
    seen during training. Assumes the `csv` logger (configs/logger/csv.yaml) is
    active, per the experiment configs in this repo.
    """
    csv_candidates = list(run_dir.rglob("metrics.csv"))
    if not csv_candidates:
        msg = f"No metrics.csv found under {run_dir} -- check logger config (expected csv logger)."
        raise FileNotFoundError(msg)

    best = None
    for csv_path in csv_candidates:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "val/condition_loss" not in (reader.fieldnames or []):
                continue
            for row in reader:
                val = row.get("val/condition_loss", "")
                if val in ("", None):
                    continue
                try:
                    val_f = float(val)
                except ValueError:
                    continue
                if best is None or val_f < best:
                    best = val_f

    if best is None:
        msg = "Could not find any 'val/condition_loss' values in metrics.csv files."
        raise ValueError(msg)
    return best


def find_best_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        msg = f"No .ckpt files found under {run_dir / 'checkpoints'}."
        raise FileNotFoundError(msg)
    # model_checkpoint callback is configured to save the best (min val/condition_loss)
    # checkpoint; if multiple are kept, take the most recently modified.
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def write_model_card(
    out_path: Path,
    experiment_name: str,
    val_loss: float,
    threshold: float,
    run_dir: Path,
) -> None:
    card = f"""---
license: cc-by-sa-4.0
tags:
  - dna-methylation
  - epigenetics
  - biological-age
  - cpgpt
  - zyinklabs
  - computagebench
---

# ZyinkLabs {experiment_name} (Version 1)

This model is a finetuned version of [CpGPT](https://github.com/lcamillo/CpGPT)
(de Lima Camillo et al.), produced by ZyinkLabs for biological age / disease-risk
estimation from DNA methylation data.

**Version 1 is trained exclusively on public data** — specifically
[ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench)
(Kriukov et al., 2024), the academic-standard benchmark and training dataset for
epigenetic aging clocks. This is the same dataset CpGPT's own authors used to
finetune CpGPT for the Biomarkers of Aging Challenge, where it placed 2nd overall
in chronological age estimation and 1st on the public leaderboard for
methylation-based mortality prediction. No ZyinkLabs proprietary or wet-lab-derived
sample data was used in this model.

## Training summary

- **Experiment:** `{experiment_name}`
- **Base model:** CpGPT large backbone (~100M params; pretrained checkpoint, see
  `configs/experiment/{experiment_name}.yaml` for the exact `model_ckpt_path` used)
- **Training data:** ComputAgeBench training split (7,419 samples, 46 studies,
  healthy controls only) — license CC BY-SA 4.0
- **Validation metric:** `val/condition_loss` = **{val_loss:.4f}**
  (acceptance threshold: <= {threshold:.4f})
- **Run directory:** `{run_dir.name}`
- **Trained:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Important caveats

- Passing the configured `val/condition_loss` threshold confirms the model fits its
  own held-out validation split (drawn from the ComputAgeBench training split). It
  does **not** by itself confirm agreement with the broader published-clock
  literature or performance distinguishing aging-accelerating conditions from
  healthy controls. Run `scripts/04_benchmark_vs_sota.py` against the **held-out
  ComputAgeBench benchmark split** (10,410 samples, 65 studies, never used in
  training) before using this model for anything customer-facing — that script
  compares this model against Horvath, Hannum, and PhenoAge using the same
  methodology as the original ComputAgeBench paper.
- This is a Version 1 / public-data-only model. As ZyinkLabs' own wet-lab cohort
  grows, Version 2 is expected to finetune further on real ZyinkLabs samples — see
  the project scope documents for details.
- CpGPT itself is MIT licensed for commercial use, with one carve-out: it may not
  be used to enter biomarkers-of-aging competitions with monetary prizes. The
  ComputAgeBench training data is CC BY-SA 4.0 (attribution + share-alike). This
  finetuned derivative should be distributed consistent with both licenses —
  confirm with legal/compliance before any commercial redistribution beyond
  internal use.

## Citation

If referencing the training data, please cite:

> Kriukov, D., Efimov, E., Kuzmina, E. A., Khrameeva, E. E., & Dylov, D. V. (2024).
> ComputAgeBench: Epigenetic Aging Clocks Benchmark. bioRxiv, 2024-06.

## Intended use

Internal ZyinkLabs research and product development. Not a diagnostic device.
"""
    out_path.write_text(card)


def push_to_hub(run_dir: Path, checkpoint_path: Path, model_card_path: Path) -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    private = os.environ.get("HF_PRIVATE", "true").lower() == "true"

    if not token or not repo_id:
        msg = "HF_TOKEN and HF_REPO_ID must be set to push to Hugging Face."
        raise RuntimeError(msg)

    log(f"Creating/confirming Hugging Face repo: {repo_id} (private={private})")
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")

    api = HfApi(token=token)

    log(f"Uploading checkpoint: {checkpoint_path.name}")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        commit_message=f"Upload checkpoint from run {run_dir.name}",
    )

    config_path = run_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        log("Uploading training config (config.yaml)")
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            commit_message="Upload training configuration",
        )

    log("Uploading model card (README.md)")
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Upload model card",
    )

    log(f"Push complete: https://huggingface.co/{repo_id}")


def main() -> None:
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    if not experiment_name:
        log("EXPERIMENT_NAME is not set. Aborting.")
        sys.exit(1)

    threshold_raw = os.environ.get("VAL_LOSS_THRESHOLD")
    if not threshold_raw:
        log("VAL_LOSS_THRESHOLD is not set -- refusing to run without an explicit gate.")
        sys.exit(1)
    threshold = float(threshold_raw)

    skip_push = os.environ.get("SKIP_PUSH", "false").lower() == "true"

    run_training(experiment_name)

    run_dir = find_latest_run_dir(experiment_name)
    log(f"Reading metrics from {run_dir}")

    val_loss = read_best_val_condition_loss(run_dir)
    log(f"Best val/condition_loss observed: {val_loss:.6f} (threshold: {threshold:.6f})")

    result_summary = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "best_val_condition_loss": val_loss,
        "threshold": threshold,
        "passed": val_loss <= threshold,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "zyinklabs_result_summary.json").write_text(json.dumps(result_summary, indent=2))

    if val_loss > threshold:
        log(
            f"FAILED validation gate: {val_loss:.6f} > {threshold:.6f}. "
            "Training completed but did NOT meet the quality bar -- nothing will be pushed to Hugging Face.",
        )
        sys.exit(2)

    log(f"PASSED validation gate: {val_loss:.6f} <= {threshold:.6f}.")

    checkpoint_path = find_best_checkpoint(run_dir)
    model_card_path = run_dir / "MODEL_CARD.md"
    write_model_card(model_card_path, experiment_name, val_loss, threshold, run_dir)

    if skip_push:
        log("SKIP_PUSH=true -- validation passed but push to Hugging Face was explicitly skipped.")
        return

    push_to_hub(run_dir, checkpoint_path, model_card_path)


if __name__ == "__main__":
    main()
