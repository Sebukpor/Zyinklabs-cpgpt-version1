#!/usr/bin/env python3
"""Download and prepare ComputAgeBench data for finetuning CpGPT.

ComputAgeBench (Kriukov et al. 2024) is the dataset this pipeline uses for Version 1
training. It is the academic standard benchmark for epigenetic aging clocks, and
critically, it is the same dataset CpGPT's own authors used to finetune CpGPT for
the "Biomarkers of Aging Challenge", where it placed 2nd overall in chronological
age estimation and 1st on the public leaderboard for methylation-based mortality
prediction. Using it here means our finetuning recipe matches the one that produced
CpGPT's own best published results, rather than an ad hoc GEO selection.

Source: https://huggingface.co/datasets/computage/computage_bench
License: CC BY-SA 4.0 (free, attribution + share-alike, no fee)
Paper:  Kriukov et al., "ComputAgeBench: Epigenetic Aging Clocks Benchmark" (2024)

Dataset shape (as of this writing):
  - Training split:  7,419 samples, 907,766 features, 46 studies (healthy controls only)
  - Benchmark split: 10,410 samples, 900,449 features, 65 studies (healthy controls +
    patients with aging-accelerating conditions -- used for held-out evaluation, NOT
    for training, to keep the benchmark honest)

This script downloads the dataset via huggingface_hub, reshapes it into the
per-sample beta-value table format the rest of this pipeline expects (matching what
the previous wet-lab pipeline produced from raw IDATs), and writes a
participant-safe train/val split. The benchmark split is kept completely separate
and untouched until scripts/04_benchmark_vs_sota.py -- it is the held-out set used to
compare our finetuned model against Horvath, Hannum, PhenoAge, GrimAge2, and
DunedinPACE, not something to train on.

Usage:
    python 01_download_computage_bench.py \
        --out_dir data/processed/computage \
        --target age

For the disease-risk model, use the benchmark split's Condition/Class columns
instead (see scripts/02_prepare_cpgpt_data.py --target condition_class, which
consumes the output of this script).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_computage_bench")

REPO_ID = "computage/computage_bench"


def download_dataset(local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (this is ~38.6GB total; train split alone is smaller)", REPO_ID, local_dir)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(local_dir))
    log.info("Download complete.")


def load_split(local_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and reshape one split ('train' or 'benchmark') into
    (samples_x_features_df, meta_df), both indexed by GEO sample ID.
    """
    data_dir = local_dir / "data" / split
    parquet_files = sorted(data_dir.glob(f"computage_{split}_data_*.parquet"))
    if not parquet_files:
        msg = f"No parquet files found under {data_dir}. Did the download complete?"
        raise FileNotFoundError(msg)

    log.info("Loading %d dataset files for split '%s'...", len(parquet_files), split)
    frames = []
    for f in parquet_files:
        # Stored as features x samples (P >> N), per ComputAgeBench's own README --
        # transpose so rows are samples, matching the rest of this pipeline.
        df = pd.read_parquet(f).T
        frames.append(df)

    # Outer-join concat: different studies cover different CpG sites, so union
    # columns and let missing values be NaN -- handled downstream by CpGPT's own
    # missing-value handling, not imputed here.
    combined = pd.concat(frames, axis=0, join="outer")
    combined.index.name = "sample_id"

    meta_path = local_dir / f"computage_{split}_meta.tsv"
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    meta.index.name = "sample_id"

    return combined, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for prepared data")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=None,
        help="Where to download the raw ComputAgeBench snapshot (default: <out_dir>/_raw_computage_bench)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "benchmark"],
        choices=["train", "benchmark"],
        help="Which splits to prepare. Both are needed: 'train' for finetuning, "
        "'benchmark' for held-out SOTA comparison (script 05).",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip the download step (use if --raw_dir already has a complete snapshot)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (args.out_dir / "_raw_computage_bench")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_dataset(raw_dir)
    else:
        log.info("Skipping download, using existing snapshot at %s", raw_dir)

    for split in args.splits:
        betas, meta = load_split(raw_dir, split)

        # species column required downstream by CpGPTDataSaver (see
        # scripts/02_prepare_cpgpt_data.py).
        betas["species"] = "homo_sapiens"

        # Carry through the metadata columns the rest of the pipeline cares about,
        # renamed to match what scripts/02 and scripts/04 expect.
        meta_renamed = meta.rename(
            columns={
                "Age": "chronological_age",
                "DatasetID": "dataset_id",
                "PlatformID": "platform_id",
                "Tissue": "tissue",
                "CellType": "cell_type",
                "Gender": "sex",
                "Condition": "condition_name",
                "Class": "condition_class",
            },
        )

        # ComputAgeBench has no participant_id concept (each GEO sample is already
        # one donor at one timepoint in virtually all constituent studies) -- use
        # sample_id as participant_id so the existing participant-level split logic
        # in scripts/02_prepare_cpgpt_data.py still behaves correctly (no accidental
        # leakage, just no grouping needed here).
        merged = betas.join(meta_renamed, how="inner")
        merged = merged.reset_index().rename(columns={"sample_id": "sample_id"})
        merged["participant_id"] = merged["sample_id"]

        # dataset_id (the GEO accession) is retained explicitly so a later analysis
        # can check performance broken out by source study, not just in aggregate.
        out_path = args.out_dir / f"computage_{split}_merged.feather"
        merged.to_feather(out_path)
        log.info(
            "Wrote %s split: %d samples, %d columns -> %s",
            split, merged.shape[0], merged.shape[1], out_path,
        )

    log.info(
        "Done. Next: scripts/02_prepare_cpgpt_data.py --input "
        "%s/computage_train_merged.feather --target chronological_age "
        "--out_dir %s/cpgpt/bioage",
        args.out_dir, args.out_dir,
    )
    log.info(
        "Keep %s/computage_benchmark_merged.feather untouched for scripts/04_benchmark_vs_sota.py "
        "-- it must stay held-out, not used in any training split.",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
