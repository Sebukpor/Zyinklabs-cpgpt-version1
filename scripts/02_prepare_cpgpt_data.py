#!/usr/bin/env python3
"""Convert a ComputAgeBench-derived sample table into CpGPT's memory-mapped data
format, split into train/val/test by PARTICIPANT (not by sample), and write out
directories ready for `cpgpt-train`.

Splitting by participant_id (not sample_id) matters: if the same person contributed
multiple samples (e.g. repeat draws across studies), letting some of their samples
land in train and others in test leaks information and inflates validation
performance. ComputAgeBench is mostly one-sample-per-donor, but the split logic
stays in place as a safety net regardless.

Feature selection: CpGPT's own FAQ recommends, for phenotype-targeted finetuning,
either filtering to a pretrained model's vocab (--vocab_filter) or fitting a ridge
regression and keeping the top-N coefficients by magnitude (--ridge_select_top_n) to
speed up training. Both can be combined (vocab filter first, then ridge selection
within the surviving columns) or used independently. Selection is always fit on the
TRAIN split only and then applied identically to val/test, to avoid leaking
information about the held-out splits into which features were chosen.

This wraps cpgpt.data.components.cpgpt_datasaver.CpGPTDataSaver, cpgpt.data.components
.dna_llm_embedder.DNALLMEmbedder, and cpgpt.data.components.illumina_methylation_prober
.IlluminaMethylationProber — see CpGPT's quick_setup.ipynb tutorial for the reference
pattern this is based on.

Usage (biological age model, using the ComputAgeBench training split, with the
FAQ-recommended ridge-selection of the top 20,000 most age-informative CpG sites):
    python 02_prepare_cpgpt_data.py \
        --input data/processed/computage/computage_train_merged.feather \
        --target chronological_age \
        --out_dir data/processed/cpgpt/bioage \
        --cpgpt_dir ../CpGPT \
        --ridge_select_top_n 20000

For the disease-risk model (using the benchmark split's condition labels — note
this is the same split held out for SOTA comparison in script 04, so for an honest
disease-risk *training* run you should partition it carefully or treat this model
as exploratory until a dedicated outcome-labeled cohort is available):
    python 02_prepare_cpgpt_data.py \
        --input data/processed/computage/computage_benchmark_merged.feather \
        --target condition_class \
        --out_dir data/processed/cpgpt/disease \
        --cpgpt_dir ../CpGPT
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_cpgpt_data")


def participant_level_split(
    df: pd.DataFrame,
    group_col: str = "participant_id",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test, keeping all rows from the same
    participant_id together in a single split.
    """
    if group_col not in df.columns:
        log.warning(
            "'%s' column not found — falling back to row-level split. "
            "If samples share participants (e.g. repeat draws), this risks data leakage.",
            group_col,
        )
        groups = df.index.to_series()
    else:
        groups = df[group_col]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    val_frac_adjusted = val_frac / (1 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_adjusted, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=groups.iloc[train_val_idx]))

    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    return train_df, val_df, test_df


def select_features_by_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    probe_cols: list[str],
    top_n: int,
    seed: int = 42,
) -> list[str]:
    """Fit a ridge regression on the TRAIN split only (per CpGPT's FAQ guidance for
    phenotype-targeted finetuning: "using a ridge regression and picking the top N
    features can speed up the training time required"), and return the top_n probe
    columns by absolute coefficient magnitude.

    Fitting only on train and applying the same selected columns to val/test avoids
    leaking held-out-split information into which features get chosen — the model
    should never have a chance to "know" something about val/test during selection.

    NaNs in probe columns (expected, since ComputAgeBench studies cover different
    CpG sites and the outer-join concat in script 01 fills gaps with NaN) are
    median-imputed using TRAIN-split medians only, for this feature-selection step
    only — CpGPT's own data loader handles missing values in the final memory-mapped
    data independently, so this imputation does not affect what the model actually
    trains on, only which columns get selected.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV

    if top_n >= len(probe_cols):
        log.info(
            "--ridge_select_top_n (%d) >= available probe columns (%d) — skipping selection, using all columns.",
            top_n, len(probe_cols),
        )
        return probe_cols

    log.info("Fitting ridge regression on train split to select top %d / %d CpG sites...", top_n, len(probe_cols))

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_df[probe_cols])
    y_train = train_df[target].to_numpy()

    # RidgeCV picks alpha via leave-one-out-style internal CV on the train split only.
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(x_train, y_train)

    coef_magnitude = np.abs(model.coef_)
    top_idx = np.argsort(coef_magnitude)[::-1][:top_n]
    selected = [probe_cols[i] for i in top_idx]

    log.info(
        "Selected %d CpG sites by ridge coefficient magnitude (alpha=%.4g, train R^2=%.4f).",
        len(selected), model.alpha_, model.score(x_train, y_train),
    )
    return selected



    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Merged beta+metadata table (.feather)")
    parser.add_argument(
        "--target",
        required=True,
        help="Metadata column to use as the prediction target (e.g. chronological_age, condition_status)",
    )
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for processed train/val/test")
    parser.add_argument(
        "--cpgpt_dir",
        type=Path,
        default=Path("../CpGPT"),
        help="Path to the cloned CpGPT repo (for dependencies_dir)",
    )
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vocab_filter",
        type=Path,
        default=None,
        help="Optional path to a model vocab.json to filter CpG sites to (recommended — "
        "see CpGPT FAQ: filtering to the finetuning model's vocab gives best performance)",
    )
    parser.add_argument(
        "--ridge_select_top_n",
        type=int,
        default=None,
        help="If set, fit a ridge regression on the train split (after any --vocab_filter) "
        "and keep only the top-N CpG sites by coefficient magnitude. Per CpGPT's FAQ, this "
        "speeds up training substantially with little accuracy cost for a single-target "
        "finetune like chronological_age. Only applies to numeric targets (regression).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.cpgpt_dir))
    try:
        from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
    except ImportError as e:
        msg = (
            f"Could not import CpGPT modules from {args.cpgpt_dir}. "
            "Check --cpgpt_dir points at the cloned CpGPT repo, and that "
            "`poetry install` has been run there."
        )
        raise ImportError(msg) from e

    df = pd.read_feather(args.input)
    if args.target not in df.columns:
        msg = f"Target column '{args.target}' not found in {args.input}. Available columns: {list(df.columns)}"
        raise ValueError(msg)

    if "benchmark" in args.input.name and args.target == "chronological_age":
        log.warning(
            "You're preparing training data from a 'benchmark' file while targeting "
            "chronological_age. The ComputAgeBench benchmark split is meant to stay "
            "HELD OUT for SOTA comparison (script 05) — training the bioage model on "
            "it would make that comparison dishonest. Use the 'train' split "
            "(computage_train_merged.feather) for chronological_age instead, unless "
            "you have a specific, deliberate reason not to.",
        )

    before = len(df)
    df = df[df[args.target].notna()].copy()
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d / %d rows with missing target '%s'", dropped, before, args.target)

    if df[args.target].dtype == object:
        log.info("Target '%s' looks categorical — encoding as integer codes.", args.target)
        df[args.target], categories = pd.factorize(df[args.target])
        log.info("Category mapping: %s", dict(enumerate(categories)))

    if args.vocab_filter and args.vocab_filter.exists():
        import json

        vocab = json.load(open(args.vocab_filter))
        keep_cols = [c for c in df.columns if c in vocab["input"]]
        non_probe_cols = ["sample_id", "species", args.target, "participant_id"]
        keep_cols = list(set(keep_cols) | set(c for c in non_probe_cols if c in df.columns))
        before_cols = df.shape[1]
        df = df[keep_cols]
        log.info("Filtered to vocab: %d -> %d columns", before_cols, df.shape[1])

    train_df, val_df, test_df = participant_level_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    log.info(
        "Split sizes — train: %d, val: %d, test: %d (participant-level, no leakage)",
        len(train_df), len(val_df), len(test_df),
    )

    KNOWN_METADATA_COLS = {
        "sample_id", "species", "participant_id", "dataset_id", "platform_id",
        "tissue", "cell_type", "sex", "condition_name", "condition_class",
        "chronological_age",
    }

    if args.ridge_select_top_n is not None:
        if not pd.api.types.is_numeric_dtype(df[args.target]):
            log.warning(
                "--ridge_select_top_n requires a numeric target; '%s' is not numeric. Skipping ridge selection.",
                args.target,
            )
        else:
            probe_cols = [c for c in train_df.columns if c not in KNOWN_METADATA_COLS]
            selected_probes = select_features_by_ridge(
                train_df, val_df, test_df, args.target, probe_cols,
                top_n=args.ridge_select_top_n, seed=args.seed,
            )
            keep_cols = list(set(selected_probes) | (KNOWN_METADATA_COLS & set(train_df.columns)))
            train_df = train_df[keep_cols]
            val_df = val_df[[c for c in keep_cols if c in val_df.columns]]
            test_df = test_df[[c for c in keep_cols if c in test_df.columns]]
            log.info("Applied ridge-selected columns to all three splits.")

    llm_dependencies_dir = args.cpgpt_dir / "dependencies" / "human"
    if not llm_dependencies_dir.exists():
        msg = (
            f"{llm_dependencies_dir} not found. Run inferencer.download_dependencies('human') "
            "first (see scripts/00_setup_environment.sh next-steps)."
        )
        raise FileNotFoundError(msg)

    embedder = DNALLMEmbedder(dependencies_dir=str(llm_dependencies_dir))
    prober = IlluminaMethylationProber(dependencies_dir=str(llm_dependencies_dir), embedder=embedder)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.empty:
            log.warning("Split '%s' is empty — skipping.", split_name)
            continue

        split_dir = args.out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        tmp_arrow = split_dir / f"_tmp_{split_name}.feather"
        split_df.reset_index(drop=True).to_feather(tmp_arrow)

        log.info("Memory-mapping '%s' split (%d samples) -> %s", split_name, len(split_df), split_dir)
        datasaver = CpGPTDataSaver(
            data_paths=str(tmp_arrow),
            processed_dir=str(split_dir),
            metadata_cols=[args.target],
        )
        datasaver.process_files(prober, embedder)
        tmp_arrow.unlink()

    log.info("Done. Point your experiment YAML's data.train_dir/val_dir/test_dir at: %s", args.out_dir)


if __name__ == "__main__":
    main()
