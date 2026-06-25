#!/usr/bin/env python3
"""Run a finetuned ZyinkLabs CpGPT checkpoint on new samples and write predictions
to a CSV.

Usage:
    python 03_run_inference.py \
        --checkpoint ../CpGPT/logs/experiments/zyinklabs_bioage/2026-06-22_10-00-00/checkpoints/zyinklabs_bioage_step_020000.ckpt \
        --config ../CpGPT/logs/experiments/zyinklabs_bioage/2026-06-22_10-00-00/.hydra/config.yaml \
        --input data/processed/new_batch_merged.feather \
        --out predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_inference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the finetuned .ckpt file")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the run's .hydra/config.yaml (saved alongside the checkpoint)",
    )
    parser.add_argument("--input", required=True, type=Path, help="Merged beta+metadata table for new samples (.feather)")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path for predictions")
    parser.add_argument("--cpgpt_dir", type=Path, default=Path("../CpGPT"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=20000)
    args = parser.parse_args()

    sys.path.insert(0, str(args.cpgpt_dir))
    from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
    from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
    from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
    from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

    inferencer = CpGPTInferencer(
        dependencies_dir=str(args.cpgpt_dir / "dependencies"),
        data_dir=str(args.cpgpt_dir / "data"),
    )
    config = inferencer.load_cpgpt_config(str(args.config))
    model = inferencer.load_cpgpt_model(config, model_ckpt_path=str(args.checkpoint), strict_load=True)

    df = pd.read_feather(args.input)
    if "sample_id" not in df.columns:
        msg = "Input table must have a 'sample_id' column."
        raise ValueError(msg)

    tmp_dir = args.out.parent / "_tmp_inference"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_arrow = tmp_dir / "input.feather"
    df.reset_index(drop=True).to_feather(tmp_arrow)

    llm_dependencies_dir = args.cpgpt_dir / "dependencies" / "human"
    embedder = DNALLMEmbedder(dependencies_dir=str(llm_dependencies_dir))
    prober = IlluminaMethylationProber(dependencies_dir=str(llm_dependencies_dir), embedder=embedder)

    processed_dir = tmp_dir / "processed"
    datasaver = CpGPTDataSaver(data_paths=str(tmp_arrow), processed_dir=str(processed_dir))
    datasaver.process_files(prober, embedder)

    datamodule = CpGPTDataModule(
        predict_dir=str(processed_dir),
        dependencies_dir=str(llm_dependencies_dir),
        batch_size=args.batch_size,
        num_workers=0,
        max_length=args.max_length,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    trainer = CpGPTTrainer(precision="16-mixed")
    result = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"],
    )

    preds = pd.DataFrame(result["pred_conditions"], index=df["sample_id"])
    preds.columns = [f"pred_{i}" for i in range(preds.shape[1])]
    preds.to_csv(args.out)

    log.info("Wrote %d predictions -> %s", len(preds), args.out)
    log.info(
        "Reminder: compare these against published clocks (PhenoAge, GrimAge2, "
        "DunedinPACE via pyaging, or CpGPT's own `clock_proxies` model) before "
        "trusting absolute values — see docs/RUNBOOK.md Validation section.",
    )


if __name__ == "__main__":
    main()
