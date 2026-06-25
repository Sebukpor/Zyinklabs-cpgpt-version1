#!/usr/bin/env python3
"""Benchmark the finetuned ZyinkLabs CpGPT model against state-of-the-art published
epigenetic clocks, using the official ComputAgeBench methodology and held-out
benchmark split.

This is the credibility check for the whole project: it answers "is our finetuned
model actually competitive with the established field," not just "did training
converge." It uses the `computage` library's own `run_benchmark` function, so the
comparison methodology is the same one used in the ComputAgeBench paper itself --
not a custom metric we invented that might flatter our own model.

Compared against, out of the box:
    - HorvathV1   (first-generation chronological age clock)
    - Hannum      (first-generation chronological age clock)
    - PhenoAgeV2  (second-generation, mortality/morbidity-informed)
  (the `computage` library supports additional in-library clocks -- see
  --list_available_clocks)

IMPORTANT: this script must be run against the ComputAgeBench BENCHMARK split only
(computage_benchmark_merged.feather from script 01) -- never the TRAIN split. If the
model was trained on samples that overlap with the benchmark split, this comparison
is meaningless. scripts/02_prepare_cpgpt_data.py already warns if you point training
at the benchmark file; this script is the other half of that guardrail.

Usage:
    python 04_benchmark_vs_sota.py \
        --checkpoint ../CpGPT/logs/experiments/zyinklabs_bioage/<run>/checkpoints/<file>.ckpt \
        --config ../CpGPT/logs/experiments/zyinklabs_bioage/<run>/.hydra/config.yaml \
        --benchmark_data data/processed/computage/computage_benchmark_merged.feather \
        --cpgpt_dir ../CpGPT \
        --out_dir benchmark_results/
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_vs_sota")

DEFAULT_REFERENCE_CLOCKS = ["HorvathV1", "Hannum", "PhenoAgeV2"]


class ZyinkLabsModelWrapper:
    """Wraps our finetuned CpGPT checkpoint in the minimal sklearn-style interface
    (.predict(X)) that `computage.run_benchmark` expects for a 'new_models' entry,
    per the ComputAge library's documented usage pattern.
    """

    def __init__(self, checkpoint_path: Path, config_path: Path, cpgpt_dir: Path) -> None:
        sys.path.insert(0, str(cpgpt_dir))
        from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

        self.cpgpt_dir = cpgpt_dir
        self.inferencer = CpGPTInferencer(
            dependencies_dir=str(cpgpt_dir / "dependencies"),
            data_dir=str(cpgpt_dir / "data"),
        )
        self.config = self.inferencer.load_cpgpt_config(str(config_path))
        self.model = self.inferencer.load_cpgpt_model(
            self.config, model_ckpt_path=str(checkpoint_path), strict_load=True,
        )

    def predict(self, betas: pd.DataFrame) -> np.ndarray:
        """betas: samples x CpG-sites DataFrame of beta values (0-1), as provided
        by the computage benchmark loader.
        """
        from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
        from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
        from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

        df = betas.copy()
        df["species"] = "homo_sapiens"
        df.index.name = "sample_id"
        df = df.reset_index()

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arrow_path = tmp_path / "input.feather"
            df.to_feather(arrow_path)

            llm_dependencies_dir = self.cpgpt_dir / "dependencies" / "human"
            embedder = DNALLMEmbedder(dependencies_dir=str(llm_dependencies_dir))
            prober = IlluminaMethylationProber(dependencies_dir=str(llm_dependencies_dir), embedder=embedder)

            processed_dir = tmp_path / "processed"
            datasaver = CpGPTDataSaver(data_paths=str(arrow_path), processed_dir=str(processed_dir))
            datasaver.process_files(prober, embedder)

            datamodule = CpGPTDataModule(
                predict_dir=str(processed_dir),
                dependencies_dir=str(llm_dependencies_dir),
                batch_size=8,
                num_workers=0,
                max_length=20000,
                dna_llm=self.config.data.dna_llm,
                dna_context_len=self.config.data.dna_context_len,
                sorting_strategy=self.config.data.sorting_strategy,
                pin_memory=False,
            )

            trainer = CpGPTTrainer(precision="16-mixed")
            result = trainer.predict(
                model=self.model,
                datamodule=datamodule,
                predict_mode="forward",
                return_keys=["pred_conditions"],
            )

        return np.asarray(result["pred_conditions"]).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--benchmark_data",
        required=True,
        type=Path,
        help="Path to computage_benchmark_merged.feather from script 01 -- MUST be the "
        "held-out benchmark split, not the training split.",
    )
    parser.add_argument("--cpgpt_dir", type=Path, default=Path("../CpGPT"))
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--reference_clocks",
        nargs="+",
        default=DEFAULT_REFERENCE_CLOCKS,
        help=f"In-library computage clocks to compare against (default: {DEFAULT_REFERENCE_CLOCKS})",
    )
    parser.add_argument(
        "--imputation",
        default="sesame_450k",
        help="NaN imputation strategy for in-library models (computage's recommended default)",
    )
    args = parser.parse_args()

    if "benchmark" not in args.benchmark_data.name:
        log.warning(
            "--benchmark_data filename ('%s') doesn't look like the ComputAgeBench "
            "benchmark split. Double-check this isn't the training split -- comparing "
            "against data the model was trained on invalidates the whole benchmark.",
            args.benchmark_data.name,
        )

    try:
        from computage import run_benchmark
    except ImportError as e:
        msg = "computage is not installed. Run: pip install computage --break-system-packages"
        raise ImportError(msg) from e

    log.info("Loading our finetuned model from %s", args.checkpoint)
    zyinklabs_model = ZyinkLabsModelWrapper(args.checkpoint, args.config, args.cpgpt_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_pkl_path = args.out_dir / "_zyinklabs_model_wrapper.pkl"
    with open(model_pkl_path, "wb") as f:
        pickle.dump(zyinklabs_model, f)

    models_config = {
        "in_library": {name: {"imputation": args.imputation} for name in args.reference_clocks},
        "new_models": {
            "zyinklabs_bioage_v1": {"path": str(model_pkl_path)},
        },
    }

    log.info(
        "Running ComputAgeBench comparison: zyinklabs_bioage_v1 vs. %s",
        args.reference_clocks,
    )
    bench = run_benchmark(
        models_config,
        experiment_prefix="zyinklabs_v1_vs_sota",
        output_folder=str(args.out_dir),
    )

    log.info("Benchmark complete. Results written under: %s", args.out_dir)
    log.info(
        "Read the output tables/plots before trusting this model for anything "
        "customer-facing -- look specifically at how it performs distinguishing "
        "aging-accelerating-condition samples from healthy controls, not just raw "
        "age-prediction error, since that's ComputAgeBench's actual point: a clock "
        "that's accurate on healthy people but doesn't detect known accelerated-aging "
        "conditions is not capturing biological age, just chronological age.",
    )

    return bench


if __name__ == "__main__":
    main()
