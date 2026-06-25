# ZyinkLabs CpGPT Finetuning Pipeline

This is the working codebase for finetuning CpGPT (an open-source DNA methylation
foundation model) into two ZyinkLabs models:

1. **Biological age estimator** — predicts biological/epigenetic age from a blood sample.
2. **Disease-risk estimator** — predicts disease/mortality association scores from the
   same sample.

**Version 1 trains exclusively on public data** —
[ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench)
(Kriukov et al., 2024), the same dataset CpGPT's own authors used to achieve their
best published results (2nd place chronological age estimation, 1st place
mortality prediction on the public leaderboard, Biomarkers of Aging Challenge). No
ZyinkLabs wet-lab sample data is used in Version 1 — see `docs/DATASET.md` for the
full reasoning, and `docs/WET_LAB_DATA_SPEC.md` for the standard that will apply
once a real ZyinkLabs cohort exists for Version 2.

It wraps the upstream [CpGPT](https://github.com/lcamillo/CpGPT) repo (MIT licensed,
commercial use permitted) with ZyinkLabs-specific scripts for dataset acquisition,
data preparation, training orchestration, inference, and — critically — benchmarking
against state-of-the-art published clocks using the official ComputAgeBench
methodology, so results are judged against the same yardstick the field already uses.

Training runs on **AWS EC2 (single GPU)** via **Docker Compose**, finetuning the
**large** CpGPT backbone (~100M params) for the strongest result the budget supports.
An automated gate only pushes a finetuned checkpoint to **Hugging Face** if it beats
a validation-loss threshold you define. See `aws/README.md` for the full cloud setup.

## Hard requirement: GPU

Finetuning needs a GPU. Version 1 uses the **large** CpGPT model, which needs 24GB+
VRAM. Dataset download and data-preparation scripts (steps 1-2 below) run fine on
CPU; only `cpgpt-train` and the SOTA benchmark step need the GPU. See `aws/README.md`
for a ready-to-use single-GPU AWS setup (`g6e.xlarge`, 1x L40S, 48GB VRAM).

## Pipeline overview

```
ComputAgeBench (public, CC BY-SA 4.0)   <- see docs/DATASET.md for why this dataset
        |
        v
[1] scripts/01_download_computage_bench.py   (HF Hub: train + benchmark splits)
        |
        v
[2] scripts/02_prepare_cpgpt_data.py         (CpGPT: beta table -> memory-mapped train/val/test)
        |
        v
[3] cpgpt-train experiment=zyinklabs_bioage      (finetune biological age head, large model)
    cpgpt-train experiment=zyinklabs_disease      (finetune disease-risk head, large model)
        |            -- wrapped by docker/orchestrate_training.py, which checks
        |               val/condition_loss against a threshold before pushing --
        v
[4] Hugging Face Hub (only if validation threshold passed)
        |
        +--> [5] scripts/03_run_inference.py        (apply finetuned model to new samples)
        |
        +--> [6] scripts/04_benchmark_vs_sota.py     (compare vs. Horvath/Hannum/PhenoAge
                                                        on the held-out ComputAgeBench
                                                        benchmark split -- the real SOTA check)
```

This whole chain (steps 1-4) runs inside Docker Compose on a single AWS GPU instance —
see `aws/README.md` and `docker/docker-compose.yml`.

## Directory layout

```
cpgpt-pipeline/
├── README.md                              <- you are here
├── scripts/
│   ├── 00_setup_environment.sh            <- one-time environment + dependency setup
│   ├── 01_download_computage_bench.py     <- public dataset download (train + benchmark splits)
│   ├── 02_prepare_cpgpt_data.py           <- beta table -> CpGPT memory-mapped format
│   ├── 03_run_inference.py                <- run a finetuned model on new samples
│   └── 04_benchmark_vs_sota.py            <- benchmark vs. Horvath/Hannum/PhenoAge via `computage`
├── configs/
│   └── experiment/
│       ├── zyinklabs_bioage.yaml          <- Hydra config: biological age finetuning (large model)
│       └── zyinklabs_disease.yaml         <- Hydra config: disease-risk finetuning (large model)
├── docker/
│   ├── Dockerfile                         <- training container (CUDA + CpGPT + ZyinkLabs scripts)
│   ├── docker-compose.yml                 <- download -> prepare -> train -> (validate+push) -> benchmark chain
│   ├── entrypoint.sh                      <- dispatches container behavior by RUN_MODE
│   ├── orchestrate_training.py            <- train, check val-loss gate, push to HF if passed
│   └── .env.example                       <- template for docker/.env (never commit the real one)
├── aws/
│   ├── README.md                          <- full AWS setup & operational runbook
│   ├── launch_instance.sh                 <- AWS CLI script to launch the GPU training instance
│   ├── bootstrap.sh                       <- EC2 user-data: installs Docker + NVIDIA toolkit
│   └── iam_policy_training_instance.json  <- scoped IAM policy for the instance role
├── data/
│   ├── raw/                               <- not used by Version 1 (kept for Version 2 wet-lab data)
│   └── processed/                         <- pipeline outputs (ComputAgeBench data, CpGPT format)
├── notebooks/
│   └── 01_sanity_check_pipeline.ipynb     <- end-to-end dry run on a small data subset
└── docs/
    ├── DATASET.md                         <- why ComputAgeBench, licensing, alternatives considered
    ├── WET_LAB_DATA_SPEC.md               <- ZL-EPI-001: standard for Version 2 wet-lab data
    └── RUNBOOK.md                         <- step-by-step local/manual operational instructions
```

## Quick start — local/manual (no AWS, no Docker)

Useful for dry-running the pipeline logic before touching cloud infra. See
`docs/RUNBOOK.md` for full detail.

```bash
bash scripts/00_setup_environment.sh
python scripts/01_download_computage_bench.py --out_dir data/processed/computage
python scripts/02_prepare_cpgpt_data.py --input data/processed/computage/computage_train_merged.feather --target chronological_age --out_dir data/processed/cpgpt/bioage --ridge_select_top_n 20000
cd CpGPT && cpgpt-train experiment=zyinklabs_bioage
```

## Quick start — AWS + Docker Compose (recommended for real training runs)

```bash
cd aws
./launch_instance.sh                              # launches a g6e.xlarge GPU instance
ssh -i zyinklabs-training.pem ubuntu@<PUBLIC_IP>
cd cpgpt-pipeline/docker
cp .env.example .env && nano .env                 # fill in HF token, repo, thresholds
docker compose run --rm download_dependencies
docker compose run --rm download_computage_bench
docker compose run --rm prepare_data
docker compose run --rm train                      # trains, validates, pushes to HF if it passes
docker compose run --rm --profile benchmark benchmark   # compares vs. SOTA published clocks
```

Full detail, including how to pick `VAL_LOSS_THRESHOLD`, in `aws/README.md`.

## Where to start reading

- **Why this dataset, and licensing implications:** `docs/DATASET.md`
- **Cloud training setup:** `aws/README.md`
- **Manual/local pipeline detail:** `docs/RUNBOOK.md`
- **Wet lab / Version 2 data collection standard:** `docs/WET_LAB_DATA_SPEC.md`
