# ZyinkLabs CpGPT Pipeline — Runbook

This is the step-by-step operational guide. Read `README.md` first for the high-level
picture; this document covers the details, decisions, and troubleshooting.

> **Version 1 scope:** this pipeline trains exclusively on public data
> ([ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench)) — no
> ZyinkLabs wet-lab samples are used. See `docs/DATASET.md` for why this dataset was
> chosen and how it compares to alternatives. The wet lab specification
> (`docs/WET_LAB_DATA_SPEC.md`) remains the standard for Version 2, once a real
> ZyinkLabs cohort exists.

## 0. Before you start

- **GPU required for finetuning.** This pipeline can be set up and dry-run (steps 1-3)
  on a CPU-only machine, but `cpgpt-train` (step 4) needs a GPU. Version 1 finetunes
  the **large** CpGPT backbone (~100M params), which needs 24GB+ VRAM. See
  `aws/README.md` for the recommended `g6e.xlarge` (48GB) setup.
- **AWS account needed** (free tier is fine) to download CpGPT's pretrained weights —
  they're hosted on a public S3 bucket but still require AWS CLI authentication.
- **Hugging Face account needed** with a write-access token, for both downloading the
  ComputAgeBench dataset (public, no token strictly required, but recommended to
  avoid rate limits) and pushing the finetuned checkpoint.

## 1. Environment setup

```bash
bash scripts/00_setup_environment.sh
```

This clones CpGPT, installs its dependencies via Poetry, installs ZyinkLabs-specific
dependencies (`computage` for dataset access and SOTA benchmarking, `huggingface_hub`
for dataset/model transfer), and installs the AWS CLI if missing.

After this, run `aws configure` if you haven't already, then download the pretrained
weights and dependencies (see the script's printed next-steps, or run directly):

```python
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir="CpGPT/dependencies", data_dir="CpGPT/data")
inferencer.download_dependencies(species="human")
inferencer.download_model("large")           # Version 1 default -- see docs/DATASET.md
inferencer.download_model("clock_proxies")   # useful reference for calibration checks
```

## 2. Download the ComputAgeBench dataset

```bash
python scripts/01_download_computage_bench.py \
    --out_dir data/processed/computage
```

This downloads both ComputAgeBench splits via `huggingface_hub`:
- **`train`** (7,419 samples, 46 studies, healthy controls only) — used for finetuning.
- **`benchmark`** (10,410 samples, 65 studies, healthy controls + aging-accelerating
  conditions) — held out for SOTA comparison in step 6. **Never train on this split.**

The script reshapes both into the same `sample_id`-indexed beta-value + metadata
table format the rest of the pipeline expects, and writes them as
`computage_train_merged.feather` / `computage_benchmark_merged.feather`.

**Note on size:** the full snapshot is ~38.6GB. Make sure your disk has headroom
(the AWS setup in `aws/README.md` provisions 200GB by default, which covers this
comfortably alongside CpGPT's own dependencies and checkpoints).

## 3. Prepare CpGPT-format data

```bash
python scripts/02_prepare_cpgpt_data.py \
    --input data/processed/computage/computage_train_merged.feather \
    --target chronological_age \
    --out_dir data/processed/cpgpt/bioage \
    --cpgpt_dir ../CpGPT \
    --ridge_select_top_n 20000
```

**`--ridge_select_top_n` is recommended, not optional, for a serious run.** CpGPT's
own FAQ explicitly recommends fitting a ridge regression and keeping the top-N CpG
sites by coefficient magnitude when finetuning for a specific phenotype target —
this both speeds up training substantially and, since ComputAgeBench's outer-join
concat across 46 studies leaves many CpG sites mostly missing for any given sample,
focuses the model on sites with actual signal and reasonable coverage. Selection is
always fit on the **train split only**, then applied identically to val/test, so no
information about the held-out splits leaks into which features get chosen.

For the disease-risk model, point `--input` at the benchmark split's labeled
conditions and swap the target (omit `--ridge_select_top_n`, since it currently only
supports numeric/regression targets) — but read the warning in
`scripts/02_prepare_cpgpt_data.py` first: the benchmark split is also what
step 6 uses for SOTA comparison, so training a disease model on it needs a careful,
deliberate train/benchmark partition, not the default flow used for biological age.

**Key design decision baked into this script:** splits are done at the
**participant level**, not the sample level — a safety net carried over from the
original wet-lab pipeline design. ComputAgeBench is mostly one-sample-per-donor, so
in practice this behaves like a normal random split here, but the protection stays
in place in case any constituent study contributes repeat samples.

**Optional but recommended — vocab filtering:** if you pass `--vocab_filter` pointing
at a downloaded model's `vocab.json` (e.g. the `clock_proxies` model's vocab), the
script filters CpG sites down to that model's feature set. Per CpGPT's own FAQ, this
gives the best chance of strong finetuning performance.

## 4. Finetune (GPU required)

```bash
cd CpGPT
cpgpt-train experiment=zyinklabs_bioage      # copy configs/experiment/zyinklabs_bioage.yaml
                                              # into CpGPT/configs/experiment/ first
```

Copy both `configs/experiment/zyinklabs_bioage.yaml` and
`configs/experiment/zyinklabs_disease.yaml` from this repo into `CpGPT/configs/experiment/`
before running — Hydra resolves `experiment=` against that directory. Both configs
are set to the **large** backbone for Version 1.

**Before launching a long run:**
- The bioage config already ships `max_steps: 100000`, grounded in the CpGPT FAQ
  formula (`max_steps ≈ (N_samples * 50) / batch_size` → ≈46,400 for the
  ComputAgeBench training split's 7,419 samples at `batch_size: 8`), with deliberate
  headroom toward CpGPT's own `template.yaml` general-purpose default of 100,000 —
  because `generative_splits: 5` (also set in the config, per the FAQ's explicit
  "best possible performance" guidance) means each step sees a different sub-sample
  of each input, which argues for more steps than the naive formula alone suggests.
  See the comments in `configs/experiment/zyinklabs_bioage.yaml` for the full
  reasoning, and recompute if your actual post-feature-selection sample count
  differs meaningfully from 7,419 (e.g. if you filter to a subset for a vocab match).
- Watch `val/condition_loss` — if it's not decreasing within the first ~10% of
  planned steps, stop and check the data pipeline before continuing (most "the model
  doesn't learn" issues trace back to a data prep bug, not an architecture problem).
- Watch GPU memory via `nvidia-smi` during the first few hundred steps — the large
  model on `g6e.xlarge`'s 48GB should have headroom at `batch_size: 8`, but confirm
  before committing to a long run. If `--ridge_select_top_n` (step 3) kept your
  per-sample feature count well below full EPIC coverage, you likely have room to
  increase `batch_size` for faster wall-clock training.

**Checkpoints land at:**
```
CpGPT/logs/experiments/<experiment_name>/<timestamp>/checkpoints/<filename>.ckpt
CpGPT/logs/experiments/<experiment_name>/<timestamp>/.hydra/config.yaml
```

If running via Docker Compose instead of manually, this whole step — including the
validation gate and conditional Hugging Face push — is handled by
`docker/orchestrate_training.py`; see `aws/README.md`.

## 5. Inference on new samples

```bash
python scripts/03_run_inference.py \
    --checkpoint CpGPT/logs/experiments/zyinklabs_bioage/<timestamp>/checkpoints/<file>.ckpt \
    --config CpGPT/logs/experiments/zyinklabs_bioage/<timestamp>/.hydra/config.yaml \
    --input data/processed/new_batch_merged.feather \
    --out predictions.csv
```

## 6. Benchmark against state-of-the-art published clocks

```bash
python scripts/04_benchmark_vs_sota.py \
    --checkpoint CpGPT/logs/experiments/zyinklabs_bioage/<timestamp>/checkpoints/<file>.ckpt \
    --config CpGPT/logs/experiments/zyinklabs_bioage/<timestamp>/.hydra/config.yaml \
    --benchmark_data data/processed/computage/computage_benchmark_merged.feather \
    --cpgpt_dir ../CpGPT \
    --out_dir benchmark_results/
```

This is the credibility check, and it uses the **official ComputAgeBench methodology**
via the `computage` library — the same one used in the published paper — rather than
a metric we invented ourselves. It compares our finetuned model against Horvath,
Hannum, and PhenoAge (configurable via `--reference_clocks`) on the held-out benchmark
split.

**Critically, this is not just an age-prediction-error comparison.** ComputAgeBench's
actual point is whether a clock can distinguish people with aging-accelerating
conditions from healthy controls — a clock that's accurate on healthy people but
doesn't detect known accelerated-aging conditions is capturing chronological age, not
biological age. Read the output tables with that distinction in mind.

**Do this before showing results to anyone outside the data science team.** A model
that doesn't perform competitively against established clocks on this benchmark needs
investigation before it's trusted for anything customer-facing — and definitely
before treating a Hugging Face push as "done."

## 7. Dry-run validation (recommended before a full-scale run)

Before committing a full GPU-hour budget to training, run the pipeline end to end
on a small subset to catch bugs cheaply — e.g. by setting `trainer.max_steps` very
low in the experiment config temporarily, or slicing `computage_train_merged.feather`
down to a few hundred rows before step 3. `notebooks/01_sanity_check_pipeline.ipynb`
walks through this.

## Decision log / open items to revisit

- **Dataset choice:** ComputAgeBench was selected because it's the same dataset
  CpGPT's authors used to achieve their best published results (2nd place
  chronological age, 1st place mortality prediction on the public leaderboard in the
  Biomarkers of Aging Challenge), it's free (CC BY-SA 4.0, no fee), and it ships with
  an official benchmarking methodology so our SOTA comparison isn't self-graded. See
  `docs/DATASET.md` for the full reasoning and alternatives considered.
- **Model size:** Large backbone (~100M params), chosen for the strongest
  production-quality result this pipeline can produce. Needs 24GB+ VRAM — see
  `aws/README.md` for the `g6e.xlarge` instance this assumes, and why `g5.xlarge`
  (under the minimum, no real margin) was deliberately rejected.
- **Disease-risk model:** ComputAgeBench's `Condition`/`Class` columns give
  diagnosis-style labels but not the longitudinal incident/outcome structure a true
  mortality or incident-disease model benefits from (the kind GrimAge-style models
  use). Treat the disease-risk model as more exploratory than the biological age
  model for Version 1, and revisit outcome-labeled cohort options (e.g. licensed
  biobank data) for Version 2 if this matters for the product.
- **License compliance:** ComputAgeBench is CC BY-SA 4.0 (attribution + share-alike).
  This has real implications for how a model finetuned on it can be redistributed —
  confirm with legal/compliance before any commercial release beyond internal use.
  See the model card generated by `docker/orchestrate_training.py` for the citation
  to include.
