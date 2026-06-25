# ZyinkLabs CpGPT Training on AWS — Setup & Runbook

This sets up a single-GPU AWS EC2 instance, runs the full CpGPT finetuning pipeline
inside Docker Compose — training on the public **ComputAgeBench** dataset only — and
**only if the trained model passes a validation threshold**, pushes the resulting
checkpoint to a Hugging Face model repo.

## Architecture at a glance

```
                AWS EC2 (g6e.xlarge, 1x L40S 48GB)
                Docker Compose chain:

  S3 (CpGPT pretrained weights) ──▶ download_dependencies ──┐
                                                              ├──▶ prepare_data ──▶ train ──▶ [pass?] ──▶ Hugging Face Hub
  HF Hub (ComputAgeBench data) ───▶ download_computage_bench ┘                      │
                                                                                      └─ [fail] ──▶ exit 2, nothing pushed

  Run explicitly after a push: benchmark (vs. SOTA clocks, on the held-out split)
```

The `train` and `benchmark` services are the only ones that need the GPU hot —
`download_dependencies`, `download_computage_bench`, and `prepare_data` all run on
CPU. If you want to minimize GPU billing, you can in principle run those CPU stages
on a cheap instance and only spin up the GPU box for `train`/`benchmark`; the
simpler version below just runs everything on one instance for clarity.

## 1. Prerequisites

- An AWS account with permission to create EC2 instances, IAM roles, and security
  groups.
- AWS CLI installed and configured locally (`aws configure`) — used to launch the
  instance.
- An EC2 key pair for SSH:
  ```bash
  aws ec2 create-key-pair --key-name zyinklabs-training \
      --query 'KeyMaterial' --output text > zyinklabs-training.pem
  chmod 400 zyinklabs-training.pem
  ```
- A Hugging Face account + a **write-access** token from
  https://huggingface.co/settings/tokens (needed both to download ComputAgeBench
  smoothly and to push the finetuned model).
- No wet-lab data is needed for Version 1 — see `docs/DATASET.md` for why
  ComputAgeBench was chosen as the training data instead.

## 2. Launch the EC2 instance

```bash
cd aws
chmod +x launch_instance.sh bootstrap.sh
./launch_instance.sh
```

This script (see `aws/launch_instance.sh` for full detail):

1. Looks up the latest AWS Deep Learning AMI (Ubuntu 22.04, NVIDIA drivers + CUDA
   preinstalled).
2. Creates a scoped IAM role (`aws/iam_policy_training_instance.json`) — read access
   to CpGPT's public S3 bucket, read/write to your own ZyinkLabs data bucket if you
   use one (**edit the placeholder bucket name in that file before running, or
   remove that statement if not needed for Version 1**), and scoped CloudWatch
   logging. No broad S3/EC2 permissions.
3. Creates a security group allowing SSH only from your current IP.
4. Launches a **`g6e.xlarge`** instance (1x NVIDIA L40S, 48GB VRAM — sized with real
   headroom above CpGPT's documented 24GB minimum for the large model. `g5.xlarge`'s
   A10G has only ~22.35GiB usable, under that minimum, so it's deliberately not used
   for Version 1 — see the instance sizing table below for the full comparison).
5. Passes `aws/bootstrap.sh` as EC2 user-data, which automatically installs Docker,
   the NVIDIA Container Toolkit, and clones this repo on first boot.

Bootstrap takes 3-5 minutes. Watch its progress:

```bash
ssh -i zyinklabs-training.pem ubuntu@<PUBLIC_IP> 'tail -f /var/log/zyinklabs-bootstrap.log'
```

**When you're done training, terminate the instance** — GPU instances bill per
second while running:

```bash
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

### Instance sizing reference

| Instance | GPU | VRAM | Fits |
|---|---|---|---|
| `g5.xlarge` | 1x A10G | 22.35 GiB usable | **Not used for Version 1** — under CpGPT's 24GB minimum for the large model, no real margin |
| `g6e.xlarge` (selected for v1) | 1x L40S | 48 GB | CpGPT `large` (~100M params) with comfortable headroom — current default in `launch_instance.sh` |
| `g6.xlarge` | 1x L4 | 24 GB | Right at the minimum for `large` — only consider if cost-constrained and willing to tune batch size carefully |
| `g6e.2xlarge` | 1x L40S | 48 GB | Same GPU, more vCPU/RAM — use if dataset download/prep feels CPU-bound |

Check current pricing for your region before committing —
https://aws.amazon.com/ec2/pricing/on-demand/.

### Plain Ubuntu AMI (if not using the Deep Learning AMI)

If you launch on a plain Ubuntu AMI instead, you must install NVIDIA drivers
yourself before `bootstrap.sh`'s Docker GPU steps will work:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535   # or latest available for your GPU
sudo reboot
nvidia-smi   # confirm it works post-reboot
```

## 3. Configure the pipeline

SSH into the instance:

```bash
ssh -i zyinklabs-training.pem ubuntu@<PUBLIC_IP>
cd cpgpt-pipeline/docker
cp .env.example .env
nano .env
```

Fill in at minimum:
- `EXPERIMENT_NAME` — `zyinklabs_bioage` or `zyinklabs_disease`
- `TARGET_COLUMN` — `chronological_age` (bioage) or `condition_class` (disease)
- `VAL_LOSS_THRESHOLD` — see Section 5 below before picking a number
- `HF_TOKEN`, `HF_REPO_ID` — your Hugging Face write token and target repo

No data upload step is needed — `download_computage_bench` (next section) pulls
the public dataset directly onto the instance.

## 4. Run the pipeline

From `cpgpt-pipeline/docker/`:

```bash
# One-time: pull CpGPT's pretrained weights + DNA LLM embeddings
docker compose run --rm download_dependencies

# One-time: pull the public ComputAgeBench dataset (train + benchmark splits, ~38.6GB)
docker compose run --rm download_computage_bench

# Beta-value table -> CpGPT memory-mapped format
docker compose run --rm prepare_data

# Finetune, validate against the threshold, and (if it passes) push to Hugging Face
docker compose run --rm train
```

Or run the dependency-ordered chain in one shot (covers everything up to and
including `train`; `benchmark` and `inference` are profile-gated and run
separately, see below):

```bash
docker compose up
```

**What "successful" means here, concretely:** the `train` service does not just run
`cpgpt-train` and stop. `docker/orchestrate_training.py` (baked into the image)
wraps it:

1. Runs `cpgpt-train experiment=$EXPERIMENT_NAME`.
2. Parses the run's `metrics.csv` for the best (lowest) `val/condition_loss`.
3. Compares it against `VAL_LOSS_THRESHOLD` from your `.env`.
4. **Only if `val/condition_loss <= VAL_LOSS_THRESHOLD`**, generates a model card
   (citing ComputAgeBench, noting the CC BY-SA 4.0 license, and pointing at the
   SOTA benchmark script) and pushes the checkpoint + config + card to `HF_REPO_ID`
   via `huggingface_hub`.
5. If it fails the threshold, the container exits with code `2` and **nothing is
   pushed** — check `docker compose logs train` for the reported value.

A `zyinklabs_result_summary.json` is written into the run's log directory either
way, recording the actual loss, threshold, and pass/fail outcome — useful for
tracking threshold calibration over multiple runs.

### Benchmarking against SOTA after a successful push

```bash
docker compose run --rm \
  -e CHECKPOINT_PATH=/workspace/CpGPT/logs/experiments/zyinklabs_bioage/<run>/checkpoints/<file>.ckpt \
  -e CONFIG_PATH=/workspace/CpGPT/logs/experiments/zyinklabs_bioage/<run>/.hydra/config.yaml \
  --profile benchmark benchmark
```

This runs `scripts/04_benchmark_vs_sota.py` inside the same container image,
comparing the model against Horvath, Hannum, and PhenoAge on the held-out
ComputAgeBench benchmark split using the official `computage` library methodology.
**This is the step that actually tells you whether the model is competitive** —
passing the training validation-loss threshold only confirms the model fits its own
held-out training-split validation data, not that it's state-of-the-art.

### Debugging inside the container

```bash
docker compose run --rm --profile debug shell
```

Drops you into an interactive shell inside the training image with the GPU
attached — useful for poking at CpGPT internals or testing config changes before a
full run.

## 5. Picking a sensible `VAL_LOSS_THRESHOLD`

Don't guess this number — calibrate it from at least one exploratory run:

1. Set `SKIP_PUSH=true` in `.env` and run `docker compose run --rm train` once.
2. Look at the printed `Best val/condition_loss observed: ...` line, and at
   `CpGPT/logs/experiments/<experiment>/<run>/zyinklabs_result_summary.json`.
3. For the **bioage** experiment (`condition_decoder_loss: mae`), this value is in
   *years of mean absolute age error* — e.g. `5.0` means predictions are off by
   ~5 years on average on the validation split. CpGPT's own reported MAE on the
   Biomarkers of Aging Challenge using this same training methodology was **2.55
   years** — a useful anchor for what "competitive" looks like, though our exact
   number will depend on vocab filtering, step count, and batch size choices.
4. For the **disease** experiment (`condition_decoder_loss: ce`), this is
   cross-entropy — harder to interpret directly; look at the actual val accuracy/AUC
   if you add that metric, or treat the first run's CE as a baseline and tighten the
   threshold on subsequent runs as the model improves.
5. Set `VAL_LOSS_THRESHOLD` to something a *clearly worse* run should fail and a
   *clearly acceptable* run should pass — then set `SKIP_PUSH=false` for real runs.

Revisit the threshold after the first full run, and again after running
`scripts/04_benchmark_vs_sota.py` — a training-split validation loss that looks
good can still under-perform on the held-out benchmark split, which is exactly why
both checks exist.

## 6. After a successful push

Your model is now at `https://huggingface.co/<HF_REPO_ID>`, containing:
- The checkpoint (`.ckpt`) — this is the full finetuned model weights (CpGPT
  finetuning updates the base model directly; there is no separate adapter/LoRA
  layer to merge, so this checkpoint already is the "merged" weights).
- `config.yaml` — the exact Hydra config used for the run.
- `README.md` — an auto-generated model card noting the validation metric,
  threshold, ComputAgeBench citation/license obligations, and important caveats
  (including that passing the loss threshold does **not** by itself confirm
  state-of-the-art performance — run `scripts/04_benchmark_vs_sota.py` separately,
  see Section 4 above).

To pull it back down elsewhere:

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="zyinklabs/cpgpt-bioage-v1", filename="<checkpoint_filename>.ckpt")
```

**Before any commercial/customer-facing release**, confirm with legal/compliance on
the CC BY-SA 4.0 obligations inherited from the ComputAgeBench training data — see
`docs/DATASET.md`.

## 7. Cost hygiene

- **Terminate the instance** when not actively training — `g6e.xlarge` bills
  continuously while running regardless of GPU utilization.
- The ComputAgeBench download (~38.6GB) is a one-time cost per instance — cached
  under `../data`. If you terminate and relaunch a fresh instance, you'll
  re-download it unless you snapshot the volume or move the data to S3 in between.
- Consider an EC2 **Spot Instance** for non-urgent training runs (interruptible, but
  meaningfully cheaper) — re-run `launch_instance.sh` with
  `aws ec2 run-instances ... --instance-market-options '{"MarketType":"spot"}'`
  added if you want to explore this; checkpointing in `cpgpt-train` (via
  `min_steps`/`max_steps` and the model checkpoint callback) means a Spot
  interruption loses at most the most recent unsaved steps, not the whole run.
- The CpGPT pretrained dependencies and DNA LLM embeddings are cached in the
  `cpgpt_dependencies` Docker volume — re-running `download_dependencies` after the
  first time is a no-op cost-wise (skips re-download), but the volume persists only
  as long as the underlying EBS volume does, i.e. until you terminate the instance.
  For a longer-lived setup, consider snapshotting that volume or pointing it at S3.
