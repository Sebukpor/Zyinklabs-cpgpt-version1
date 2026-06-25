# Dataset Selection — ComputAgeBench (Version 1)

## Decision

Version 1 of the ZyinkLabs biological age / disease-risk model finetunes CpGPT
exclusively on **[ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench)**
(Kriukov et al., 2024), accessed via the `computage` Python library and Hugging
Face Hub. No ZyinkLabs wet-lab data is used in Version 1.

## Why ComputAgeBench, specifically

There are many public DNA methylation datasets (individual GEO series, dbGaP/EGA
cohorts, etc.) that could plausibly serve as v1 training data. ComputAgeBench was
chosen over assembling an ad hoc GEO selection for four concrete reasons:

1. **It's the dataset CpGPT's own authors used for their best published results.**
   CpGPT was finetuned on ComputAgeBench data for the "Biomarkers of Aging
   Challenge," an open competition for epigenetic age/mortality prediction models.
   In that competition, CpGPT placed **2nd overall in chronological age
   estimation** (MAE of 2.55 years) and **1st on the public leaderboard for
   methylation-based mortality prediction**. Using the same training data as the
   model's own best-documented result gives us the closest thing to a validated
   recipe to start from, rather than an unproven combination of model + novel
   dataset.

2. **It's curated specifically for clock benchmarking, not just age labels.**
   Most GEO series only give chronological age. ComputAgeBench's defining
   methodology is comparing model predictions between healthy controls and
   patients with **aging-accelerating conditions (AACs)** — the dataset was built
   around the idea that a real biological age signal should detect known
   accelerated-aging conditions, not just track birthdays. This gives us a
   training set already structured around the right scientific question, and a
   built-in, methodologically rigorous way to check our model against that
   standard (see below).

3. **It ships with an official, independent benchmarking methodology.**
   The companion `computage` library (`pip install computage`) implements
   `run_benchmark()`, which scores any new model — including ours — against
   Horvath, Hannum, PhenoAge, and other published clocks using the exact
   methodology from the ComputAgeBench paper. This means our SOTA comparison
   (`scripts/04_benchmark_vs_sota.py`) is not a metric we invented to flatter our
   own results; it's the same yardstick the field already uses to compare clocks
   against each other.

4. **It's free and large enough to be useful.** CC BY-SA 4.0 license — no fee,
   only attribution + share-alike requirements (see the Licensing section below).
   The training split alone is **7,419 samples across 46 studies**; the held-out
   benchmark split is **10,410 samples across 65 studies**. This is a larger and
   more methodologically diverse pool than any single GEO accession, without
   needing to manually harmonize dozens of studies' worth of platform and
   normalization differences ourselves — ComputAgeBench's authors already did
   that harmonization work.

## Dataset structure

| Split | Samples | Studies | Composition | Use |
|---|---|---|---|---|
| `train` | 7,419 | 46 | Healthy controls only | Finetuning (script 02) |
| `benchmark` | 10,410 | 65 | Healthy controls + 19 aging-accelerating condition categories | Held-out SOTA comparison only (script 04) — **never used for training** |

Both splits span Illumina 27K, 450K, and EPIC/EPICv2 array generations, in blood,
saliva, and buccal tissue (predominantly blood). Full feature/platform detail is in
the dataset's own card on Hugging Face.

## Why not other options

- **Hand-picking individual GEO series** (e.g. GSE40279, GSE87571): viable, and
  was the original plan for this project, but means re-doing the harmonization,
  QC, and benchmark-construction work ComputAgeBench's authors already published
  and validated. It also means inventing our own evaluation methodology instead of
  using one the field already trusts.
- **dbGaP / EGA controlled-access cohorts** (Framingham, GOLDN, WHI, etc.): richer
  longitudinal phenotype data, genuinely valuable for a future disease-risk model,
  but requires an institutional data use agreement and an approval/review period —
  not compatible with training Version 1 quickly on public data alone. Worth
  revisiting for Version 2.
- **A different foundation-model finetuning dataset entirely**: no other
  public dataset currently has both the scale and the "aging-accelerating
  condition" structure that ComputAgeBench provides, as far as this research
  identified at time of writing.

## Licensing — read before any redistribution

ComputAgeBench is released under **CC BY-SA 4.0** (Creative Commons
Attribution-ShareAlike). This is meaningfully different from CpGPT's own MIT
license:

- **Attribution required**: any use must credit the ComputAgeBench paper (citation
  below).
- **ShareAlike**: derivative works — which a model finetuned on this data may
  count as — may need to be released under a compatible license if redistributed.

This combination (MIT base model + CC BY-SA training data) needs a legal/compliance
review before any commercial redistribution of the finetuned weights beyond internal
ZyinkLabs use. The model card automatically generated by
`docker/orchestrate_training.py` includes the required citation as a starting point,
but does not substitute for that review.

**Citation:**

> Kriukov, D., Efimov, E., Kuzmina, E. A., Khrameeva, E. E., & Dylov, D. V. (2024).
> ComputAgeBench: Epigenetic Aging Clocks Benchmark. bioRxiv, 2024-06.

```bibtex
@article{kriukov2024computagebench,
  title={ComputAgeBench: Epigenetic Aging Clocks Benchmark},
  author={Kriukov, Dmitrii and Efimov, Evgeniy and Kuzmina, Ekaterina A and Khrameeva, Ekaterina E and Dylov, Dmitry V},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Sources

- Dataset: https://huggingface.co/datasets/computage/computage_bench
- Library/code: https://github.com/ComputationalAgingLab/ComputAge
- Paper: https://doi.org/10.1101/2024.06.06.597715
- CpGPT's use of this data for its own Biomarkers of Aging Challenge result:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11565782/
