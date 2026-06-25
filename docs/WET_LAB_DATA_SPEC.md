# ZyinkLabs — Wet Lab Sample & Metadata Submission Specification

**Document ID:** ZL-EPI-001
**Version:** 1.0
**Effective date:** 21 June 2026
**Owner:** ZyinkLabs Data Science / Bioinformatics
**Distribution:** Wet Lab Technicians, Sample Coordinators, QA, Bioinformatics

**Purpose:** Defines exactly what samples, metadata, and file formats the wet lab
must provide to support training and validation of ZyinkLabs' custom biological age
and disease-risk models.

---

## Table of Contents

1. [Purpose & How This Data Will Be Used](#1-purpose--how-this-data-will-be-used)
2. [Methylation Platform & Assay Requirements](#2-methylation-platform--assay-requirements)
3. [Fixed Processing Protocol (SOP)](#3-fixed-processing-protocol-sop)
4. [Sample Metadata — What Must Be Recorded](#4-sample-metadata--what-must-be-recorded)
5. [Sample Handling, Labeling & Chain of Custody](#5-sample-handling-labeling--chain-of-custody)
6. [Data & File Submission](#6-data--file-submission)
7. [Consent, Privacy & De-identification](#7-consent-privacy--de-identification)
8. [Pre-Submission Checklist](#8-pre-submission-checklist)
9. [Questions & Escalation](#9-questions--escalation)

---

## 1. Purpose & How This Data Will Be Used

ZyinkLabs is building a custom epigenetic aging and disease-risk model by
fine-tuning a pretrained DNA methylation foundation model on our own cohort. The
model will be used to:

- Estimate **biological age** from a blood or saliva sample.
- Estimate **disease-risk and morbidity association scores** (e.g. cardiometabolic,
  inflammatory conditions) from the same sample.

The quality of both models depends entirely on the quality and completeness of what
the wet lab provides. A methylation array result with no phenotype data attached, or
with inconsistent processing, is not usable for model training — even if the assay
itself worked perfectly.

> **CRITICAL:** Every sample submitted for this program must be accompanied by a
> complete metadata record (Section 4) and must follow the fixed processing protocol
> (Section 3). Samples that do not meet the minimum mandatory fields in Section 4.2
> will be excluded from the training set.

## 2. Methylation Platform & Assay Requirements

### 2.1 Approved Platform

All samples for this program must be run on the **Illumina Infinium Methylation EPIC
v2.0 BeadChip ("EPIC v2")**. Do not substitute platforms (e.g. legacy EPIC v1, 450K,
WGBS, RRBS, or targeted panels) without prior written sign-off from Bioinformatics —
mixing platforms within a cohort introduces probe-mapping inconsistencies that
materially degrade model quality.

### 2.2 Sample Material

| Specimen Type | Requirement | Notes |
|---|---|---|
| Whole blood | Preferred. EDTA tube (lavender top), 2–4 mL | Highest data quality; matches majority of public reference cohorts |
| Buffy coat / isolated leukocytes | Acceptable if blood unavailable | Note isolation method in metadata |
| Saliva | Acceptable only with prior approval | Higher epithelial/immune cell-type variability; flag explicitly |
| Dried blood spot | Not approved for this program | Lower DNA yield/quality; excluded for now |

### 2.3 Minimum DNA Quality Thresholds

| Metric | Minimum Threshold | Action if Below Threshold |
|---|---|---|
| DNA concentration | ≥ 50 ng/µL | Re-extract; do not submit |
| Total DNA yield | ≥ 500 ng (for bisulfite conversion + array) | Re-extract or pool replicate draws |
| Purity (A260/A280) | 1.7 – 2.0 | Re-purify if outside range |
| Integrity | No visible degradation on gel/Tapestation; gDNA band intact | Re-extract if degraded |

> **NOTE:** Record the actual measured values for every sample — these go into the
> metadata file as QC fields, not just a pass/fail.

## 3. Fixed Processing Protocol (SOP)

This sequence must be identical for every sample in the cohort, including any
public/reference samples processed in-house for comparison. Consistency here matters
more than any single step being "better" — batch effects from inconsistent
processing are the single most common reason in-house epigenetic models fail to
validate.

### 3.1 Required Sequence

1. Sample intake & chain-of-custody logging (Section 5)
2. DNA extraction (standardized kit/protocol — confirm current approved kit with
   Bioinformatics before starting if unsure)
3. DNA QC — concentration, purity, integrity (Section 2.3)
4. Bisulfite conversion
5. EPIC v2 array hybridization
6. Scanning (iScan or equivalent) — raw IDAT file generation
7. Upload of raw IDAT files + metadata to the designated secure data repository
   (Section 6)

### 3.2 Batch Design Requirements

- Every plate/batch must include at least one **technical control sample** (a
  reference DNA sample run repeatedly across all batches) to allow batch-effect
  correction downstream.
- **Randomize** case/control and demographic groups across plates and chip
  positions — do not process all of one group (e.g. all older participants, or all
  diseased participants) on the same plate or chip. Confounding batch with biology
  cannot be corrected for after the fact.
- Record **plate ID, chip ID, and well/position** for every sample (Section 4.3) —
  this is mandatory, not optional, even though it feels like overhead.
- Where possible, run **technical replicates** (same DNA, separate bisulfite
  conversion + array run) for at least 5% of samples to support
  reliability/reproducibility analysis.

## 4. Sample Metadata — What Must Be Recorded

This is the section that determines whether a sample is usable for model training.
Methylation data without matching phenotype/outcome metadata can only be used for
unsupervised pretraining-style tasks, not for the supervised age and disease-risk
models we are building.

### 4.1 Submission Format

- One row per sample in a **single spreadsheet** (.csv or .xlsx) — do not split
  across multiple files per batch.
- Use the **exact column headers** specified in Sections 4.2–4.4 (case-sensitive, no
  spaces — use underscores).
- Dates in **ISO 8601 format**: `YYYY-MM-DD`.
- **Leave genuinely unknown fields blank** — do not enter "N/A", "0", "unknown", or
  "999" as placeholder values; these get misread as real data by automated
  pipelines.
- One unique `sample_id` per row, matching the physical sample label and the IDAT
  filename prefix exactly.

### 4.2 Mandatory Fields (every sample must have these, or it is excluded)

| Field name | Format / Example | Description |
|---|---|---|
| `sample_id` | `ZL-0001` | Unique identifier. Must match IDAT filename and physical label. |
| `participant_id` | `P-00042` | De-identified participant code (see Section 7). Allows linking repeat samples from the same person. |
| `collection_date` | `2026-06-21` | Date sample was drawn. |
| `chronological_age` | `47.3` | Age in years at collection_date, to 1 decimal place if available. |
| `sex` | `F / M / Intersex / Unknown` | Biological sex at birth, not gender identity. Required — methylation patterns differ by sex and the model corrects for this. |
| `specimen_type` | `whole_blood / buffy_coat / saliva` | As defined in Section 2.2. |
| `plate_id`, `chip_id`, `position` | `Plate03`, `203923450012`, `R03C01` | Three separate columns. Required for batch-effect correction. |
| `dna_conc_ng_ul`, `dna_purity_a260_280` | `62.4`, `1.84` | QC values from Section 2.3, recorded as numbers. |
| `consent_status` | `consented_research_use` | Confirmation that informed consent covers algorithm training use — see Section 7. |

### 4.3 Strongly Recommended Fields (needed for biological age model calibration)

- `BMI`, `height_cm`, `weight_kg`
- `self_reported_ethnicity` (free text, plus a standardized category column if your
  intake form supports it — methylation clocks can show ethnicity-related
  calibration drift, so this is needed to check for and correct it, not to be used
  for exclusion)
- `smoking_status` (`never` / `former` / `current`) and `pack_years` if current or
  former
- `alcohol_use` (standardized drinks/week or local equivalent)
- `fasting_status` at draw (`fasted` / `non-fasted`), since this affects some blood
  biomarkers used for biological age calibration
- Standard clinical biomarkers if available from the same draw or a closely dated
  one: lipid panel (total cholesterol, HDL, LDL, triglycerides), HbA1c or fasting
  glucose, CRP (C-reactive protein), creatinine, albumin, white blood cell
  count/differential

> **NOTE:** These clinical biomarkers matter because published biological-age
> algorithms (e.g. PhenoAge, GrimAge) are themselves built by first predicting these
> exact biomarkers from methylation. If you can capture them, even on a subset of
> participants, it substantially increases what we can validate and calibrate.

### 4.4 Disease-Risk Model Fields (required only for the disease/outcomes component)

This is the harder data to gather and the more valuable. A model that flags disease
risk needs to know, per participant: who currently has or later develops a
condition, and when.

| Field | Format | Notes |
|---|---|---|
| `condition_name` | `type_2_diabetes` | Use a controlled list provided by Bioinformatics — request this list before data entry begins so terms are consistent across the cohort. |
| `condition_status` | `present_at_collection / incident_during_followup / absent` | Whether the participant had the condition at sample collection, developed it after, or has no record of it. |
| `diagnosis_date` | `YYYY-MM-DD` or `YYYY-MM` if day unknown | Required if condition_status is not "absent". |
| `followup_status` | `alive / deceased / lost_to_followup` | Required for any participant with more than one contact point. |
| `last_followup_date` | `YYYY-MM-DD` | Most recent date the participant's status was confirmed. This defines how long they were "observed" for outcome modeling. |
| `cause_of_death` | free text + ICD-10 code if available | Only if followup_status = deceased. |

> **NOTE:** One row per condition per participant is fine if a participant has
> multiple conditions — use `participant_id` to link rows. Talk to Bioinformatics
> before designing this table if your intake system handles multi-condition records
> differently; we will adapt the format to what you can realistically extract rather
> than forcing a rebuild of your existing systems.

> **CRITICAL:** If real diagnosis/outcome data isn't available for a participant,
> leave the disease fields blank rather than guessing or inferring from indirect
> signals (e.g. medication lists). Incorrect outcome labels are worse than missing
> ones — they actively teach the model the wrong thing.

## 5. Sample Handling, Labeling & Chain of Custody

- Label every physical tube/sample with `sample_id` only (not participant name or
  any other identifying information) — see Section 7.
- Log date/time of collection, date/time of receipt in lab, and date/time of each
  processing step (extraction, conversion, array run) against the `sample_id`.
- Store extracted DNA at −20°C if not processed within 48 hours of extraction;
  document any freeze-thaw cycles.
- Any deviation from SOP (delayed processing, temperature excursion, repeat
  extraction, etc.) must be logged as a free-text note against the `sample_id` — do
  not silently work around problems. A flagged anomaly is usable data; an unflagged
  one can silently corrupt a whole batch's results.

## 6. Data & File Submission

### 6.1 What to Submit Per Batch

1. Raw IDAT files (both Grn and Red per sample) — do not submit only
   normalized/processed beta values; Bioinformatics needs the raw files to apply a
   single, consistent normalization pipeline across the whole cohort.
2. The sample metadata spreadsheet (Section 4), one row per sample, matching IDAT
   filenames exactly.
3. The plate/chip layout map for the batch.
4. A short batch summary note: date range, number of samples, any deviations or QC
   flags.

### 6.2 Naming Convention

Use: `ZL-[4-digit sample number]_[Grn|Red].idat` — e.g. `ZL-0001_Grn.idat`,
`ZL-0001_Red.idat`. The `sample_id` in the metadata file (`ZL-0001`) must match this
prefix exactly.

### 6.3 Where to Submit

Upload to the designated secure repository path provided by Bioinformatics (not
email, not general shared drives). Confirm the current path with Bioinformatics
before the first batch — this will be filled in once the storage environment is
finalized.

## 7. Consent, Privacy & De-identification

> **CRITICAL:** Do not submit any directly identifying information (full name, date
> of birth, address, national ID number, contact details) in any file that reaches
> Bioinformatics or the modeling pipeline. All linkage to identity must stay inside
> the lab's own secure participant registry, accessible only via `participant_id`.

- Confirm before sample collection that participant consent explicitly covers: (a)
  methylation/genetic-style analysis, and (b) use of de-identified data to train and
  validate an internal predictive algorithm. Generic "research use" consent may not
  cover this — check with whoever manages your consent forms / IRB-equivalent
  process if uncertain.
- Maintain the identity-to-`participant_id` mapping in a separate, access-restricted
  system from the metadata and methylation data described in this document.
- If a participant withdraws consent, notify Bioinformatics immediately with the
  `participant_id` so their samples can be removed from any training set in
  progress.

## 8. Pre-Submission Checklist

Run through this before uploading each batch:

- [ ] Platform confirmed as EPIC v2 for every sample in the batch
- [ ] All DNA QC values meet thresholds in Section 2.3, and actual values (not just
      pass/fail) are recorded
- [ ] Plate/chip/position randomized across case-control and demographic groups; at
      least one technical control included
- [ ] Metadata spreadsheet complete for all Section 4.2 mandatory fields — no
      placeholder values used for missing data
- [ ] `sample_id` values match IDAT filenames and physical labels exactly
- [ ] No directly identifying information present in any submitted file
- [ ] Consent status confirmed for every participant in the batch
- [ ] Any deviations from SOP logged and flagged in the batch summary note
- [ ] Raw IDAT files (Grn + Red) included — not just processed/normalized values

## 9. Questions & Escalation

If anything in this document is ambiguous, or a real-world situation doesn't fit a
defined category (e.g. an unusual specimen type, missing consent paperwork, an
unclear `condition_name`), **pause and check with Bioinformatics before proceeding**
rather than making an assumption. A held sample can always be processed later; a
mislabeled or wrongly-categorized one quietly degrades the whole model and is very
hard to detect after the fact.
