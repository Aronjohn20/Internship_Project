# Explanation 7 — Stage 4 & 5: The Safety Inspection and the Report Card

> **Scripts explained**: `stage4_data_validation_and_cleaning.py` and `stage5_dataset_statistics.py`
> **Story**: Before a car leaves the factory, inspectors run a battery of checks. Before 118,654 medical records go into an AI, we do the same — 8 clinical safety checks, one final cleaning step, and a complete statistical profile of what we have.

---

## 🎬 The Story

Stage 3 gave us 118,670 records with definitive binary labels. But having labels isn't enough — the data could still be silently wrong in dangerous ways:

- What if some image files are empty paths pointing to nothing?
- What if a patient appears twice under different IDs?
- What if someone's age is recorded as 414 (an obvious data entry error)?
- What if some records have an image but no clinical features, creating a broken multimodal pair?

Any of these problems would silently corrupt model training in ways that are very hard to debug later. Stage 4 is the systematic safety inspection that catches all of this *before* the data touches any model.

Stage 5 then reads the clean data and writes a complete statistical report — the "report card" — so the team knows exactly what they're working with.

---

## 🔍 Stage 4: The 8 Clinical Safety Checks

The script runs exactly 8 checks, each with its own section and PASS/FAIL status written to a JSON report.

---

### ✅ Check 1: Image Validity & Traceability

```python
empty_paths = df["image_path"].isna().sum() + (df["image_path"].str.strip() == "").sum()
valid_extensions = df["image_path"].apply(lambda p: p.lower().endswith((".png", ".jpg", ".jpeg"))).sum()
chex_files_exist = chex_df["image_path"].apply(os.path.isfile).sum()
```

Three things are verified:
1. **No empty image paths** — every record must point somewhere.
2. **All paths end in `.png`, `.jpg`, or `.jpeg`** — no corrupted or mislabeled file extensions.
3. **CheXpert images actually exist on disk** — physically checks if `os.path.isfile()` returns True for each CheXpert X-ray path.

Result: **0 empty paths, 0 invalid extensions, 0 missing CheXpert files.**

*Note: NIH images aren't verified on disk here because there are 112,000+ of them and disk verification at this scale is slow. CheXpert's 6,550 images are small enough to check quickly.*

---

### ✅ Check 2: Patient & Study Identifiers

```python
null_patient_ids = df["patient_id"].isna().sum() + (df["patient_id"].str.strip() == "").sum()
null_study_ids = df["study_id"].isna().sum() + (df["study_id"].str.strip() == "").sum()
```

Every record must have a valid, non-empty patient ID and study ID. Without these, you can't perform patient-level grouping in Stages 6 and 7, and data leakage would be impossible to prevent.

Result: **0 null patient IDs, 0 null study IDs.**

---

### ✅ Check 3: Image-Clinical-Label Pairing Alignment

```python
aligned_rows = (
    (~df["image_path"].isna()) &
    (~df["age"].isna()) &
    (~df["sex"].isna()) &
    (~df["binary_label"].isna())
).sum()
```

This checks that every record has ALL four essential components: an image path, an age, a sex, and a label. A record missing any one of these is broken — you can't feed it to a multimodal model that needs all four.

Result: **100% of records are complete pairs — 0 discrepancies.**

---

### ✅ Check 4: Label Correctness

```python
label_set = set(df["binary_label"].unique())
invalid_labels = [x for x in label_set if x not in (0, 1)]
```

By this point, the only valid labels are `0` (Non-Pneumonia) and `1` (Pneumonia). If any other value appears — like `-1` (Stage 2's placeholder that Stage 3 forgot to fill), or `2`, or `NaN` — the model would receive nonsense targets during training.

Result: **Labels are strictly `{0, 1}` — no invalid values.**

---

### ✅ Check 5: Duplicate Records & Images

```python
dup_sample_ids = df["sample_id"].duplicated().sum()
dup_image_paths = df["image_path"].duplicated().sum()
```

Two separate checks:
1. **No duplicate `sample_id`**: If the same scan ID appears twice, the model might see the exact same training example multiple times, artificially inflating its confidence on that sample.
2. **No duplicate `image_path`**: If two different records point to the same image file, they're effectively duplicates (same image, possibly with conflicting labels).

Result: **0 duplicate sample IDs, 0 duplicate image paths.**

---

### 🔧 Check 6: Demographic Cleaning (The Only Check That Actually Removes Data)

This is the only check where the script doesn't just report — it actively cleans.

```python
over_105_age = (df["age"] > 105).sum()
df_cleaned = df[~(df["age"] > 105)].copy()
```

**Why remove ages > 105?**
The oldest verified living human in history was 122 years old. In medical datasets, any age above 105 is virtually certainly a data entry error — likely someone accidentally typed "145" instead of "45" or "414" instead of "41". For a neural network, these outlier values would distort the age z-score normalization and confuse the model's understanding of demographic patterns.

The script found **16 such records** (including one NIH patient listed as age 414). All 16 were removed, leaving **118,654 clean records.**

---

### ✅ Check 7: Class Balance Assessment

```python
imbalance_ratio = non_pneu_clean / pneu_clean
```

Reports the class imbalance in the clean dataset. This isn't a PASS/FAIL check — it's an important measurement:

- **Class 1 (Pneumonia)**: 6,105 records (5.15%)
- **Class 0 (Non-Pneumonia)**: 112,549 records (94.85%)
- **Imbalance Ratio**: **18.44 : 1**

This means for every 1 Pneumonia case, there are 18 Non-Pneumonia cases. This is severe imbalance — if the model just always predicts "Not Pneumonia", it would be 94.85% accurate but clinically useless. The imbalance ratio (18.44) becomes the `pos_weight = 18.60` used in the model's loss function to compensate.

---

### ✅ Check 8: Patient-Level Grouping Audit

```python
unique_patients_clean = df_cleaned["patient_id"].nunique()
patient_record_counts = df_cleaned["patient_id"].value_counts()
```

Reports how many unique patients are in the clean dataset and how many scans each patient has on average. This confirms the dataset is structured correctly for patient-level grouping in Stage 6 (where entire patients, not individual scans, are assigned to train/val/test).

Result: **36,324 unique patients** (average ~3.27 scans per patient).

---

### 📋 The Validation Report

All 8 check results are saved to `data_prep/stage4_cleaning_report.json` — a machine-readable audit trail that proves every check was performed and passed.

---

## 📊 Stage 5: The Statistical Report Card

After Stage 4, Stage 5 reads the clean data and computes comprehensive statistics, saving them as both a JSON (for programmatic use) and a Markdown report (for human reading).

### Key Statistics Computed

**Why does the imbalance ratio matter so much?**
Stage 5 computes this explicitly and this number directly informs a critical hyperparameter:

```
pos_weight = 18.60 (non-pneumonia count / pneumonia count in training set)
```

This value is later used in PyTorch's `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([18.60]))`. Without this, the loss function would treat a missed pneumonia case as equally bad as a missed non-pneumonia case. With it, the model is penalized 18.6× harder for missing a real pneumonia case — reflecting the clinical reality that missing a pneumonia diagnosis is far more dangerous than a false alarm.

**Per-source breakdown reveals an interesting asymmetry:**

| Source | Pneumonia Rate |
|---|---|
| NIH | 1.28% (1,430 / 112,104) |
| CheXpert | **71.37%** (4,675 / 6,550) |
| Combined | 5.15% |

CheXpert's retained records are heavily Pneumonia-positive because most of CheXpert's non-pneumonia records were in the "Unlabeled" category (code 0) and got excluded in Stage 3. The few CheXpert records that survived are predominantly confirmed Pneumonia cases.

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| 8 separate named checks with individual PASS/FAIL | Medical-grade accountability — you can prove to reviewers exactly what was validated and when. |
| Only remove age > 105 (not other anomalies) | Only age had verifiable medical impossibilities. All other checks were informational. |
| Save both JSON and Markdown statistics | JSON for programmatic consumption (other scripts can read it); Markdown for human readability in the project repository. |
| Record the imbalance ratio explicitly | The ratio of 18.44:1 directly drives the `pos_weight` hyperparameter in the loss function. |

---

## 📝 One-Sentence Summary

> Stage 4 runs 8 clinical safety checks on 118,670 records, removing 16 records with impossible ages to produce 118,654 clean, verified records with a complete audit trail; Stage 5 then produces a comprehensive statistical profile of this clean data that quantifies the 18.44:1 class imbalance — the critical number that will drive the model's loss function weighting.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_7.md*
