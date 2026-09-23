# Explanation 4 — Stage 1: The Great Data Harvest (NIH & CheXpert Metadata Extraction)

> **Scripts explained**: `stage1_prepare_nih.py` and `stage1_prepare_chexpert.py`
> **Story**: Before we can train any model, we need to gather our raw ingredients — patient records, X-ray paths, demographics. This is where FedMed's data story begins.

---

## 🎬 The Story

Imagine you are a data scientist hired by a hospital to build a pneumonia detection AI. You walk into the hospital's records room. There are two giant filing cabinets — one labeled **NIH ChestX-ray14** and another labeled **CheXpert**. Each cabinet has thousands of folders, and each folder contains an X-ray image, a patient's age, their sex, and a doctor's note about what disease was found.

Your job in Stage 1 is simple: **don't open the X-ray envelopes yet** (they're huge and heavy), but do read every label on every folder — the patient ID, the age, the sex, the diagnosis codes — and write them all down in a notebook.

That notebook is the output of Stage 1.

---

## 📂 Part A: NIH ChestX-ray14 (`stage1_prepare_nih.py`)

### What the NIH dataset looks like on disk

The NIH dataset lives as **6 large Parquet files** (called "shards") on the local hard drive:
```
Datasets/my_nih_chest_xr_dataset/data/
  train-00000-of-00006.parquet
  train-00001-of-00006.parquet
  ...
  train-00005-of-00006.parquet
```

Each Parquet file is like a compressed spreadsheet. Each row is one chest X-ray, and the columns include — critically — an `image` column that stores the actual raw binary image bytes. One single shard can easily be several gigabytes just because of the images.

### The Smart Move: Column Projection

Here is the key decision in Stage 1 NIH:

```python
columns_to_read = ["patient_id", "scan_id", "age", "sex", "labels"]
table = pq.read_table(shard_path, columns=columns_to_read)
```

Instead of reading the entire Parquet file (which would load massive image bytes into RAM), the code uses **PyArrow column projection** — it's like telling the filing cabinet: *"Give me only the label on the outside of the folder. Don't open it and hand me the X-ray inside."*

This is why the script can process 112,120 records across 6 shards in seconds without running out of memory.

### What the Code Extracts Per Record

For each NIH record, the code pulls:
- `patient_id`: a string like `"00012345"` identifying the patient.
- `scan_id`: an integer identifying the specific scan.
- `age`: the patient's age (raw float — some have impossible values like 414!).
- `sex`: `"M"` or `"F"` after `.str.upper()` standardization.
- `labels`: a list of disease codes (e.g., `[7]` means Pneumonia, `[0]` means No Finding, `[2, 11]` means two other diseases).

### Building the Image Path

The actual X-ray PNG files have already been exported to a local folder. The code constructs the expected path for each scan:

```python
df_nih["image_filename"] = patient_id + "_" + scan_id + ".png"
df_nih["image_path"] = r"R:\FedMed_Data\nih_png\" + image_filename
```

This path isn't verified here — it's just recorded. The image validation happens later in Stage 4.

### Diagnosing the Data Before Saving

Before saving anything, the script runs a quick self-audit:
- How many patients are unique?
- What is the sex distribution?
- Are there any age anomalies? (Answer: 16 records with age > 105, including one with age 414 — clearly a data entry error.)
- How many records have the Pneumonia label (label 7)?
- Are there any inconsistent records with both "No Finding" (0) AND "Pneumonia" (7) at the same time? (Answer: 0 — good.)

### Output

Saves `data_prep/nih_prepared_meta.parquet` — an intermediate "notebook" with 112,120 rows and metadata for every NIH scan.

---

## 📂 Part B: CheXpert (`stage1_prepare_chexpert.py`)

### What Makes CheXpert Different

CheXpert lives on the Hugging Face Hub (online), not on a local Parquet shard. The code connects to it using **streaming mode**:

```python
dataset = load_dataset("danjacobellis/chexpert", split="train", streaming=True).remove_columns(["image"])
```

The `.remove_columns(["image"])` is the same smart trick as before — strip out the heavy image column before any data is downloaded. Instead of downloading all 223,414 records' images, the code streams only the metadata for each record one at a time, printing a progress report every 20,000 records.

### CheXpert's Unique Challenges

CheXpert has two complications NIH doesn't:

**1. Lateral vs Frontal Views:**
NIH is 100% frontal X-rays. CheXpert has both frontal and lateral (side-view) X-rays. The code detects view type from the filename:

```python
if "frontal" in raw_path.lower():
    view_type = "Frontal"
elif "lateral" in raw_path.lower():
    view_type = "Lateral"
```

Lateral views will be excluded in Stage 3 — only frontal X-rays are clinically comparable to NIH's images and valid for our model.

**2. Uncertain and Unlabeled Pneumonia:**
Unlike NIH where label 7 simply means "Pneumonia present", CheXpert uses a 4-code system for the Pneumonia column:
- Code **0** = Unlabeled (radiologist didn't evaluate pneumonia)
- Code **1** = Uncertain (radiologist said "possible pneumonia")
- Code **2** = Absent (no pneumonia)
- Code **3** = Present (confirmed pneumonia)

Stage 1 just records this raw code as-is. The decision of what to do with each code (keep or discard) is made deliberately in Stage 3, not here.

### Extracting Patient and Study IDs

CheXpert paths look like:
```
CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
```

The code parses this path to extract:
- `patient_id = "patient00001"`
- `study_id = "study1"`

This is important because patient-level grouping (preventing a patient's X-ray from appearing in both training and testing) depends on having correct patient IDs.

### Output

Saves `data_prep/chexpert_prepared_meta.parquet` — an intermediate notebook with 223,414 rows.

---

## 📊 What Stage 1 Produced

| Dataset | Records | Unique Patients | Pneumonia Cases | Age Anomalies |
|---|---|---|---|---|
| **NIH** | 112,120 | 30,805 | 1,431 | 16 (age > 105) |
| **CheXpert** | 223,414 | 64,540 | 6,039 (code 3) | 0 |
| **Total** | **335,534** | **95,345** | — | **16** |

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| Column projection (don't read image bytes) | Loading 335,534 X-ray images into RAM would require hundreds of GB and crash the system. |
| Record raw labels without translating them yet | Each dataset has different label systems. Translating them here would mix concerns. Clean separation: Stage 1 = collect, Stage 3 = decide. |
| Record image paths without verifying existence | Verifying 335,000 files on disk would take very long and isn't needed yet. That's Stage 4's job. |
| Streaming CheXpert instead of downloading | Avoids downloading the entire ~200GB dataset when only metadata is needed. |
| Self-audit statistics before saving | Catches data quality issues early so you don't discover surprises 3 stages later. |

---

## 📝 One-Sentence Summary

> Stage 1 reads the label on every folder in both filing cabinets (NIH and CheXpert) without opening the heavy image envelopes inside, building two intermediate metadata notebooks that record every patient's ID, age, sex, view type, and raw diagnosis codes — ready for harmonization in Stage 2.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_4.md*
