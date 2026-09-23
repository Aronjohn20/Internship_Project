# Explanation 5 — Stage 2: Teaching Two Different Languages to Speak the Same Tongue (Canonical Schema)

> **Script explained**: `stage2_canonical_schema.py`
> **Story**: NIH and CheXpert were built by different research teams with different conventions. Before they can work together, someone has to act as a translator and create a universal language. That's Stage 2.

---

## 🎬 The Story

After Stage 1, you have two separate notebooks — one for NIH, one for CheXpert. Both notebooks have roughly the same information (patient ID, age, sex, label, image path), but they're written in completely different formats.

Imagine hiring two translators, one who only speaks French and one who only speaks German, and asking them both to describe the same patient. They'd use different words, different structures, and different conventions. If you just stapled their notes together and handed them to the model, the model would be confused.

Stage 2 is about creating a **universal language** — a single, standardized table structure that both datasets conform to. In database terms, this is called a **canonical schema**.

---

## 🔤 The Problem: Two Datasets, Two Dialects

Here is a side-by-side comparison of how NIH and CheXpert differ at Stage 1's output:

| Field | NIH Stage 1 Format | CheXpert Stage 1 Format |
|---|---|---|
| Patient ID | `"00012345"` (just a number) | `"patient00001"` (name-prefixed) |
| Scan identifier | `scan_id` = integer (e.g., 12) | extracted from file path string |
| Sex encoding | `"M"` or `"F"` (already string) | `0` or `1` (integer codes!) |
| Projection | Unknown (not in NIH shards) | `0` = AP, `1` = PA, `2` = Unknown |
| Labels | List of integers, e.g., `[2, 7, 11]` | Single integer pneumonia code `3` |
| Label storage | `[2, 7, 11]` as a Python list | `"pneumonia_code_3"` as a string |

If you tried to use these two tables together without harmonization, your code would crash or produce wrong results almost immediately.

---

## 🏗️ What Stage 2 Builds: The 11-Column Canonical Table

Stage 2 creates a standardized table with exactly these 11 columns for every record from both datasets:

```
sample_id        → Unique ID for each scan (globally unique across all 335,534 records)
dataset_source   → "NIH" or "CheXpert"
patient_id       → Globally unique patient ID (prefixed to prevent collision)
study_id         → Unique study/scan identifier
image_path       → Full path to the local image file on disk
age              → float32 (same type for both datasets)
sex              → "M" or "F" (always string, always uppercase)
view_type        → "Frontal" or "Lateral" (always string)
projection       → "AP", "PA", or "Unknown" (always string)
raw_label_info   → The raw label as a string (to be interpreted in Stage 3)
binary_label     → -1 (placeholder — to be filled in Stage 3)
```

Let's walk through how each dataset is transformed to fit this schema.

---

## 🔄 NIH Transformation

**Patient ID — Adding a prefix to prevent collision:**
```python
df_nih_canon["patient_id"] = "nih_p_" + df_nih_raw["patient_id"].astype(str)
```
NIH patient IDs are plain numbers like `"00012345"`. CheXpert patient IDs are strings like `"patient00001"`. What if NIH has a patient `"00001"` and CheXpert has a patient named `"patient00001"`? They're different people but their IDs could theoretically clash. By prefixing all NIH IDs with `"nih_p_"`, you guarantee they can never collide with any CheXpert ID.

**Sample ID — Creating a globally unique scan identifier:**
```python
df_nih_canon["sample_id"] = "nih_" + patient_id + "_" + scan_id
```
Combines patient ID and scan ID into one unique string identifier for every scan.

**Labels — Preserving raw data as a string:**
```python
df_nih_canon["raw_label_info"] = df_nih_raw["labels"].apply(
    lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x)
)
```
The NIH labels are a Python list like `[2, 7, 11]`. This line converts that list to a comma-separated string `"2,7,11"` for consistent storage. Stage 3 will parse this string to decide if the record is Pneumonia or not.

**View type:** Set to `"Frontal"` for all NIH records — NIH ChestX-ray14 is 100% frontal X-rays.

---

## 🔄 CheXpert Transformation

**Sex — Decoding integer codes to human-readable strings:**
```python
sex_map = {0: "M", 1: "F"}
df_chex_canon["sex"] = df_chex_raw["sex"].map(sex_map).fillna("Unknown")
```
CheXpert stores sex as integers (`0` = Male, `1` = Female). The map converts these to the same `"M"`/`"F"` strings that NIH uses — one universal format.

**Projection — Decoding integer codes to strings:**
```python
proj_map = {"0": "AP", "1": "PA", "2": "Unknown"}
df_chex_canon["projection"] = df_chex_raw["projection"].astype(str).map(proj_map).fillna("Unknown")
```
CheXpert projection codes (`0`, `1`, `2`) are mapped to meaningful strings (`"AP"`, `"PA"`, `"Unknown"`).

**Labels — Preserving the pneumonia code:**
```python
df_chex_canon["raw_label_info"] = "pneumonia_code_" + df_chex_raw["raw_pneumonia"].astype(str)
```
CheXpert's pneumonia label is stored with a prefix so Stage 3 can distinguish it from NIH's label format: `"pneumonia_code_3"` means "CheXpert says Pneumonia is Present."

---

## 🔗 Merging Into One Table

```python
canonical_df = pd.concat([df_nih_canon, df_chex_canon], ignore_index=True)
```

This single line stacks both harmonized tables on top of each other, producing one unified table with **335,534 rows** and exactly **11 columns**.

---

## 🔍 The Critical Uniqueness Check

After merging, the code runs this assertion:

```python
duplicate_sample_ids = canonical_df["sample_id"].duplicated().sum()
assert duplicate_sample_ids == 0
```

This is non-negotiable. If any `sample_id` appears twice, it means two different records think they're the same scan. That would cause the model to receive conflicting information about the same scan and could corrupt training. The assertion hard-stops the script if this ever happens. In practice, it found **0 duplicates** — all 335,534 `sample_id` values are globally unique.

---

## 📊 What Stage 2 Produced

| Metric | Value |
|---|---|
| Total canonical records | 335,534 |
| Unique global patients | 95,345 |
| Null values in any column | 0 |
| Duplicate sample IDs | 0 |
| Runtime | 1.68 seconds |

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| Prefix NIH patient IDs with `"nih_p_"` | Prevents ID collision if NIH and CheXpert happen to share the same numeric patient ID. |
| Convert all labels to raw string format | Avoids mixing label systems — Stage 3 explicitly interprets each source's labels separately. |
| Hardcode `binary_label = -1` | Makes it obvious that labels haven't been assigned yet. If Stage 3 is ever skipped by accident, `-1` labels will immediately cause training to fail (rather than silently training on wrong labels). |
| Assert 0 duplicate sample IDs | Hard-fails the pipeline if the merge produced any duplicate records, preventing silent data corruption. |

---

## 📝 One-Sentence Summary

> Stage 2 acts as a universal translator between NIH and CheXpert's incompatible formats — standardizing patient IDs, sex codes, projection codes, and label representations into one clean 11-column canonical table of 335,534 rows where every record has the exact same structure, zero nulls, and zero duplicate IDs.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_5.md*
