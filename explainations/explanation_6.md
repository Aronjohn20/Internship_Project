# Explanation 6 — Stage 3: The Great Filter (Binary Labels & The Art of Saying No)

> **Script explained**: `stage3_binary_labels.py`
> **Story**: Not all 335,534 records deserve to make it into the model's training data. Some are ambiguous, some are irrelevant, and one dataset has four different shades of "maybe". Stage 3 is the bouncer at the door.

---

## 🎬 The Story

You have your unified notebook from Stage 2 — 335,534 records, all formatted consistently. Now comes the hard question:

**"For each of these 335,534 records, what should the model learn?"**

The answer must be simple and unambiguous: **Pneumonia (1)** or **Not Pneumonia (0)**. No maybes. No "we think so." No "the radiologist wasn't sure." Machine learning models, especially for medical applications, need clean, definitive ground truth.

Stage 3 is about making that binary decision for each record — and having the courage to say **"no, this one doesn't make the cut"** for anything ambiguous.

By the end, 216,864 records are thrown out. Only 118,670 make it through.

---

## 🏛️ Two Different Courtrooms: NIH and CheXpert Have Different Label Systems

Stage 3 processes NIH records and CheXpert records completely differently because they use completely different label conventions. Think of it as two separate courtrooms with different laws.

---

### ⚖️ Courtroom A: NIH ChestX-ray14

**How NIH labels work:**
NIH uses a multi-label system. A single chest X-ray can have multiple diseases labelled simultaneously. Each disease is assigned a number. The relevant ones are:
- **Label 0** = "No Finding" (nothing abnormal detected)
- **Label 7** = Pneumonia

A record's `raw_label_info` might look like `"2,7,11"` — meaning that patient has Disease 2, Pneumonia (7), and Disease 11.

**The NIH Decision Rules:**

| Condition | Decision | Binary Label |
|---|---|---|
| Labels contain `7` (and not `0`) | ✅ Pneumonia | `1` |
| Labels do NOT contain `7` | ✅ Non-Pneumonia | `0` |
| Labels contain BOTH `0` AND `7` | ❌ Excluded — Inconsistent! | Removed |

**Why exclude the `[0, 7]` combination?**

If a radiologist simultaneously says "No Finding" AND "Pneumonia" for the same scan, something is wrong with the label. It's self-contradictory. Either the report was generated erroneously, or there was a data entry mistake. Training a model on contradictory labels would teach it conflicting information. So these records are safely discarded. In practice: **0 such records were found** — no inconsistencies in the NIH dataset.

---

### ⚖️ Courtroom B: CheXpert

CheXpert has two separate policies applied in sequence.

**Policy 1: View Type Filter**

```python
if view_type != "Frontal":
    → Exclude (reason: "CheXpert_Lateral_View")
```

A lateral (side-view) X-ray looks completely different from a frontal X-ray — different anatomy visible, different structures, different perspective. NIH is 100% frontal. If we mixed CheXpert's lateral views into the same training set as NIH's frontal views, the model would get confused: the same pattern that means "bottom-right lung lobe" in a frontal view means something completely different in a lateral view.

By keeping only frontal CheXpert records, we ensure visual consistency with NIH. This excluded **32,387 lateral views** immediately.

**Policy 2: Pneumonia Label Filter (Frontal views only)**

For the remaining CheXpert frontal records, the pneumonia column uses a 4-code system:

| Code | Meaning | Decision | Binary Label |
|---|---|---|---|
| `3` | Pneumonia Present | ✅ Retained | `1` |
| `2` | Pneumonia Absent | ✅ Retained | `0` |
| `1` | Uncertain ("possible pneumonia") | ❌ Excluded | Removed |
| `0` | Unlabeled (radiologist didn't evaluate) | ❌ Excluded | Removed |

**Why exclude uncertain (Code 1) cases?**

If a radiologist wrote "possible pneumonia" or "cannot rule out pneumonia", that's not a clear diagnosis. If you train a model to predict "Pneumonia" on cases a radiologist was uncertain about, the model learns fuzzy, inconsistent patterns. When it then encounters a clear-cut pneumonia case in the real world, it may under-predict confidence. For the 30% milestone, clean and definitive labels are more valuable than volume.

**Why exclude unlabeled (Code 0) cases?**

If a radiologist simply didn't evaluate the pneumonia finding (perhaps the report focused on a different disease), you don't have ground truth for that scan. You don't know if it's Pneumonia or Not Pneumonia — you simply don't know. Using "we don't know" as a training label would corrupt the model.

This excluded **15,981 uncertain** and **168,496 unlabeled** CheXpert records.

---

## 📋 The Complete Accounting

Stage 3 is meticulous. Every excluded record is saved to a separate log file (`stage3_excluded_records.parquet`) with the exact reason it was excluded. This is medical-grade accountability — you can always trace back exactly why any record was removed.

```
Total Input:        335,534 records
───────────────────────────────────
RETAINED:           118,670 records (35.37%)
  NIH Pneumonia:      1,431
  NIH Non-Pneumonia: 110,689
  CheXpert Pneumonia: 4,675
  CheXpert Non-Pneu:  1,875

EXCLUDED:           216,864 records (64.63%)
  CheXpert Lateral:  32,387
  CheXpert Uncertain: 15,981
  CheXpert Unlabeled: 168,496
  NIH Inconsistent:   0
───────────────────────────────────
CHECK: 118,670 + 216,864 = 335,534 ✅
```

The final assertion verifies this exactly:

```python
assert total_retained + total_excluded == total_raw
assert set(df_retained["binary_label"].unique()).issubset({0, 1})
```

If any record somehow got assigned a label that isn't 0 or 1, the script hard-stops immediately.

---

## 😮 The Uncomfortable Truth: 64.63% Thrown Away

You might wonder: isn't throwing away 64.63% of your data wasteful?

No — for two reasons:

1. **Quality over quantity**: A model trained on 118,670 clean, definitively labeled records will vastly outperform one trained on 335,534 records with ambiguous, contradictory, or irrelevant labels.

2. **Most exclusions are principled**: 168,496 of the excluded records were simply "unlabeled" by CheXpert's radiologists — they don't contain any pneumonia ground truth at all. Keeping them would mean feeding the model records with no answer to learn from.

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| Exclude CheXpert lateral views | Visual consistency with NIH (which is 100% frontal). Mixing view types confuses the model. |
| Exclude uncertain CheXpert cases (code 1) | Ambiguous labels produce a confused model. Clean labels produce a reliable model. |
| Exclude unlabeled CheXpert cases (code 0) | No ground truth = no learning signal. Including them would just add noise. |
| Save all excluded records with reasons | Medical-grade auditability — every removal must be traceable and justifiable. |
| Hard assertion: retained + excluded = total | Guarantees no record was silently lost or double-counted during filtering. |

---

## 📝 One-Sentence Summary

> Stage 3 is the strict gatekeeper that converts raw multi-label disease codes into definitive Pneumonia (1) / Non-Pneumonia (0) binary labels, throwing out 216,864 ambiguous, inconsistent, or view-incompatible records while keeping a complete audit trail of every exclusion, leaving 118,670 clean, definitively labeled records for the model to learn from.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_6.md*
