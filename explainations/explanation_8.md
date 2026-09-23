# Explanation 8 — Stage 6 & 7: Dividing the Kingdom Fairly (Patient-Level Split & Client Partitioning)

> **Scripts explained**: `stage6_patient_level_split.py` and `stage7_client_partitions.py`
> **Story**: You now have 118,654 clean records. The next challenge is to divide them — first into training, validation, and test zones (Stage 6), then into 3 separate hospital territories (Stage 7). And the golden rule: no patient can ever appear in two zones at the same time.

---

## 🎬 The Story

Imagine you are a teacher with 36,324 students (patients) and you need to:
1. Divide them into a study group (training), a practice-test group (validation), and a final-exam group (test) — so students only appear in one group.
2. Then divide the study group into 3 separate classrooms (hospital clients) — so each classroom has different students.

The challenge: some students have multiple folders (multiple X-ray scans). You can't put Student A's Monday folder in the study group and their Tuesday folder in the final exam group — that would mean the exam sees a student the model already studied, making the exam results meaningless.

This is the **patient-level split** problem. Stage 6 solves it.

---

## 🏫 Stage 6: Patient-Level Train / Validation / Test Split

### Why "Patient-Level" Instead of Record-Level?

The naive approach would be to randomly shuffle all 118,654 records and assign 70% to training, 15% to validation, 15% to testing. Simple, right?

**Wrong.** This approach has a fatal flaw.

One patient might have 10 different X-ray scans (taken over 10 hospital visits over many years). If you split by record, 7 of those scans might go to training and 3 to testing. The model would "remember" patterns specific to that patient during training, and when it sees the patient's other scans in testing, it would look artificially good — not because it learned to generalize, but because it memorized patient-specific patterns.

This is **data leakage**, and it produces falsely optimistic evaluation results that don't hold up in the real world where the model encounters patients it's never seen before.

**The solution:** Group by patient first. Assign the *entire patient* — all their scans — to exactly one split.

---

### How the Code Splits Patients

**Step 1: Determine each patient's class**

```python
patient_labels = df.groupby("patient_id")["binary_label"].max().reset_index()
```

For each patient, this takes the maximum label across all their scans. If a patient has even one Pneumonia scan (label=1), they are classified as a "positive patient." This is used for **stratified splitting**.

**Step 2: Stratified shuffle — keeping proportions balanced**

```python
np.random.seed(42)
pos_p_list = list(... patients with pneumonia ...)
neg_p_list = list(... patients without pneumonia ...)
np.random.shuffle(pos_p_list)
np.random.shuffle(neg_p_list)
```

The code separates patients into two lists: those who have had at least one pneumonia scan (positive patients) and those who never had pneumonia (negative patients). It then shuffles each list independently using a fixed random seed (`42` — chosen for reproducibility so the split is identical every time the script runs).

**Step 3: Assign 70/15/15 splits within each group**

```python
pos_train_end = int(0.70 * n_pos)
pos_val_end = pos_train_end + int(0.15 * n_pos)

pos_train = pos_p_list[:pos_train_end]
pos_val = pos_p_list[pos_train_end:pos_val_end]
pos_test = pos_p_list[pos_val_end:]
```

70% of positive patients go to training, 15% to validation, 15% to testing. The same split is done independently for negative patients. Then:

```python
train_patient_set = set(pos_train) | set(neg_train)  # Union of positive and negative training patients
```

This ensures the training, validation, and testing sets each have roughly the same ratio of positive to negative patients — preventing a scenario where, by bad luck, all the pneumonia patients ended up in the test set.

**Step 4: The Mathematical Proof of Zero Leakage**

This is the most important part of the entire script:

```python
train_val_overlap = train_patient_set & val_patient_set   # Set intersection
train_test_overlap = train_patient_set & test_patient_set
val_test_overlap = val_patient_set & test_patient_set

assert len(train_val_overlap) == 0
assert len(train_test_overlap) == 0
assert len(val_test_overlap) == 0
assert len(train_patient_set) + len(val_patient_set) + len(test_patient_set) == total_patients
```

The `&` operator between Python sets gives the intersection — patients who appear in BOTH sets. All three assertions verify these intersections are empty (zero patients shared). The fourth assertion verifies no patient was lost or doubled.

If any assertion fails, the script crashes immediately with a clear error message. It's an automatic, mathematical guarantee of zero data leakage.

**Step 5: Assign all rows**

```python
train_df = df[df["patient_id"].isin(train_patient_set)].copy()
```

Once patient sets are verified leak-free, all records (scans) belonging to training patients go to training, etc.

### Final Split Results

| Partition | Samples | Unique Patients | Pneumonia % | Imbalance Ratio |
|---|---|---|---|---|
| **Train** | 83,696 | 25,426 | 5.10% | **18.60 : 1** |
| **Validation** | 17,470 | 5,448 | 5.33% | 17.74 : 1 |
| **Test** | 17,488 | 5,450 | 5.16% | 18.39 : 1 |
| **Total** | **118,654** | **36,324** | 5.15% | — |

Notice that all three partitions have nearly identical Pneumonia percentages (~5.1–5.3%). This is stratified splitting working correctly — balanced class ratios across all partitions.

---

## 🏥 Stage 7: Dividing Training Data into 3 Hospital Clients

With the training partition established (83,696 records, 25,426 patients), Stage 7 divides it across 3 simulated hospital clients. The same golden rule applies: **no patient can appear in more than one client.**

### Why Patient-Level Again?

Same reason as Stage 6. If Patient X's scans appeared in both Client 1 and Client 2, then during federated training:
- Client 1 trains on some of Patient X's scans.
- Client 2 trains on some of Patient X's other scans.
- When the global model is evaluated, Patient X's patterns are already known from two clients.

This would make the federated model appear to generalize better than it actually does.

### How the Code Divides Patients Across 3 Clients

```python
pos_splits = np.array_split(pos_patients, 3)  # Split pneumonia patients into 3 equal groups
neg_splits = np.array_split(neg_patients, 3)  # Split non-pneumonia patients into 3 equal groups

for i in range(3):
    c_patients = set(pos_splits[i]) | set(neg_splits[i])  # Combine for client i
```

`np.array_split(list, 3)` divides the list into 3 roughly equal chunks. The same is done for positive and negative patient groups separately — ensuring each client gets roughly the same proportion of pneumonia patients (stratified partitioning).

### The Three-Way Overlap Proof

```python
c1_c2 = client_patient_sets[0] & client_patient_sets[1]
c1_c3 = client_patient_sets[0] & client_patient_sets[2]
c2_c3 = client_patient_sets[1] & client_patient_sets[2]

assert len(c1_c2) == 0
assert len(c1_c3) == 0
assert len(c2_c3) == 0
assert sum(len(s) for s in client_patient_sets) == total_train_patients
```

Three pairwise intersection checks — every pair of clients must share zero patients. And the final assertion verifies all 25,426 training patients are accounted for across the 3 clients.

Each client partition is saved as both CSV and Parquet, with a `client_id` column added (`"Hospital_1"`, `"Hospital_2"`, `"Hospital_3"`).

### Final Client Partition Results

| Client | Samples | Unique Patients | Pneumonia % |
|---|---|---|---|
| **Client 1 (Hospital_1)** | 27,593 | 8,476 | 5.14% |
| **Client 2 (Hospital_2)** | 27,878 | 8,475 | 5.12% |
| **Client 3 (Hospital_3)** | 28,225 | 8,475 | 5.05% |
| **Total** | **83,696** | **25,426** | 5.10% |

Notice the remarkably even distribution — all three clients have almost identical pneumonia rates (~5%). This is IID (Independent and Identically Distributed) partitioning at work. The random shuffle combined with stratified splitting produced naturally balanced clients.

---

## 🗺️ The Complete Data Journey (Stages 1–7 Summary)

```
Stage 1: Harvest metadata from NIH (112,120) + CheXpert (223,414)
                              = 335,534 raw records

Stage 2: Harmonize into unified 11-column canonical table
                              = 335,534 canonical records

Stage 3: Filter ambiguous labels, lateral views, exclusions
                              = 118,670 clean records (64.63% discarded)

Stage 4: 8 safety checks + remove 16 age anomalies
                              = 118,654 final clean records

Stage 6: Patient-level 70/15/15 split (verified zero leakage)
  Train:       83,696 records (25,426 patients)
  Validation:  17,470 records  (5,448 patients)
  Test:        17,488 records  (5,450 patients)

Stage 7: Divide training into 3 hospital clients (verified zero overlap)
  Client 1:    27,593 records  (8,476 patients)
  Client 2:    27,878 records  (8,475 patients)
  Client 3:    28,225 records  (8,475 patients)
```

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| Patient-level splitting (not record-level) | Prevents data leakage — ensures the test set contains only patients the model has never seen during training. |
| Stratified split (positive and negative patients split separately) | Guarantees each partition has a proportional mix of Pneumonia vs Non-Pneumonia patients — avoiding lopsided splits. |
| Fixed random seed (`np.random.seed(42)`) | Makes the split perfectly reproducible — every team member running the script gets identical train/val/test splits. |
| Mathematical `assert` zero-overlap checks | Provides a machine-verifiable, crash-on-failure guarantee that no patient appears in two partitions. |

---

## 📝 One-Sentence Summary

> Stage 6 takes 36,324 unique patients and mathematically assigns each one — along with all their scans — to exactly one of Train/Validation/Test using stratified shuffling and zero-leakage assertions; Stage 7 then takes the 25,426 training patients and divides them equally across 3 simulated hospital clients using the exact same zero-overlap guarantee, producing the privacy-separated data partitions that the federated learning simulation will train on.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_8.md*
