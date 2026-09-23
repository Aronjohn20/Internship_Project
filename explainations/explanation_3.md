# Explanation 3 — The Data Contract: What Is a Batch and Why Does It Look Like This?

> **What this file is**: A plain-English breakdown of the "Data Contract" table from `data/README.md` — what every key in the batch dictionary means, why it is shaped the way it is, and why these design choices matter for the model.

---

## 🧠 Start Here: What Is a "Batch" and Why Does It Exist?

When you train a neural network, you do not feed it the entire dataset at once. That would require enormous amounts of RAM and would make training extremely slow. Instead, you feed it small chunks of data called **batches**.

A batch is simply a group of samples processed together in one go. If your batch size is 16, it means you're feeding 16 chest X-rays (plus their associated clinical data and labels) to the model at the same time.

In PyTorch, a DataLoader automatically collects individual samples from your dataset and groups them into these batches. After grouping, it hands each batch to your training loop as a **Python dictionary** — a container with named keys, where each key holds a different piece of information about those 16 samples.

In FedMed, every single batch — whether it comes from Hospital Client 1, Client 2, Client 3, Validation, or Test — always has exactly the same 6 keys. This is the **Data Contract**.

---

## 📦 The Batch Dictionary — Key by Key

### 🖼️ Key 1: `batch["image"]`

```
Type:   torch.Tensor
Shape:  [B, 3, 224, 224]
Dtype:  torch.float32
```

**What it is**: The chest X-ray image(s) in this batch, ready to be fed into the Vision Transformer.

**Breaking down the shape `[B, 3, 224, 224]`:**

| Dimension | Size | Meaning |
|---|---|---|
| **B** | Batch size (e.g., 16) | There are B images in this batch. |
| **3** | 3 color channels | Red, Green, Blue — even though X-rays are grayscale, the ViT pre-trained on ImageNet expects 3-channel RGB input. The grayscale image is replicated across all 3 channels. |
| **224** | Height in pixels | Every X-ray has been resized to 224 pixels tall. |
| **224** | Width in pixels | Every X-ray has been resized to 224 pixels wide. |

**Why 224×224?**
The Vision Transformer (ViT) was originally designed and pre-trained on 224×224 images (the ImageNet standard). Using this size means we can leverage pre-trained ViT weights that already "know" how to extract visual features — instead of learning everything from scratch. This is called **transfer learning** and it dramatically speeds up training and improves accuracy on limited medical datasets.

**Why `torch.float32`?**
Raw pixel values from an image file are integers from 0 to 255 (e.g., a bright white pixel is 255, a black pixel is 0). PyTorch neural networks need floating point numbers (decimals) to do the gradient math during training. So the values are converted to float32.

**Why ImageNet normalization?**
Before being stored in the batch, pixel values are normalized using ImageNet statistics:
- Mean: `[0.485, 0.456, 0.406]` (one per RGB channel)
- Std: `[0.229, 0.224, 0.225]` (one per RGB channel)

This means: for each pixel, `normalized_value = (pixel_value / 255 - mean) / std`.

The reason: the ViT's pre-trained weights were learned on ImageNet data that was normalized this way. If you feed it un-normalized pixel values, the model's internal math becomes misaligned and it will perform much worse — like speaking to someone in a language they don't understand. By normalizing the same way, the ViT can immediately apply its pre-learned visual knowledge to chest X-rays.

---

### 🏥 Key 2: `batch["clinical"]`

```
Type:   torch.Tensor
Shape:  [B, 2]
Dtype:  torch.float32
```

**What it is**: The clinical tabular features for each patient in the batch — age and sex — encoded as numbers ready to go into the MLP Clinical Encoder.

**Breaking down the shape `[B, 2]`:**

| Dimension | Size | Meaning |
|---|---|---|
| **B** | Batch size (e.g., 16) | One row per patient in the batch. |
| **2** | 2 features | Feature 0 = Age (z-score), Feature 1 = Sex (binary). |

**Index 0 — Patient Age (z-score normalized):**

The raw age values (e.g., 47, 3, 82) are converted to z-scores using the dataset-wide statistics:
- Mean age: **μ = 47.70 years**
- Standard deviation: **σ = 16.80 years**

Formula: `age_zscore = (raw_age - 47.70) / 16.80`

Examples:
- A 47-year-old (the average patient) → z-score = 0.0
- A 64-year-old (1 standard deviation older than average) → z-score ≈ +1.0
- A 31-year-old (1 standard deviation younger than average) → z-score ≈ -1.0

**Why z-score?** Without normalization, age ranges from 0 to 105 while the sex feature ranges from 0 to 1. When these are fed into the MLP together, the model's gradient updates are dominated by the much larger age numbers, effectively ignoring sex information. Z-scoring puts all features on the same scale so the model gives each feature a fair chance to be learned from.

**Index 1 — Patient Sex (binary encoded):**

Sex is stored as:
- `0.0` = Male
- `1.0` = Female

**Why binary?** Sex is a categorical variable (it has no meaningful ordering or mathematical distance between categories). Converting it to 0/1 binary is the standard encoding for two-class categorical variables in neural networks. The MLP can then learn a different weight for each sex without implying that one sex is "twice as much" as the other.

---

### 🏷️ Key 3: `batch["label"]`

```
Type:   torch.Tensor
Shape:  [B, 1]
Dtype:  torch.float32
```

**What it is**: The ground truth answer — what this X-ray is actually diagnosed as. This is what the model is trained to predict.

**Values:**
- `1.0` = **Pneumonia** (positive class)
- `0.0` = **Non-Pneumonia / Normal** (negative class)

**Why shape `[B, 1]` and not just `[B]`?**
The model's final output is a single logit (one number per sample) with shape `[B, 1]`. The loss function `nn.BCEWithLogitsLoss` expects the predictions and the labels to have the same shape. By keeping labels as `[B, 1]` (a column vector), the shapes match perfectly and no extra reshaping is needed during training.

**Why `float32` instead of `int`?**
`nn.BCEWithLogitsLoss` performs floating point mathematics. If you feed it integer labels, PyTorch will raise a dtype mismatch error. Pre-converting labels to float32 prevents this.

**Why only `1.0` and `0.0`?**
This is a **binary classification** problem — every X-ray is either Pneumonia or it isn't. There is no middle ground or uncertainty at the label level. Uncertain cases (like CheXpert's "Uncertain" label, code 1) were explicitly excluded during Stage 3 of data preparation for exactly this reason — the model should only train on clean, definitive diagnoses.

---

### 👤 Key 4: `batch["patient_id"]`

```
Type:   list of str
Length: B
```

**What it is**: The unique patient identifier for each sample in the batch.

Examples:
- `"nih_p_00012345"` (NIH patient)
- `"chexpert_patient12345"` (CheXpert patient)

**Why store this in the batch at all?**
The model never uses `patient_id` during training (it's not fed into the ViT or MLP). So why include it?

1. **Zero patient leakage verification**: During the data pipeline (Stages 6 and 7), we proved mathematically that no patient appears in more than one split. By carrying `patient_id` through to batch level, downstream code can re-verify this at any time — just check if any patient_id in the test batches appears in the training batches.

2. **Clinical auditability**: In a real medical AI system, every prediction must be traceable back to a specific patient and scan. Carrying `patient_id` in the batch means you can log "for patient nih_p_00012345, the model predicted Pneumonia with 87% confidence" — enabling clinical review of edge cases.

3. **Debugging**: If you spot a strange batch with wildly off predictions, `patient_id` lets you trace back exactly which patient caused the issue.

---

### 🔬 Key 5: `batch["sample_id"]`

```
Type:   list of str
Length: B
```

**What it is**: The unique identifier for the specific scan/image (not the patient — a patient may have multiple X-rays taken at different times).

**Why is this different from `patient_id`?**
One patient can have multiple chest X-rays (e.g., taken on different dates for follow-up). In the NIH dataset especially, many patients have dozens of scans. The `patient_id` identifies the person; the `sample_id` identifies the specific scan.

- `patient_id` = "who is this person?"
- `sample_id` = "which specific X-ray image is this?"

This matters for duplicate detection (the pipeline verifies zero duplicate `sample_id` values), reproducible experiment logging, and error tracing.

---

### 🏦 Key 6: `batch["dataset_source"]`

```
Type:   list of str
Length: B
```

**What it is**: The name of the original dataset each sample came from.

**Values:**
- `"nih"` = NIH ChestX-ray14
- `"chexpert"` = CheXpert

**Why include this?**
1. **Per-source performance analysis**: NIH and CheXpert have different X-ray quality, scanner types, and patient populations. During evaluation, you might want to ask: "Does the model perform equally well on NIH images and CheXpert images, or is there a performance gap between sources?" You can only answer this question if you know which source each test sample came from.

2. **Future extensibility**: When MIMIC-CXR is added to supplement the dataset, it will appear as `"mimic_cxr"` in this field. The model code never changes — the source is just tracked in the batch for analysis purposes.

3. **Debugging data pipeline issues**: If the model is performing strangely on a subset of samples, `dataset_source` lets you quickly check if the issue is localized to one source (e.g., maybe CheXpert images were improperly normalized at some stage).

---

## 🔗 How the Batch Flows Through the Model

Here is how each batch key is used once it enters the training loop:

```
batch = {
    "image":          [B, 3, 224, 224]  ─────────────────────→  ViT Image Encoder
                                                                      │
    "clinical":       [B, 2]            ─────────────────────→  MLP Clinical Encoder
                                                                      │
                                                             Fusion Head (concatenate
                                                             visual + clinical embeddings)
                                                                      │
    "label":          [B, 1]            ─→  BCEWithLogitsLoss ← Binary logit output [B, 1]
                                                                      │
    "patient_id":     list of str       ─────────────────────→  Logging & Auditability only
    "sample_id":      list of str       ─────────────────────→  Logging & Auditability only
    "dataset_source": list of str       ─────────────────────→  Per-source metrics analysis
}
```

The model only receives `image` and `clinical` as inputs. The loss function uses `label`. The remaining three keys (`patient_id`, `sample_id`, `dataset_source`) are for human analysis and system transparency — they ride along in the batch but are never fed into any neural network computation.

---

## ✅ Summary Table

| Key | Shape | Used By | Why This Design |
|---|---|---|---|
| `image` | `[B, 3, 224, 224]` | ViT encoder | 224×224 RGB for ViT compatibility; ImageNet normalized for pre-trained weight alignment |
| `clinical` | `[B, 2]` | MLP encoder | Age z-scored for scale uniformity; Sex binary encoded for categorical representation |
| `label` | `[B, 1]` | Loss function | Float32 for BCEWithLogitsLoss; shape `[B,1]` matches model output shape |
| `patient_id` | `list[str]` | Auditing/Debugging | Patient-level traceability; enables zero-leakage re-verification |
| `sample_id` | `list[str]` | Auditing/Debugging | Scan-level unique identifier; enables per-image error tracing |
| `dataset_source` | `list[str]` | Evaluation analysis | Per-source performance breakdown; future MIMIC-CXR extensibility |

---

## 📝 One-Sentence Version

> The batch is a dictionary of 6 keys: the X-ray image tensor (for the ViT), the clinical features tensor (for the MLP), the ground truth label (for the loss function), and three string identifiers (patient, scan, and source) that ride along for transparency, auditing, and per-source evaluation but are never fed into the neural network itself.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_3.md*
