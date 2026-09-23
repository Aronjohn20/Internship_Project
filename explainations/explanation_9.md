# Explanation 9 — multimodal_dataset.py: The Bridge Between Data Files and the Neural Network

> **File explained**: `data/multimodal_dataset.py`
> **Story**: The Parquet files and CSVs sitting in `data_prep/` are just rows in a spreadsheet. The model can't use spreadsheet rows — it needs PyTorch tensors. This file is the bridge that converts patient records into model-ready batches, on demand, one image at a time.

---

## 🎬 The Story

After all 7 stages of data preparation, you now have clean, verified Parquet files:
- `client_1.parquet`, `client_2.parquet`, `client_3.parquet`
- `val.parquet`, `test.parquet`

But a neural network doesn't know what a Parquet file is. It doesn't know what age `47.5` or sex `"M"` means. It only speaks one language: **PyTorch tensors** — multi-dimensional arrays of floating-point numbers.

`multimodal_dataset.py` is the translator. It reads the Parquet files, does all the necessary preprocessing (resizing images, normalizing values, encoding features), and packages everything into the standardized batch dictionary that the ViT and MLP models consume.

This file is the **final product of all 7 stages** — the interface point where data engineering hands off to model engineering.

---

## 🏗️ The Three Things This File Provides

```python
# 1. The Dataset class — knows how to load one sample
class MultimodalDataset(Dataset):  ...

# 2. Three DataLoader factory functions — create ready-to-use data streams
def get_client_loader(client_id, ...)  # For Hospital 1, 2, or 3
def get_val_loader(...)                # For validation
def get_test_loader(...)               # For test evaluation
```

---

## 🔬 Part 1: The MultimodalDataset Class

### `__init__`: Setting Up the Fast Lookup Tables

```python
def __init__(self, df: pd.DataFrame, transform=None, age_mean=47.70, age_std=16.80):
```

When the Dataset is created, it doesn't load any images yet (images are huge — loading all 83,000+ of them at once would consume hundreds of GB of RAM). Instead, it pre-processes everything it *can* do cheaply upfront:

**Pre-computing image paths:**
```python
self.image_paths = self.df["image_path"].values
```
Stores all image file paths as a NumPy array for fast indexing. Getting the path for sample #5432 is now O(1) — instant.

**Pre-computing clinical features (the big preprocessing step):**
```python
# Age: z-score normalization
ages = self.df["age"].values.astype(np.float32)
norm_ages = (ages - 47.70) / 16.80   # Standardize to mean=0, std=1

# Sex: binary encoding
sexes = np.where(self.df["sex"].values == "F", 1.0, 0.0).astype(np.float32)

# Stack into [N, 2] array — one row per sample, two columns (age_zscore, sex_binary)
self.clinical_features = np.stack([norm_ages, sexes], axis=1)
```

This processes ALL N samples' clinical features once at initialization. During training, when the DataLoader asks for sample #5432, its clinical features `[age_zscore, sex_binary]` are already computed — retrieved instantly from the array.

**Pre-computing labels:**
```python
self.labels = self.df["binary_label"].values.astype(np.float32).reshape(-1, 1)
```
Labels as float32 (required by `BCEWithLogitsLoss`) shaped as `[N, 1]` — a column vector where each row is one sample's label.

---

### `__getitem__`: Loading One Sample on Demand (Lazy Loading)

```python
def __getitem__(self, idx):
```

This method is called by PyTorch's DataLoader every time it needs sample number `idx`. It only loads the image for that one specific sample — not all 83,000.

**Image loading:**
```python
if os.path.exists(img_path):
    with Image.open(img_path) as pil_img:
        image = pil_img.convert("RGB")      # Convert grayscale to 3-channel RGB
        image_tensor = self.transform(image) # Resize to 224×224 + normalize
else:
    image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)  # Fallback
```

Two key things happen here:

1. **`.convert("RGB")`**: X-ray images are inherently grayscale (single channel). But ViT pre-trained on ImageNet expects 3-channel RGB input. `.convert("RGB")` duplicates the grayscale channel across all 3 channels, making the image look like `[R=grayscale, G=grayscale, B=grayscale]`. The ViT can now accept it without architectural changes.

2. **Fallback zero tensor**: If an image file doesn't exist on disk (which can happen during development when some NIH images haven't been exported yet), instead of crashing the entire training run, the code returns a black image (all zeros). This is a robustness measure — the pipeline keeps running and you can investigate the missing files separately.

**The transform pipeline:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),                              # Scale to 224×224
    transforms.ToTensor(),                                      # PIL → float32 tensor [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],           # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])
```

Three operations happen in sequence:
1. **Resize**: The original X-rays come in various sizes (e.g., 1024×1024, 2048×2048). All are resized to exactly 224×224 pixels — the size the ViT expects.
2. **ToTensor**: Converts PIL image (pixel values 0–255 as integers) to a PyTorch tensor (values 0.0–1.0 as float32).
3. **Normalize**: Subtracts the ImageNet mean and divides by the ImageNet std, channel by channel. After this, pixel values can be negative (which is fine for neural network mathematics). Without this step, the pre-trained ViT's internal mathematics — which were calibrated on ImageNet-normalized inputs — would be misaligned.

**Assembling the sample dictionary:**
```python
return {
    "image": image_tensor,      # [3, 224, 224] tensor
    "clinical": clinical_tensor, # [2] tensor
    "label": label_tensor,       # [1] tensor
    "patient_id": ...,           # string
    "sample_id": ...,            # string
    "dataset_source": ...        # string
}
```

This is the standard batch dictionary defined in the Data Contract (see explanation_3.md). Every sample from every partition always has this exact structure.

---

## 🔧 Part 2: The DataLoader Factory Functions

Three simple, clean functions that abstract away the file path handling:

```python
def get_client_loader(client_id: int, batch_size=32, shuffle=True, num_workers=0, data_dir=...):
    file_path = os.path.join(data_dir, f"client_{client_id}.parquet")
    df = pd.read_parquet(file_path)
    dataset = MultimodalDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
```

**What `DataLoader` does:**
The PyTorch `DataLoader` wraps the `MultimodalDataset` and handles:
- **Batching**: Calls `__getitem__` for `batch_size` samples and stacks them into batch tensors. `[3, 224, 224]` per sample becomes `[B, 3, 224, 224]` for the batch.
- **Shuffling**: Randomizes the order of samples each epoch so the model doesn't memorize the order.
- **Parallel loading**: With `num_workers > 0`, multiple CPU processes can load images in parallel while the GPU trains.

**Why shuffle=True for training, shuffle=False for val/test?**
Training with shuffled data prevents the model from learning patterns based on the order of samples (e.g., "the 500th sample of every epoch is always Pneumonia"). For validation and test, you want deterministic, reproducible evaluation — same order every time.

**How teammates use this:**
```python
# Member 3 (FL Lead) — loading hospital 2's data for federated training
from data.multimodal_dataset import get_client_loader
loader = get_client_loader(client_id=2, batch_size=32, shuffle=True)
for batch in loader:
    images = batch["image"]      # [32, 3, 224, 224]
    clinical = batch["clinical"] # [32, 2]
    labels = batch["label"]      # [32, 1]
    # → Feed to model
```

No knowledge of file paths, Parquet reading, or preprocessing required. The DataLoader handles everything.

---

## 📊 What Happens When the DataLoader Iterates

```
Epoch begins → DataLoader shuffles sample indices

Iteration 1:
  DataLoader picks indices [412, 7823, 45, 99821, ...]  (batch_size samples)
  For each index:
    → Reads image path from self.image_paths[idx]
    → Opens image from disk (lazy!)
    → Converts to RGB, resizes to 224×224, normalizes
    → Gets clinical features from pre-computed array (instant)
    → Gets label from pre-computed array (instant)
  → Stacks all samples into batch tensors
  → Yields batch dict to training loop

Iteration 2: picks next batch of indices ...
...
Epoch ends when all samples have been seen once
```

The key insight: **images are loaded lazily (on demand), but clinical features and labels are pre-computed (instant)**. This is the optimal memory/speed tradeoff for a dataset with large images.

---

## 🔑 Key Design Decisions and Why

| Decision | Why |
|---|---|
| Lazy image loading | Loading 118,654 X-ray images at once would require ~500+ GB of RAM. Loading one batch at a time uses only megabytes. |
| Pre-computing clinical features at init | Clinical features are tiny (just 2 floats per sample). Pre-computing them once is faster than re-computing during training. |
| `.convert("RGB")` on grayscale X-rays | ViT pre-trained on ImageNet requires 3-channel input. RGB conversion makes it compatible without changing the model architecture. |
| ImageNet normalization | Required for compatibility with pre-trained ViT weights — the model "expects" inputs in this normalized range. |
| Fallback zero tensor for missing images | Prevents a missing image file from crashing an entire training run. A warning can be logged and training continues. |
| Three separate factory functions | Clean, named interfaces — `get_client_loader(2)` is self-documenting. Teammates don't need to know about Parquet files. |

---

## 📝 One-Sentence Summary

> `multimodal_dataset.py` is the final handoff between data engineering and model engineering — it wraps the clean Parquet files in a PyTorch `Dataset` that lazily loads X-ray images from disk (resizing and ImageNet-normalizing them on the fly), pre-computes clinical feature tensors and binary labels at initialization for speed, and exposes three simple `get_*_loader()` factory functions that any teammate can call to get a ready-to-train DataLoader without touching a single file path or preprocessing step.

---
*File created: 2026-09-23 | FedMed project — main_project/explanations/explanation_9.md*
