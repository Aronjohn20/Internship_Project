# Explanation 1 — Why Are We Doing This? (30% Milestone Goals)

> **What this file is**: A plain-English breakdown of *why* each of the 4 goals in the 30% milestone exists, what problem each one solves, and how they all connect to the bigger picture of FedMed.

---

## 🧠 The Core Problem First

Let's start with the real-world situation that FedMed is trying to solve.

Imagine you are a hospital in Mumbai. You have thousands of chest X-ray scans of patients who were diagnosed with pneumonia. Another hospital in Delhi has thousands more. If you could combine all of that data and train one powerful AI model on it, the model would be incredibly accurate at detecting pneumonia.

**But here's the problem**: You can't just share patient X-rays and clinical records between hospitals. Those records are private and legally protected (think HIPAA in the US, or equivalent laws in India). Hospitals are not allowed to hand over raw patient data to anyone — not even to another hospital trying to save lives.

So the traditional approach — collect all the data in one place and train one model — **violates patient privacy**.

FedMed's answer to this is **Federated Learning**: instead of sharing the *data*, hospitals share the *knowledge* their model has learned. The raw data never leaves the hospital. Only the model's weights (mathematical numbers that encode what the model learned) are shared.

But to prove this approach actually works, you need to build it, test it, and show the numbers. That is exactly what the 30% milestone is doing.

---

## 🎯 Goal 1: Multimodal Data Pipeline

### What is being done?
- Download Chest X-ray images (from NIH ChestX-ray14 and CheXpert), resize them to 224×224 RGB.
- Pair each X-ray with clinical tabular features: **Age** (as a standardized z-score) and **Sex** (as a 0/1 binary).
- Divide all of this data across **3 simulated hospital clients** (Client 1, Client 2, Client 3) with zero patient overlap between them.

### Why is this necessary?

**Why pair X-rays with clinical features?**

An X-ray image alone is already powerful, but a doctor doesn't just look at the image in isolation — they also consider the patient's age, sex, and clinical history. A 70-year-old male presenting with chest symptoms has very different risk factors compared to a 25-year-old female. By combining the image with this clinical context, the AI model gets a more complete picture, just like a real radiologist does.

This is what **multimodal** means — multiple modes or types of input data (images *and* clinical tabular data) working together to produce a better prediction.

**Why z-score normalize age?**

Raw age values (e.g., 47, 3, 82) have very different magnitudes. If you feed these directly into a neural network alongside small pixel values, the model may over-prioritize the large age number. A z-score transforms age into a standardized value (mean = 0, standard deviation = 1) so no single feature dominates just because of its numerical scale.

**Why strict patient-level separation?**

This is one of the most critical steps in medical AI. If the same patient's X-ray appears in both the training set *and* the test set, the model has essentially already "seen" that patient during training. When you test it later, it will appear to perform better than it actually is — this is called **data leakage**, and it leads to falsely optimistic results that won't hold up in the real world. By enforcing strict patient-level separation, every patient belongs to *either* training, validation, or testing — never to more than one.

---

## 🧬 Goal 2: Multimodal Model Architecture

### What is being done?
Building a neural network with three parts:
1. **Vision Transformer (ViT)** — reads and understands the X-ray image.
2. **MLP Clinical Encoder** — reads and understands the patient's age and sex.
3. **Fusion Head** — combines both into a single prediction: Pneumonia or Non-Pneumonia.

### Why this architecture?

**Why a Vision Transformer (ViT) instead of a regular CNN?**

A Convolutional Neural Network (CNN) like ResNet reads an image by scanning local patches one after another. A Vision Transformer works differently: it divides the X-ray into a grid of small patches (e.g., 16×16 pixels), converts each patch into a token (a numeric vector), and then uses **attention mechanisms** to let every patch "look at" every other patch simultaneously.

This matters for chest X-rays because pneumonia can manifest as subtle patterns distributed across the lung fields. A ViT can capture long-range spatial relationships across the entire chest — for example, recognizing that a shadowing pattern in the lower right lobe is more significant when accompanied by a diffuse haziness in the left lobe. A CNN's local receptive fields can miss these global spatial relationships.

This design choice is directly inspired by **Khader et al. (2023)**, the base academic paper the team has adopted.

**Why an MLP for clinical features?**

An MLP (Multi-Layer Perceptron) is simply a stack of fully connected neural network layers. For tabular clinical features (which are just a small list of numbers like age and sex), an MLP is the right tool — it learns non-linear relationships between these demographic features and outputs a compact dense "clinical embedding" that captures what the clinical data implies about the patient's condition.

**Why combine them in a Fusion Head?**

Neither the image alone nor the clinical data alone gives the full picture. By concatenating the visual representation from the ViT with the clinical representation from the MLP, and then passing this through one final classification layer, the model learns to weigh both sources of information together when making its final binary decision: Pneumonia (1) or Non-Pneumonia (0).

---

## 🔗 Goal 3: Federated Learning (FedAvg) Simulation

### What is being done?
- Each of the 3 simulated hospital clients trains the multimodal model on their own local data partition.
- After local training, each client sends its updated model weights to a central server.
- The central server runs **Federated Averaging (FedAvg)**: it takes a weighted average of all 3 clients' model weights and produces a single updated **global model**.
- The global model is sent back to all clients to begin the next round.
- This repeats for multiple **communication rounds**.

### Why is this the heart of FedMed?

**Why keep data local?**

This is the entire privacy argument. In a federated setup, raw patient X-rays and clinical records **never leave the hospital's servers**. Only model weights — which are just millions of floating point numbers that describe mathematical relationships, not actual patient records — are transmitted. A model's weights cannot be easily reverse-engineered to reconstruct a specific patient's X-ray. This makes federated learning far safer than sharing raw data.

**Why FedAvg specifically?**

Federated Averaging is the foundational, simplest, and most well-understood algorithm in federated learning, introduced in a seminal 2016 paper by McMahan et al. It works like this: each client's contribution to the new global model is weighted by how many training samples that client has. A hospital with 30,000 samples contributes more to the global model than one with 10,000 samples. This is mathematically sound and practically fair.

For the 30% milestone, FedAvg establishes the baseline. More sophisticated algorithms like FedProx (designed to handle data heterogeneity better) are planned for later milestones.

**What are communication rounds?**

A communication round is one full cycle of:
1. Clients receive the global model from the server.
2. Clients train locally for a few epochs on their private data.
3. Clients send their updated model weights back to the server.
4. Server averages all weights → new improved global model.

You run many rounds (e.g., 10–20) because after each round the global model improves. Plotting how the model improves round by round gives you the **federated convergence curve**.

---

## 📊 Goal 4: Clinical Evaluation & Benchmark Comparison

### What is being done?
- Train a **second model** — the **Centralized Baseline** — by pooling all 3 clients' training data together and training on it directly (ignoring privacy, just for comparison).
- Evaluate **both models** on the same held-out test set using medical-grade metrics.
- Compare the two side-by-side to quantify the privacy-utility trade-off.

### Why is this the most important evaluation step?

**Why train a Centralized Baseline at all?**

Because federated learning has an inherent disadvantage: each client trains on only a fraction of the data, and communication between rounds is imperfect. So naturally, a centralized model trained on all the data at once will likely perform better. The centralized baseline is the **performance ceiling** — the best possible result you could achieve if privacy didn't matter.

The key scientific question FedMed is asking is:

> *"How much performance do we sacrifice in order to gain full privacy?"*

If the centralized model achieves **86% Sensitivity** and the federated model achieves **83% Sensitivity**, that 3% gap is the cost of privacy. For a medical application, the team and clinicians can then judge whether that trade-off is acceptable.

**Why use Sensitivity, Specificity, ROC-AUC, and F1 instead of just Accuracy?**

Plain accuracy is misleading for highly imbalanced datasets. In FedMed's dataset, only ~5.15% of records are Pneumonia cases. This means a model that blindly predicts "Not Pneumonia" for every single patient would achieve **94.85% accuracy** without learning anything useful — and it would miss every single real pneumonia case.

Here is what each clinical metric actually measures:

| Metric | Formula | What It Catches | Why It Matters in Medical AI |
|---|---|---|---|
| **Sensitivity (Recall)** | TP / (TP + FN) | Of all actual Pneumonia patients, what % did the model correctly identify? | Missing a pneumonia case (false negative) can cost a patient their life. This must be high. |
| **Specificity** | TN / (TN + FP) | Of all actual Normal patients, what % did the model correctly label as Normal? | Falsely alarming a healthy patient (false positive) causes unnecessary invasive procedures and anxiety. |
| **ROC-AUC** | Area under the Sensitivity vs. (1–Specificity) curve | Overall ability of the model to distinguish Pneumonia from Normal at all decision thresholds. | The gold standard summary metric in medical AI evaluation. A score of 1.0 is perfect; 0.5 is no better than random guessing. |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | The balance between catching Pneumonia cases and avoiding false alarms. | Useful as a single combined score, especially under class imbalance. |

**Why compare convergence curves?**

A convergence curve shows how quickly and stably a model improves over time:
- For the **centralized model**: performance vs. training **epochs** (passes through the dataset).
- For the **federated model**: performance vs. **communication rounds**.

Comparing these two curves tells you:
- *Does the federated model eventually approach the centralized model's performance?*
- *Does it take more rounds than epochs? Is the trade-off in training time acceptable?*
- *Is the federated training stable, or does it oscillate wildly from round to round?*

These are the practical engineering and clinical questions that determine whether FedMed's federated approach is viable in the real world.

---

## 🔵 How All 4 Goals Connect — The Full Picture

```
GOAL 1: Data Pipeline
   ➜  Produces 3 clean, privacy-separated hospital data partitions
   ➜  (118,654 records, 36,324 unique patients, strictly zero leakage)
            │
            ▼
GOAL 2: Multimodal Model Architecture (ViT + MLP + Fusion Head)
   ➜  The SAME architecture is used in BOTH paths below
            │
       ┌────┴────┐
       │         │
       ▼         ▼
  GOAL 3A:    GOAL 3B:
  Federated   Centralized
  Learning    Baseline
  (FedAvg,    (All data
   Privacy    pooled —
   Preserved) no privacy)
       │         │
       └────┬────┘
            │
            ▼
GOAL 4: Clinical Evaluation on Held-Out Test Set
   ➜  Sensitivity, Specificity, ROC-AUC, F1 for BOTH models
   ➜  Convergence curves compared side-by-side
   ➜  "How much performance do we trade away for full privacy?"
            │
            ▼
RESULT: FedMed proves hospitals can collaborate to detect pneumonia
        without ever sharing a single patient record.
```

---

## 📝 Summary in One Paragraph

FedMed simulates a real-world scenario where 3 hospitals train a shared AI model on pneumonia detection without ever sharing raw patient data. We build a multimodal model (Vision Transformer for X-rays + MLP for clinical data), train it in a federated way using the Flower (`flwr`) framework and FedAvg algorithm, and then compare it against a centralized model that cheats by pooling all data together. By evaluating both on the same held-out test set using clinical metrics like Sensitivity and ROC-AUC, we can precisely answer the core research question: *"How much diagnostic performance do we trade away in exchange for complete patient privacy?"* The 30% milestone is the proof-of-concept that this entire pipeline — data, model, federation, and evaluation — actually works end-to-end before we add advanced features like Differential Privacy and Grad-CAM explainability in later milestones.

---
*File created: 2026-09-22 | FedMed project — main_project/explanations/explanation_1.md*
