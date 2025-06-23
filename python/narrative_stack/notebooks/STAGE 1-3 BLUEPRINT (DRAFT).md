# Project Blueprint: A Three-Stage Financial Representation Learning System

**Version:** 1.0
**Date:** June 17, 2025

---

### **1. Overview**

The primary objective of this project is to create a highly compressed, structured, and queryable representation of complex financial data sourced from SEC filings. This is achieved through a three-stage pipeline that progressively refines raw financial facts into a dense latent vector, which is ultimately suitable for advanced time-series analysis.

The system is designed to be modular, with each stage performing a distinct and specialized task.

### **2. Architectural Diagram**

> Note: All three stages are autoencoders.

```text
[Raw SEC Filings]
|
▼
+----------------------+
|       STAGE 1        |
| Atomic Feature       |   ---> Produces 6 categorized stacks of 256-dim vectors
| Extraction           |        (e.g., credit/duration, debit/instant, etc.)
+----------------------+
|
▼
+----------------------+
|       STAGE 2        |
| Structured Latent    |   ---> Produces a single, 512-dim vector for each time-slice
| Representation       |
+----------------------+
|
▼
+----------------------+
|       STAGE 3        |
| Temporal Dynamics    |   ---> Analyzes sequences of vectors for forecasting, etc.
| Modeling             |
+----------------------+
|
▼
[Final Task Output]
```


---

### **3. Stage 1: Atomic Feature Extraction**

This stage is responsible for converting raw, individual financial facts into rich, high-dimensional vector representations, which we term "atomic embeddings."

* **Objective:** To create a dense vector for every single financial fact that captures its semantic meaning and numerical value.
* **Input:** Triplets of `(US GAAP Concept, Unit of Measure, Value)` extracted from SEC digital records.
* **Architecture & Process:**
    1.  The `US GAAP Concept` (e.g., "Revenues") and `Unit of Measure` (e.g., "USD") are combined into a descriptive sentence.
    2.  This sentence is fed into a pre-trained **BGE-en-large** model to produce a **1024-dimensional** semantic vector.
    3.  Principal Component Analysis (PCA) is applied to reduce this vector to **243 dimensions**, while retaining 95% of the original variance. This creates a dense, semantic "direction" for the financial concept.
    4.  This 243-dim vector is then "meshed" with its corresponding numerical **Value**. The final **256-dimensional** atomic embedding is constructed to represent both the semantic direction and the value's magnitude. *(Note: This likely involves concatenating the 243-dim vector with the normalized value and potentially other metadata to reach the 256-dim total).*
* **Output:** A large dataset of 256-dimensional atomic embeddings. For Stage 2, these are pre-sorted into **6 categorized stacks** based on their financial attributes (`balance_type` and `period_type`).
* **Model Size:** Approximately **7.7 million** parameters.

---

### **4. Stage 2: Structured Latent Representation**

This stage takes the categorized stacks of atomic embeddings for a given time period and compresses them into a single, holistic, and queryable latent vector.

* **Objective:** To create a single vector that summarizes the entire financial state for a time-slice and can be "interrogated" to reconstruct specific parts of that state.
* **Input:** The 6 categorized stacks of 256-dim atomic embeddings from Stage 1.
* **Architecture & Process:**
    * **Encoders:** **Six parallel `PerceiverStackEncoder` modules** process each of the 6 input stacks independently. Each encoder uses a latent bottleneck to efficiently handle variable numbers of embeddings in each stack.
    * **Compression:** The 6 output vectors from the encoders are concatenated and projected by a linear layer into a single **`shared_latent_vector`** (e.g., 512-dimensional).
    * **Shared Decoder:** A **single `PerceiverDecoder` module** is used for all reconstruction tasks. This promotes parameter efficiency and knowledge transfer between the categories.
    * **Compositional Queries:** The decoder is steered by **learnable queries**. These queries are not arbitrary but are constructed by combining embeddings representing the attributes of the desired output (e.g., `embedding('credit') + embedding('duration')`). This allows the model to learn the independent meaning of each attribute.
    * **Reconstruction:** During training, the decoder is called 6 separate times—once for each query—to reconstruct all 6 original input stacks from the single shared vector. The total loss is the sum of the reconstruction errors from all 6 stacks.
* **Output:** A single, fixed-size latent vector (e.g., 512-dim) representing the financial state for one time-slice.
* **Model Size:** Approximately **7.9 million** parameters.

---

### **5. Stage 3: Temporal Dynamics Modeling**

This future stage will focus on learning patterns from sequences of the latent vectors produced by Stage 2.

* **Objective:** To understand and model how the compressed financial state evolves over time.
* **Input:** A sequence of latent vectors from Stage 2 (e.g., one vector for each quarter over several years).
* **Architecture & Process:** The specific architecture is to be determined but will likely involve a time-series model like a **Transformer** or **LSTM**. It will analyze time windows of the Stage 2 outputs to perform downstream tasks.
* **Output:** Task-specific predictions, such as financial forecasting, anomaly detection, or trend analysis.

---

### **6. Training & Deployment Strategy**

The pipeline is designed for a highly efficient, two-phase training workflow to maximize modularity and reduce computational load.

* **Phase 1: Pre-computation:** The entire dataset of raw financial facts is processed by the **Stage 1 model once**. The resulting atomic embeddings are saved to disk, already sorted into their respective categories.
* **Phase 2: Autoencoder Training:** The **Stage 2 model** is trained exclusively on the pre-computed embeddings loaded from disk. This decouples the two stages, allowing for rapid experimentation on the autoencoder without the computational overhead of re-running Stage 1 for every training batch.
