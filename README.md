# Support Integrity Auditor (SIA)

Support Integrity Auditor (SIA) is an automated triage optimization engine designed to police and validate incoming customer support CRM logs. By using a multi-layered hybrid architecture (Deep Learning + Rule-Based Heuristics + Semantic Clustering), SIA cross-checks human-assigned ticket priorities against the actual underlying semantic context to identify **SLA Breach Risks** (under-classification) and **Efficiency Alerts** (over-classification).
Link for streamlit web app: 
---https://support-integrity-auditor-mars.streamlit.app/

PLEASE NOTE THAT THE notebook.ipynb HAS SOME WIDGETS WHICH I ADDED DURING TRAINING ON COLAB. BUT THEY CANNOT BE VIEWED ON GITHUB, SO IT IS SHOWING INVALID NOTEBOOK. THIS FILE CAN BE VIEWED BY DOWNLOADING IT.

## Repository Structure & Hybrid Architecture

To bypass GitHub's file size limits while maintaining standard open-source portability, this repository utilizes a **hybrid storage model**: Lightweight logic, tokenizers, and structural configurations sit directly in this GitHub repo, while the heavy deep learning weights are fetched dynamically at runtime.

```text
├── .github/                  # CI/CD configuration files
├── sia_distilBERT_mismatch_model/
│   ├── config.json           # Model architecture configuration
│   ├── tokenizer_config.json # Tokenizer hyperparameters
│   └── tokenizer.json        # Vocabulary mapping arrays
│   
├── app.py                    # Streamlit Dashboard UI Pipeline
├── predict.py                # Core Inference & Multi-layer Ensemble Engine
├── requirements.txt          # Third-party operational dependencies
├── notebook.ipynb            # Full reproducible pipeline
                                  (pseudo-labeling → training → inference).
└── README.md                 # System Documentation

```
## Heavy Weight Hosting
Model Weights Asset: model.safetensors

External CDN Link: Hugging Face Hub Repository

Runtime Mechanics: The system downloads or pulls the binary weights directly from Hugging Face memory structures using huggingface_hub on startup, caching them natively to prevent system latency.

## Code Files & Functioning
### 1. notebook.ipynb
This is the whole pipeline in which experimentation has been done. It includes Data preprocessing,  cleaning, Stage 1 signal formation, Stage 2 classifier training and stage 3 evidence dossier creation.
#### 2. train_pipeline.py
It includes the training script of the project. It can be run locally to get to the results mentioned. It will create the folder with saved model, which can be used further for prediction.
### 3. predict.py
The absolute logical core of the system. It initializes the SupportIntegrityAuditor class which constructs an evaluation ensemble.

```Initialization (__init__)```: Reads local configuration parameters and tokenizer files from ./sia_distilBERT_mismatch_model, then securely drops down model.safetensors using the Hugging Face API to load into PyTorch.

```audit_ticket()```: Evaluates an individual record payload, routing data streams across the deep learning layer, static regex arrays, and a pseudo-randomized semantic fallback logic array to stitch together a unified Evidence Dossier.

```audit_batch()```: Accepts complete Pandas dataframes, running rows concurrently to append macro-level operational metrics (SIA_Verdict, SIA_Confidence_Pct).

### 4. app.py
The presentation wrapper built on Streamlit. It optimizes server interactions by wrapping the auditor instantiation inside @st.cache_resource to preserve GPU/CPU cycles across user refreshes.

Tab 1 (Single Ticket Audit): Allows dynamic text area configuration inputs to parse quick sample tickets.

Tab 2 (Bulk Log Pipeline): Handles spreadsheet asset parsing (.csv, .xlsx), lets users map dynamic columns, outputs analytical metrics summaries, and yields downloadable customized .json portfolio dossiers.

### 5. requirements.txt
   It has all the modules and library required to run this project. Install them on your system using:
   ```pip install -r requirements.txt```

### 6. evidence_dossiers.json
   This file includes all the generated evidence dossier in a json file. Dossiers are built on the test split of the enhanced_customer_support_data.csv which is downloaded from https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data

## Methodology and Core Logic

The architectural workflow operates through a definitive three-stage transition pipeline to evaluate human operational compliance without prior golden labeling:
```
[Raw CRM Records] 
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Self-Supervised Pseudo-Label Fusion           │
│ ├─ Rule-Based Heuristic (Keyword Constraints)          │
│ └─ Semantic Proximity Engine (Dense Vector Anchor)     │
└────────────────────────────────────────────────────────┘
       │
       ▼ (Generates Inferred Severity vs Assigned Priority Labels)
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Fine-Tuned Sequence Classifier                │
│ └─ DistilBERT Core (Metadata Tags + Text Concat Input) │
└────────────────────────────────────────────────────────┘
       │
       ▼ (Flags MISMATCH_ALERT / CONSISTENT)
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Hallucination-Free Evidence Dossier           │
│ └─ Strict Traceable Mapping Schema                     │
└────────────────────────────────────────────────────────┘
```

### Stage 1: Self-Supervised Pseudo-Label Generation:
Three signals are extracted from the data:
1. Rule-based NLP Layer: Tracks localized keyword concentrations across explicit dictionary scopes (critical_keywords vs trivial_keywords) to detect structural context mismatches.
2. Resolution time regression: Severity was assigned according to the time taken for the resolution of the issue.
3. Semantics Embeddings clustering: Embeddings generated using the sentence transformer all-MiniLM-L6-v2 , which were given labels according to Anchors defined for each category. Similarity was calculated for each text to these anchors to assign a label.
4. LLM zero shot scoring: An LLM Phi-3-mini-4k-instruct was used to generate the labels from the inductive reasoning of the LLM. Numbers from 1 to 4 were generated and taken as LLM inferred severity.
#### Fusion Strategy:
Weights are given to each signal
Rule-based NLP: 0.05
Resolution Time: 0.05
Semantics Embedding clustering: 0.25
LLM inferred severity: 0.65
The derived Inferred Severity (1 = Low, 4 = Critical) is weighed against the human Assigned Priority to compute severity delta:
  Severity delta = Inferred Severity Value - Assigned Priority Value
If severity delta is greater than or equal to 2, it was assigned a mismatch (1) label, otherwise a consistent (0) label.

### Stage 2: Classifier Training
Instead of using a frozen zero-shot model, a localized DistilBERT Sequence Classifier is trained directly on the pseudo-labeled data. Inputs are structured into complete metadata contextual blocks to ensure the model captures categorical weights alongside the text:
```
  def serialize_row(row):
        metadata = (
            f"[METADATA] Channel: {row['Ticket_Channel']} | "
            f"Category: {row['Issue_Category']} | "
            f"Assigned Priority: {int(row['assigned_priority_num'])} | "
            f"Tier: {int(row['customer_tier'])} | "
            f"Agent: {row['Assigned_Agent']} | "
            f"Sub-Day: {row['submission_day_name']} | "
            f"Customer Satisfaction: {row['Satisfaction_Score']} | "
            f"Resolution Time: {row['Resolution_Time_Hours']:.1f} hours"
        )
        text_data = f"[TEXT] Subject: {row['Ticket_Subject']} | Description: {row['Ticket_Description']}"
        return f"{metadata} {text_data}"

    df['serialized_input'] = df.apply(serialize_row, axis=1)
```

### Stage 3 : Grounded Evidence Dossier Generation
For flagged deviations, an exact tracking schema is produced. Hard Rule: Every value entry in feature_evidence is strictly mapped to the literal input attributes, eliminating any risk of generative hallucinations.

---

## 📊 Verification Metrics & Passing Thresholds

The system is evaluated against the mandatory validation split using the following operational thresholds:

| Metric Target | Required Threshold | Current Engine Achievement | Status |
| :--- | :--- | :--- | :--- |
| **Binary Classification Accuracy** | $\ge 83\%$ | **94.90%** | ✅ PASS |
| **Macro F1 Score** | $\ge 0.82$ | **0.9273** | ✅ PASS |
| **Per-Class Recall (Consistent)** | $\ge 0.78$ | **0.95** | ✅ PASS |
| **Per-Class Recall (Mismatched)** | $\ge 0.78$ | **0.91** | ✅ PASS |

## Local Execution Setup Guide

1. Clone the Codebase Footprint
   ```git clone [https://github.com/VarunSinsinwar/Support-Integrity-Auditor.git](https://github.com/VarunSinsinwar/Support-Integrity-Auditor)```
```cd Support-Integrity-Auditor```

2. Install Pinned Dependencies
   ```pip install -r requirements.txt```

3. Launch the Operational Web Portal
      ```streamlit run app.py```
