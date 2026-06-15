# Support-Integrity-Auditor

Support Integrity Auditor (SIA) is an automated triage optimization engine designed to police and validate incoming customer support CRM logs. By using a multi-layered hybrid architecture (Deep Learning + Rule-Based Heuristics + Semantic Clustering), SIA cross-checks human-assigned ticket priorities against the actual underlying semantic context to identify **SLA Breach Risks** (under-classification) and **Efficiency Alerts** (over-classification).

---

## 🏗️ Repository Structure & Hybrid Architecture

To bypass GitHub's file size limits while maintaining standard open-source portability, this repository utilizes a **hybrid storage model**: Lightweight logic, tokenizers, and structural configurations sit directly in this GitHub repo, while the heavy deep learning weights are fetched dynamically at runtime.

```text
├── .github/                  # CI/CD configuration files
├── sia_distilBERT_mismatch_model/
│   ├── config.json           # Model architecture configuration
│   ├── tokenizer_config.json # Tokenizer hyperparameters
│   ├── tokenizer.json        # Vocabulary mapping arrays
│   └── special_tokens_map.json
├── app.py                    # Streamlit Dashboard UI Pipeline
├── predict.py                # Core Inference & Multi-layer Ensemble Engine
├── requirements.txt          # Third-party operational dependencies
└── README.md                 # System Documentation
