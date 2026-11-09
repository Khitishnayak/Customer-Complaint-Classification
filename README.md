# AI-Powered Customer Complaint Classification & Triaging System

An end-to-end NLP system that automatically reads incoming customer complaints, classifies them into relevant department/product categories, calculates prediction confidence, and detects escalation urgency (legal threats, fraud, eviction, foreclosure).

---

## 🚀 Key Features

1. **Automatic Multi-Class Classification**:
   - Classifies customer complaints into 9 core business categories with **83.1% test accuracy** and **83.0% macro F1**.
2. **Modular Dataset Expansion**:
   - Pre-integrated with the **CFPB Consumer Complaint Database** (over 1.28M records).
   - Easily ingest and merge any additional CSV, JSON, or Parquet datasets with custom column mapping.
3. **Automated Urgency / Escalation Detection**:
   - Scans text for high-risk legal/fraud keywords (`lawyer`, `lawsuit`, `fraud`, `identity theft`, `foreclosure`) to tag complaints as `CRITICAL`, `MEDIUM`, or `NORMAL` priority.
4. **Interactive Dashboard**:
   - Streamlit web interface with single ticket testing, probability breakdown charts, and bulk CSV processing.

---

## 📁 Repository Structure

```
CUSTOMER COMPLAINT/
├── .venv/                         # Virtual environment
├── data/
│   ├── raw/                       # Symlink / source raw datasets
│   └── processed/                 # Balanced & cleaned master datasets
├── models/
│   └── complaint_classifier_pipeline.joblib  # Trained model & preprocessing artifact
├── notebooks/                     # EDA & experimentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset ingestion, balancing, and multi-dataset merger
│   ├── preprocessor.py            # Text cleaning, anonymization stripping & urgency detection
│   ├── train.py                   # Model training and evaluation pipeline
│   └── predict.py                 # Real-time and batch inference service
├── app/
│   └── app.py                     # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 📊 Pre-Trained Benchmark Results

Trained on a balanced sample of 44,178 real customer complaints:

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| **Mortgages & Home Loans** | 0.93 | 0.93 | **0.93** |
| **Student Loans** | 0.93 | 0.89 | **0.91** |
| **Money Transfers & Virtual Currency** | 0.90 | 0.84 | **0.87** |
| **Credit Cards & Prepaid** | 0.82 | 0.83 | **0.83** |
| **Retail Banking & Accounts** | 0.80 | 0.86 | **0.83** |
| **Vehicle Loans** | 0.79 | 0.82 | **0.80** |
| **Credit Reporting** | 0.77 | 0.79 | **0.78** |
| **Debt Collection** | 0.77 | 0.80 | **0.78** |
| **Personal & Consumer Loans** | 0.77 | 0.71 | **0.74** |
| **Overall Accuracy** | | | **83.06%** |

---

## 🛠️ Quick Start

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Run Inference in Python or CLI
```bash
python3 src/predict.py
```

### 3. Launch the Interactive Web App
```bash
streamlit run app/app.py
```

---

## ➕ How to Add More Datasets

To merge any new customer dataset (e.g., e-commerce reviews, Zendesk export, support emails):

```python
from src.data_loader import ComplaintDataLoader

loader = ComplaintDataLoader()

# 1. Load CFPB base dataset
df_cfpb = loader.load_cfpb_dataset(max_samples_per_category=5000)

# 2. Add your external dataset (CSV/JSON/Parquet)
df_new = loader.add_external_dataset(
    file_path="data/raw/my_new_tickets.csv",
    text_column="customer_message",
    category_column="department",
    category_mapping={
        "Billing Dept": "Retail Banking & Accounts",
        "Card Issues": "Credit Cards & Prepaid",
        # Custom category mappings...
    },
    source_name="zendesk_export"
)

# 3. Merge, deduplicate, and save as master dataset
loader.merge_and_save([df_cfpb, df_new], output_name="complaints_master.csv")
```

After merging your new dataset, simply re-run training:
```bash
python3 src/train.py
```
