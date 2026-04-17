# Agentic Debate RAG for Financial Risk Extraction & Prediction

## 📌 Overview

This project presents a **hybrid Retrieval-Augmented Generation (RAG) system enhanced with an Agentic Debate framework** to extract and predict financial risks from SEC filings (10-K and 10-Q reports).

The system combines:
- Hybrid Information Retrieval (BM25 + Dense Embeddings)
- Multi-Agent LLM Reasoning (Detector, Skeptic, Synthesizer)
- Feature Engineering from extracted risks
- Predictive Modeling (Logistic Regression, Random Forest, LSTM)

The goal is to build a **reliable, interpretable, and scalable pipeline** for financial risk analysis using unstructured corporate disclosures.

## 🎯 Motivation & Justification

Traditional financial risk prediction relies heavily on:
- Structured financial ratios
- Market indicators

However, **critical risk signals are embedded in unstructured text**, such as:
- Risk Factors (Item 1A)
- Management Discussion & Analysis (MD&A)

### ❗ Key Challenges
- Single-agent LLMs suffer from **hallucination**
- Poor **factual grounding**
- Weak integration with predictive models

### ✅ Our Solution
We introduce an **Agentic Debate Framework**:
- 🧠 **Detector** → Extracts risks
- 🔍 **Skeptic** → Validates evidence (reduces hallucination)
- 📊 **Synthesizer** → Produces structured outputs with confidence

This improves:
- Reliability
- Interpretability
- Downstream prediction performance
## 🏗️ System Architecture

SEC Filings → Preprocessing → Chunking → Retrieval (BM25 + Dense)
→ Agentic Debate (Detector → Skeptic → Synthesizer)
→ Feature Engineering → ML/DL Models → Predictions

## 📂 Project Structure
``` text
project_root/
│
├── logs/
│ ├── logcounter.py
│ └── preprocessing_log.txt
│
├── notebooks/
│ ├── evaluation_visuals.ipynb
│ └── exploratory_analysis.ipynb
│
├── src/
│ ├── data_ingestion/
│ │ ├── checker.py
│ │ ├── company_selector.py
│ │ ├── filter_chunks.py
│ │ └── sec_api_downloader.py
│ │
│ ├── evaluation/
│ │ └── retrieval_evaluator.py
│ │
│ ├── features/
│ │ ├── feature_engineering.py
│ │ ├── metadata_extractor.py
│ │ └── sequence_builder.py
│ │
│ ├── generation/
│ │ ├── batch_llm_pipeline.py
│ │ ├── detector_agent.py
│ │ ├── llm_interface.py
│ │ ├── optimized_agent.py
│ │ ├── skeptic_agent.py
│ │ ├── synthesizer_agent.py
│ │ ├── test_detector.py
│ │ └── test_skeptic.py
│ │
│ ├── llm/
│ │ └── agentic_debate_engine.py
│ │
│ ├── models/
│ │ ├── baseline_models.py
│ │ └── lstm_model.py
│ │
│ ├── preprocessing/
│ │ ├── chunking/
│ │ │ ├── batch_chunk_pipeline.py
│ │ │ └── chunk_generator.py
│ │ │
│ │ └── section_extraction/
│ │   ├── document_parser.py
│ │   ├── section_extractor.py
│ │   └── test_section_extraction.py
│ │
│ ├── retrieval/
│ │ ├── bm25_retriever.py
│ │ ├── dense_retriever.py
│ │ ├── embedding_generator.py
│ │ ├── hybrid_retriever.py
│ │ ├── retrieval_evaluator.py
│ │ ├── test.py
│ │ ├── test_bm25.py
│ │ ├── test_dense.py
│ │ └── test_hybrid.py
│ │
│ └── tests/
│   └── test_full_pipeline.py
│
├── main_pipeline.py
├── .gitignore
├── Link.txt
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/agentic-financial-risk.git
cd agentic-financial-risk
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\\Scripts\\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### 🔹 Step 1: Download SEC Filings
(If not already available)
```bash
python -m src.data_collection.download_filings
```

### 🔹 Step 2: Preprocessing (Extract Sections + Chunking)
```bash
python -m src.preprocessing.run_preprocessing
```

Output:
- Cleaned text
- JSON chunks

### 🔹 Step 3: Build Retrieval Index
```bash
python -m src.retrieval.build_index
```

Includes:
- BM25 index
- Dense embeddings (.npy)

### 🔹 Step 4: Test Retrieval
```bash
python -m src.retrieval.test_hybrid
```

### 🔹 Step 5: Run Agentic Debate Pipeline
```bash
python -m src.generation.run_pipeline
```

⚠️ Note:
- This is computationally expensive
- Uses Ollama (local LLM)

### 🔹 Step 6: Feature Engineering
```bash
python -m src.feature_engineering.build_features
```

Output:
- feature_dataset.csv

### 🔹 Step 7: Train Models
```bash
python -m src.modeling.train_models
```

Models:
- Logistic Regression
- Random Forest
- LSTM

## 📊 Example Output

### Risk Extraction (JSON)
```json
[
  {
    "risk_category": "Credit Risk",
    "confidence": 0.9
  },
  {
    "risk_category": "Supply Chain Risk",
    "confidence": 0.8
  }
]
```

### Feature Dataset
liquidity_risk_count,credit_risk_count,...,mean_confidence
1,0,...,0.8
## 📈 Results Summary

Model Accuracy F1-score

Logistic Regression
0.96
0.94

Random Forest
0.98
0.97

LSTM
0.63
0.53

Key Insights:
- Hybrid retrieval improves relevance
- Agentic debate reduces hallucination
- LLM-derived features are predictive
- Tree-based models outperform LSTM (low data setting)

## 🧪 Evaluation

Retrieval:
- Precision@K
- Recall@K
- MRR

Prediction:
- Accuracy
- Precision
- Recall
- F1-score

Ablation Study:
- With Skeptic vs Without Skeptic

## ⚠️ Limitations

- Proxy labels (no real distress labels)
- High computational cost (LLM pipeline)
- Limited dataset size
- CPU-based constraints

## 🔮 Future Work

- Use real financial distress labels
- Scale dataset (1000+ companies)
- Optimize LLM inference speed
- Add cross-encoder reranking
- Improve feature engineering (embeddings + sentiment)

## 🧠 Technologies Used

- Python
- BM25 (rank_bm25)
- Sentence Transformers
- Ollama (Local LLM)
- Scikit-learn
- PyTorch / TensorFlow (LSTM)

## 👥 Authors

Grace Gaikwad  
Himanshu Rajput  
Prajwal Bhandarkar  
Tushar Puntambekar  

MSc Data Analytics  
National College of Ireland

## 📜 License

This project is for academic/research purposes.

## ⭐ Final Note

This project demonstrates how combining:
- Retrieval
- Multi-agent LLM reasoning
- Machine learning

can create a robust and interpretable financial risk analysis system.
