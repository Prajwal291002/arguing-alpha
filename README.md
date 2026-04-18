# Agentic Debate RAG for Financial Risk Extraction & Prediction

## 📌 Overview

This project implements a **scalable, end-to-end pipeline** for extracting and predicting financial risks from SEC filings using:

- Hybrid Retrieval (BM25 + Dense Embeddings)
- Agentic Multi-LLM Debate (Detector, Skeptic, Synthesizer)
- Feature Engineering from textual risk signals
- Predictive Modeling (ML + LSTM)

The system processes **10-K and 10-Q filings**, extracts structured risk signals, and uses them for **financial distress prediction**.

---

## 🎯 Motivation & Justification

Traditional financial risk prediction relies heavily on structured data (financial ratios), but:

- ❌ Ignores rich textual disclosures
- ❌ Cannot capture nuanced risk signals
- ❌ LLMs alone suffer from hallucination

### 🚨 Core Problem
Single-agent LLM pipelines:
- Over-extract risks
- Produce unsupported claims
- Lack internal validation

### ✅ Our Solution
We introduce an **Agentic Debate Framework**:

| Agent | Role |
|------|------|
| Detector | Extracts risks |
| Skeptic | Validates evidence |
| Synthesizer | Produces structured output |

This ensures:
- Higher factual grounding
- Reduced hallucination
- Better downstream predictions

---

## 🏗️ System Pipeline

SEC API → Raw Filings → Section Extraction → Chunking  
→ Hybrid Retrieval (BM25 + Dense)  
→ Agentic Debate (Detector → Skeptic → Synthesizer)  
→ Feature Engineering → ML/DL Models → Prediction

---

## 📂 Project Structure

```text
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
│
│ ├── data_ingestion/
│ │ ├── sec_api_downloader.py
│ │ ├── company_selector.py
│ │ ├── checker.py
│ │ └── filter_chunks.py
│
│ ├── preprocessing/
│ │ ├── section_extraction/
│ │ │ ├── document_parser.py
│ │ │ ├── section_extractor.py
│ │ │ └── test_section_extraction.py
│ │ │
│ │ └── chunking/
│ │   ├── chunk_generator.py
│ │   └── batch_chunk_pipeline.py
│
│ ├── retrieval/
│ │ ├── bm25_retriever.py
│ │ ├── dense_retriever.py
│ │ ├── embedding_generator.py
│ │ ├── hybrid_retriever.py
│ │ ├── retrieval_evaluator.py
│ │ └── test_*.py
│
│ ├── generation/
│ │ ├── batch_llm_pipeline.py
│ │ ├── detector_agent.py
│ │ ├── skeptic_agent.py
│ │ ├── synthesizer_agent.py
│ │ ├── optimized_agent.py
│ │ └── llm_interface.py
│
│ ├── llm/
│ │ └── agentic_debate_engine.py
│
│ ├── features/
│ │ ├── feature_engineering.py
│ │ ├── metadata_extractor.py
│ │ └── sequence_builder.py
│
│ ├── models/
│ │ ├── baseline_models.py
│ │ └── lstm_model.py
│
│ ├── evaluation/
│ │ └── retrieval_evaluator.py
│
│ └── tests/
│   └── test_full_pipeline.py
│
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <repo_url>
cd project_root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 How to Run (STEP-BY-STEP)

🔹 1. Download SEC Filings
```bash
python -m src.data_ingestion.sec_api_downloader
```

🔹 2. Extract Sections (Item 1A & MD&A)
```bash
python -m src.preprocessing.section_extraction.document_parser
```

🔹 3. Chunk Documents
```bash
python -m src.preprocessing.chunking.batch_chunk_pipeline
```

🔹 4. Generate Embeddings
```bash
python -m src.retrieval.embedding_generator
```

🔹 5. Test Retrieval
```bash
python -m src.retrieval.test_hybrid
```

🔹 6. Run LLM Pipeline (⚠️ Heavy Step)
```bash
python -m src.generation.batch_llm_pipeline
```

Uses Ollama (local LLM)  
Processes chunks → extracts risks

🔹 7. Feature Engineering
```bash
python -m src.features.feature_engineering
```

🔹 8. Build Sequences
```bash
python -m src.features.sequence_builder
```

🔹 9. Train Models
```bash
python -m src.models.baseline_models
python -m src.models.lstm_model
```

🔹 10. Full Pipeline Test
```bash
python -m src.tests.test_full_pipeline
```

## 📊 Example Output

### Risk Extraction
```json
[
  {
    "risk_category": "Credit Risk",
    "confidence": 0.9
  }
]
```

### Feature Dataset
```text
liquidity_risk_count,credit_risk_count,...,mean_confidence
1,0,...,0.8
```

## 📈 Key Results

- Hybrid Retrieval > BM25 / Dense individually
- Agentic Debate improves reliability
- Random Forest performs best (0.98 accuracy)
- LSTM underperforms due to small dataset

## ⚠️ Limitations

- Proxy labels (no ground truth)
- High LLM computation cost
- Limited sequence dataset
- CPU-based execution constraints

## 🔮 Future Work

- Add real distress labels
- Scale dataset (1000+ companies)
- Optimize LLM inference
- Add reranking (cross-encoder)
- Improve feature richness

## 🧠 Tech Stack

- Python
- rank_bm25
- sentence-transformers
- Ollama (LLM)
- scikit-learn
- PyTorch / TensorFlow

## 👥 Authors

- Grace Gaikwad
- Himanshu Rajput
- Prajwal Bhandarkar
- Tushar Puntambekar
- National College of Ireland

## ⭐ Final Note

This project demonstrates how combining:
- Retrieval
- Multi-agent reasoning
- Machine learning

can produce a reliable and interpretable financial risk analysis system.
