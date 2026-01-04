# A-Real-Time-Deep-Learning-Based-Intrusion-Detection-System-Using-CTM-Guided-LSTM-Modeling
 Overview

This repository presents a real-time Intrusion Detection System (IDS) that integrates Correlation-based Traffic Modeling (CTM) feature weighting with a Long Short-Term Memory (LSTM) neural network for network traffic analysis.

The system supports:

Multiclass attack classification

Binary anomaly detection

Real-time streaming inference

Live alert visualization via dashboard

The project is designed as a research-ready, modular, and extensible system, suitable for academic evaluation and future deployment.

 Objectives

Design a sequence-based IDS capable of detecting temporal attack patterns

Integrate CTM feature selection to improve efficiency and interpretability

Enable real-time inference using a sliding window mechanism

Provide offline evaluation and live monitoring capabilities

Maintain a clean, reproducible research codebase

 Model Architecture
CTM + LSTM Pipeline

CTM Feature Weighting

Statistical feature importance vector applied prior to inference

LSTM Network

Processes fixed-length sequences of network features

Outputs multiclass probabilities

Anomaly Scoring

Anomaly score = 1 − P(normal)

Threshold-Based Decision

Centralized threshold logic for anomaly detection

See Figure 1 in the report: “LSTM-based IDS Architecture with CTM Feature Selection”

 System Design (Real-Time)

The system operates using a stream-based inference pipeline:

Feature Stream
      ↓
Sliding Window Buffer
      ↓
CTM + LSTM Model
      ↓
Threshold Decision
      ↓
Alert Logging
      ↓
Live Dashboard


This design ensures:

Separation of concerns

Low-latency inference

Easy extension to deployment environments

 See Figure 2: “Real-Time IDS Pipeline”

 Experimental Evaluation
Offline Evaluation Includes:

Ground truth labels

Predicted classes

Class probabilities

Anomaly scores

Metrics:

Accuracy

Precision / Recall / F1-score

Confusion Matrix

Threshold sensitivity analysis

 Planned outputs:

Table 2: Classification Performance Metrics

Figure 3: Confusion Matrix

Figure 4: Anomaly Score Distribution

Figure 5: Threshold vs Recall / False Positive Rate

 Live Simulation & Visualization

A live traffic simulator feeds feature rows into the IDS engine continuously.
A Streamlit dashboard visualizes system behavior in real time:

Total alerts detected

Latest anomaly score

Score timeline

Recent alerts

 See Figure 6: “Live Intrusion Detection Dashboard”

 Project Structure
Bot-Lot-dataset/
│
├── ids/
│   ├── inference.py        # Model loading and prediction
│   ├── thresholds.py       # Centralized anomaly threshold logic
│   ├── stream_ids.py       # Sliding window IDS engine
│   └── __init__.py
│
├── analysis/
│   ├── evaluate_results.py # Offline evaluation
│   ├── visualize_results.py# Plot generation
│   └── figures/
│
├── simulation/
│   └── simulate_stream.py  # Live traffic simulation
│
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── alerts.jsonl        # Alert log
│
├── notebook/
│   ├── ctm_lstm_ids_model.keras
│   └── ctm_weights.npy
│
├── api/
│   └── app.py              # (Future deployment interface)
│
├── requirements.txt
└── README.md


 Appendix A in the report documents this structure.

 How to Run
1️⃣ Setup Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

2️⃣ Run Live Simulation
python -m simulation.simulate_stream

3️⃣ Launch Dashboard
streamlit run dashboard/app.py

 Current Status

✔ Fully implemented
✔ Offline evaluation completed
✔ Real-time simulation validated
✔ Live dashboard operational

 Future Work

Controlled attack injection

PCAP-based feature extraction

Adaptive thresholding

FastAPI + Docker deployment

Edge and cloud inference

🎓Academic Context

This project is suitable for:

Master’s thesis

Research publication

IDS prototyping

Security analytics demonstrations

 License

This project is provided for academic and research purposes.
Please cite appropriately if used in scholarly work.

