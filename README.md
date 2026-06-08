# 🔍 Fraud Detection with Graph Neural Networks (GCN)

> **Detecting fraud through transaction graph analysis**
> GCN sees not just the transaction — but its connections across the entire network

---

## 🚀 Live Demo

| Service | Link |
|---|---|
| 🕸️ **Interactive Graph Visualizer** | [fraud-gnn.streamlit.app](https://fraud-gnn-j6nh9vg4mb4tdabnshx2n3.streamlit.app) |
| ⚡ **REST API** | [fraud-gnn-1.onrender.com](https://fraud-gnn-1.onrender.com) |
| 📖 **API Docs (Swagger)** | [fraud-gnn-1.onrender.com/docs](https://fraud-gnn-1.onrender.com/docs) |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Test AUC-ROC** | **0.7807** |
| **Fraud Recall** | **0.62** — catches 62% of all frauds |
| Fraud Precision | 0.10 |
| Graph Nodes | 590,540 transactions |
| Graph Edges | 493,718 connections |
| Model Parameters | 5,122 |

> 💡 **Why Recall > Precision in fraud detection?**
> Better to block 10 legitimate transactions than miss 1 fraudulent one.
> Financial losses from fraud far exceed the inconvenience of a false positive.

---

## 🧠 Key Idea — Why GNN beats traditional ML

```
Traditional ML (LightGBM):          Graph Neural Network (GCN):
─────────────────────────           ────────────────────────────
Sees each transaction               Sees transactions AND
in ISOLATION                        their CONNECTIONS

Features: amount, time,             Aggregates neighbor features
card type only                      automatically through graph

Misses: fraud rings,                Detects: coordinated fraud,
shared card patterns                money mule networks

card_001 ──── txn_A (normal)
         ──── txn_B (FRAUD)   ← GCN "infects" neighbors with suspicion
         ──── txn_C (normal?) ← elevated risk due to connection with txn_B
```

This is exactly how Visa and Mastercard's production anti-fraud systems work.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Graph ML** | PyTorch Geometric (`GCNConv`) |
| **Graph Construction** | NetworkX |
| **REST API** | FastAPI |
| **Visualization** | Streamlit + Plotly |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |
| **Dataset** | IEEE-CIS Fraud Detection (Kaggle) |
| **Deployment** | Render (API) + Streamlit Cloud (Visualizer) |

---

## ⚙️ Architecture

```
IEEE-CIS Dataset (590K transactions)
        │
        ▼
graph_builder.py
  Node  = one transaction (590,540 nodes)
  Edge  = two transactions share the same card (card1)
  Node features = amount, address, time deltas, counters (12 features)
        │
        ▼
GCN Model (model.py)
  GCNConv(12 → 64) → ReLU → Dropout(0.3)   ← 1st order neighbors
  GCNConv(64 → 64) → ReLU                   ← 2nd order neighbors
  Linear(64 → 2)   → Softmax                ← classification

  Total parameters: 5,122
        │
        ▼
train.py
  Weighted Cross-Entropy Loss
  (compensates class imbalance: 3.5% fraud vs 96.5% normal)
        │
        ├──► FastAPI /scan  (Render)          ← REST API
        └──► Streamlit app  (Streamlit Cloud) ← Graph Visualizer
```

---

## 🕸️ Interactive Graph Visualizer

Streamlit app with 3 tabs:

**Tab 1 — Graph Visualization**
- Interactive transaction graph — hover over nodes for details
- Node color = GNN fraud score (green → red)
- Node size = transaction amount
- Fraud propagation simulation — see how fraud signal spreads through the network

**Tab 2 — Analytics**
- KPI cards: fraud count, high-risk nodes, connectivity
- Amount distribution (fraud vs normal)
- GNN fraud score histogram

**Tab 3 — Live Scanner**
- Input transaction parameters via sidebar
- GCN model computes real-time fraud probability
- Gauge chart + decision (ALLOW / BLOCK)
- Explanation: amount vs average, recent activity, time since last

---

## ⚡ REST API

### GET /
```json
{
  "service": "Fraud GNN API",
  "model": "Graph Convolutional Network (GCN)",
  "metrics": {
    "auc_roc": 0.7807,
    "fraud_recall": 0.62,
    "graph_nodes": 590540,
    "graph_edges": 493718
  }
}
```

### POST /scan
```json
// Request
{
  "card_id": "card_001",
  "amount": 150000.00,
  "days_since_last_txn": 0.1,
  "txn_count_7d": 25,
  "avg_amount_30d": 3000.0
}

// Response
{
  "card_id": "card_001",
  "fraud_probability": 0.8821,
  "decision": "BLOCK",
  "risk_level": "HIGH",
  "model_used": "GCN-v1",
  "explanation": {
    "amount_vs_avg": "50.0x above average",
    "recent_activity": "25 transactions in last 7 days",
    "days_since_last": "0.1 days"
  }
}
```

### GET /health
```json
{
  "status": "healthy",
  "model": "GCN-v1",
  "auc_roc": 0.7807,
  "fraud_recall": 0.62
}
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/RaNurbekov/fraud-gnn.git
cd fraud-gnn
```

### 2. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install pandas numpy scikit-learn networkx fastapi uvicorn streamlit plotly
```

### 3. Download dataset
Download [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle:
- `train_transaction.csv`
- `train_identity.csv`

Place both files in `data/` folder.

### 4. Build the graph
```bash
python graph_builder.py
# → data/graph.pt (~590K nodes, ~493K edges)
```

### 5. Train the model
```bash
python train.py
# → best_model.pt | Test AUC-ROC: 0.7807
```

### 6. Run the API
```bash
cd api
uvicorn main:app --reload
# → http://127.0.0.1:8000/docs
```

### 7. Run the Visualizer
```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## 📁 Project Structure

```
fraud-gnn/
├── api/
│   └── main.py             # FastAPI: POST /scan, GET /health
├── data/                   # CSV files (gitignored)
│   └── graph.pt            # Built graph (gitignored)
├── graph_builder.py        # Graph construction from transactions
├── model.py                # FraudGCN architecture (GCNConv)
├── train.py                # Training + evaluation
├── streamlit_app.py        # Interactive graph visualizer
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔮 Roadmap

| Improvement | Expected Impact |
|---|---|
| **GAT** (Graph Attention Network) | +3-5% AUC via attention mechanism |
| **Heterogeneous graph** | Add `email` and `device` nodes — wider fraud network |
| **GraphSAGE** | Better scaling to larger graphs |
| **Temporal features** | Time between transactions per card |
| **SMOTE for graphs** | Oversampling of minority fraud class |

---

## 🔗 Related Projects

Part of a Fintech ML ecosystem:

- [**fraud-detection-api**](https://github.com/RaNurbekov/fraud-detection-api) — Hybrid anti-fraud: Redis Velocity Check + LightGBM + A/B Testing
- [**credit-risk-api**](https://github.com/RaNurbekov/credit-scoring-ml-api.) — Credit scoring with MLflow + SHAP + Evidently AI
- [**kafka-fraud-streaming**](https://github.com/RaNurbekov/kafka_anti_fraud) — Real-time Kafka streaming pipeline

> 💡 **GNN vs Classical ML:** `fraud-detection-api` uses LightGBM (AUC ~0.89, fast inference).
> This project uses GCN (AUC 0.78, but detects collaborative fraud through graph topology).
> Both approaches complement each other in production.

---

## 📫 Author

**Rashid Nurbekov** — ML Engineer | Fintech & Generative AI | Almaty, Kazakhstan 🇰🇿

[![Telegram](https://img.shields.io/badge/Telegram-@Ytyglika-2CA5E0?style=flat&logo=telegram&logoColor=white)](https://t.me/Ytyglika)
[![Email](https://img.shields.io/badge/Email-nurbekovrashidjob@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nurbekovrashidjob@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-RaNurbekov-181717?style=flat&logo=github&logoColor=white)](https://github.com/RaNurbekov)
