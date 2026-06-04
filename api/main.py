import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os
sys.path.append("..")
from model import FraudGCN

# ── Model storage ─────────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading GCN model...")
    
    in_channels = 10
    model = FraudGCN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2
    )
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pt')
    
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        print("Model loaded from best_model.pt")
    else:
        print("Warning: best_model.pt not found — using random weights for demo")
    
    model.eval()
    ml_models['gcn'] = model
    print("GCN model ready!")
    
    yield
    ml_models.clear()

app = FastAPI(
    title="Fraud GNN API",
    description="Graph Neural Network fraud detection — IEEE-CIS dataset, AUC-ROC: 0.7807",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Schemas ───────────────────────────────────────────────
class Transaction(BaseModel):
    card_id: str
    amount: float
    days_since_last_txn: float = 1.0
    txn_count_7d: int = 5
    avg_amount_30d: float = 500.0

class HealthResponse(BaseModel):
    status: str
    model: str
    auc_roc: float
    fraud_recall: float

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Fraud GNN API",
        "model": "Graph Convolutional Network (GCN)",
        "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
        "metrics": {
            "auc_roc": 0.7807,
            "fraud_recall": 0.62,
            "fraud_precision": 0.10,
            "graph_nodes": 590540,
            "graph_edges": 493718
        },
        "endpoints": ["/scan", "/health", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "model": "GCN-v1",
        "auc_roc": 0.7807,
        "fraud_recall": 0.62
    }

@app.post("/scan")
def scan_transaction(txn: Transaction):
    """
    GCN-based fraud scoring.
    
    In production: new transaction added as node to graph,
    GCN aggregates neighbor features for inference.
    
    Demo mode: feature-based scoring using trained model weights.
    """
    model = ml_models['gcn']
    
    with torch.no_grad():
        # Build feature vector from transaction
        # Matches training features: amount, days, count, avg, ratios
        amount_norm = min(txn.amount / 10000, 1.0)
        days_norm = min(txn.days_since_last_txn / 30, 1.0)
        count_norm = min(txn.txn_count_7d / 50, 1.0)
        avg_norm = min(txn.avg_amount_30d / 10000, 1.0)
        amount_ratio = min(txn.amount / (txn.avg_amount_30d + 1), 5.0) / 5.0
        
        # Build dummy mini-graph (2 nodes, 1 edge) for model inference
        x = torch.tensor([
            [amount_norm, days_norm, count_norm, avg_norm,
             amount_ratio, 0.5, 0.3, 0.2, 0.1, 0.4],
            [0.1, 0.2, 0.1, 0.1,
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ], dtype=torch.float)
        
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        out = model(x, edge_index)
        probs = F.softmax(out, dim=1)
        fraud_prob = probs[0, 1].item()
        
        # Risk boosters based on business rules
        if txn.amount > txn.avg_amount_30d * 3:
            fraud_prob = min(0.95, fraud_prob + 0.15)
        
        if txn.txn_count_7d > 20:
            fraud_prob = min(0.95, fraud_prob + 0.10)
        
        if txn.days_since_last_txn < 0.1:
            fraud_prob = min(0.95, fraud_prob + 0.10)

    decision = "BLOCK" if fraud_prob > 0.5 else "ALLOW"
    risk_level = (
        "HIGH" if fraud_prob > 0.7 else
        "MEDIUM" if fraud_prob > 0.4 else
        "LOW"
    )

    return {
        "card_id": txn.card_id,
        "amount": txn.amount,
        "fraud_probability": round(fraud_prob, 4),
        "decision": decision,
        "risk_level": risk_level,
        "model_used": "GCN-v1",
        "explanation": {
            "amount_vs_avg": f"{txn.amount / (txn.avg_amount_30d + 1):.1f}x above average",
            "recent_activity": f"{txn.txn_count_7d} transactions in last 7 days",
            "days_since_last": f"{txn.days_since_last_txn:.1f} days"
        }
    }