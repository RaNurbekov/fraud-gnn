import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append("..")
from model import FraudGCN

app = FastAPI(title="Fraud GNN API")

# Загружаем модель один раз при старте сервера
model = None
graph_data = None

@app.on_event("startup")
def load_model():
    global model, graph_data
    graph_data = torch.load("../data/graph.pt", weights_only=False)
    
    in_channels = graph_data.x.shape[1]
    model = FraudGCN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2
    )
    model.load_state_dict(torch.load("../best_model.pt", map_location="cpu", weights_only=False))
    model.eval()
    print("Модель загружена!")

class Transaction(BaseModel):
    card_id: str
    amount: float

@app.post("/scan")
def scan_transaction(txn: Transaction):
    """
    В production здесь мы бы добавляли новый узел в граф.
    Для демо — возвращаем mock предсказание на основе суммы.
    """
    with torch.no_grad():
        # В реальном проекте: найти узел карты в графе и запустить инференс
        # Здесь упрощённая версия для демонстрации API
        out = model(graph_data.x, graph_data.edge_index)
        fraud_probs = F.softmax(out, dim=1)[:, 1]
        avg_fraud_prob = fraud_probs.mean().item()
        
        # Эвристика: большая сумма = выше риск (для демо)
        risk_score = min(0.95, avg_fraud_prob + (txn.amount / 10000))
        
        return {
            "card_id": txn.card_id,
            "amount": txn.amount,
            "fraud_probability": round(risk_score, 4),
            "decision": "BLOCK" if risk_score > 0.5 else "ALLOW",
            "model_used": "GCN-v1"
        }