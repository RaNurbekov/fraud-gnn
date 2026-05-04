import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGCN(nn.Module):
    """
    Graph Convolutional Network для детекции мошенничества.
    
    Как работает GCN (простыми словами):
    - Обычная нейросеть смотрит только на саму транзакцию
    - GCN смотрит на транзакцию И её соседей в графе
    - Если у "нормальной" транзакции 3 соседа-фрода — это подозрительно
    - GCNConv автоматически агрегирует информацию от соседей
    """
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        """
        in_channels     — количество признаков каждого узла (= кол-во фичей)
        hidden_channels — размер скрытого слоя (64 — хороший дефолт)
        out_channels    — 2 класса: нормальный / фрод
        """
        super().__init__()
        
        # Слой 1: агрегирует информацию от соседей 1-го порядка
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Слой 2: агрегирует информацию от соседей 2-го порядка
        # (т.е. "видит" соседей соседей — более широкий контекст)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Финальный классификатор: обычный линейный слой
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Dropout — случайно "выключает" нейроны во время обучения
        # Помогает не переобучиться на тренировочных данных
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x, edge_index):
        """
        x          — матрица признаков узлов [N, in_channels]
        edge_index — список рёбер [2, E]
        """
        
        # Слой 1: GCN + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)            # нелинейность — без неё сеть = просто матричное умножение
        x = self.dropout(x)
        
        # Слой 2: GCN + ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Классификация: из embedding → вероятности классов
        x = self.classifier(x)
        
        return x  # логиты [N, 2] — потом применим softmax