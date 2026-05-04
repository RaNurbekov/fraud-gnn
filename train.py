import torch
import torch.nn.functional as F
from model import FraudGCN
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

def train_epoch(model, data, optimizer):
    """Один шаг обучения"""
    model.train()           # включаем режим обучения (dropout активен)
    optimizer.zero_grad()   # сбрасываем градиенты с прошлого шага
    
    # Прямой проход: получаем предсказания для ВСЕХ узлов
    out = model(data.x, data.edge_index)
    
    # Считаем loss ТОЛЬКО на тренировочных узлах (train_mask)
    # weight — веса классов чтобы компенсировать дисбаланс (фрода мало)
    # Если фрода 3.5%, то его вес = 1/0.035 ≈ 28
    fraud_weight = (data.y[data.train_mask] == 0).sum().float() / \
                   (data.y[data.train_mask] == 1).sum().float()
    weight = torch.tensor([1.0, fraud_weight.item()])
    
    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask],
        weight=weight
    )
    
    loss.backward()   # обратный проход: считаем градиенты
    optimizer.step()  # обновляем веса модели
    
    return loss.item()

@torch.no_grad()  # во время валидации градиенты не нужны — экономит память
def evaluate(model, data, mask):
    """Оцениваем модель на заданной маске (val или test)"""
    model.eval()  # выключаем dropout
    
    out = model(data.x, data.edge_index)
    pred_probs = F.softmax(out, dim=1)[:, 1]  # вероятность фрода
    pred_labels = out.argmax(dim=1)           # предсказанный класс
    
    # Берём только нужные узлы
    y_true = data.y[mask].numpy()
    y_prob = pred_probs[mask].numpy()
    y_pred = pred_labels[mask].numpy()
    
    auc = roc_auc_score(y_true, y_prob)
    return auc, y_true, y_pred

def main():
    # Загружаем граф
    print("Загружаем граф...")
    data = torch.load("data/graph.pt", weights_only=False)
    print(f"Узлов: {data.num_nodes}, Рёбер: {data.num_edges}")
    
    # Создаём модель
    in_channels = data.x.shape[1]  # кол-во фичей автоматически
    model = FraudGCN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2
    )
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # Adam — стандартный оптимизатор, lr=0.01 хорошо работает для GCN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Обучение
    best_val_auc = 0
    patience = 10       # если нет улучшения 10 эпох — останавливаемся
    no_improve = 0
    
    print("\nОбучение...")
    for epoch in range(1, 101):
        loss = train_epoch(model, data, optimizer)
        
        if epoch % 5 == 0:
            val_auc, _, _ = evaluate(model, data, data.val_mask)
            print(f"Эпоха {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), "best_model.pt")
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break
    
    # Финальная оценка на тесте
    print("\n--- Результаты на тестовой выборке ---")
    model.load_state_dict(torch.load("best_model.pt"))
    test_auc, y_true, y_pred = evaluate(model, data, data.test_mask)
    print(f"Test AUC: {test_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

if __name__ == "__main__":
    main()