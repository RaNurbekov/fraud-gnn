import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path="data/"):
    """
    Загружаем два CSV от Kaggle:
    - train_transaction.csv  — основные данные о транзакции (сумма, время и т.д.)
    - train_identity.csv     — данные об устройстве и email
    """
    txn = pd.read_csv(f"{path}train_transaction.csv")
    idn = pd.read_csv(f"{path}train_identity.csv")
    
    # Объединяем по TransactionID (не все транзакции имеют identity)
    df = txn.merge(idn, on="TransactionID", how="left")
    print(f"Загружено транзакций: {len(df)}")
    print(f"Фрод: {df['isFraud'].mean():.2%}")  # ожидаем ~3.5%
    return df

def build_graph(df):
    """
    Строим гомогенный граф: каждая транзакция = узел.
    Рёбра: две транзакции связаны если у них одна карта (card1).
    
    Почему так:
    - Мошенник использует одну карту для нескольких транзакций
    - GCN "смотрит" на соседей узла и обнаруживает подозрительные паттерны
    """
    
    # --- 1. Готовим признаки узлов (node features) ---
    
    # Выбираем числовые колонки (GNN работает только с числами)
    feature_cols = [
        'TransactionAmt',   # сумма
        'card1', 'card2',   # номер карты (закодированный)
        'addr1', 'addr2',   # адрес
        'dist1', 'dist2',   # расстояние
        'C1', 'C2', 'C3',   # счётчики (Countings)
        'D1', 'D2',         # временные дельты
    ]
    
    # Берём только те колонки которые реально есть в данных
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    
    # Заполняем пропуски нулями (в реальном данных их много)
    X = X.fillna(0)
    
    # Нормализация: приводим все числа к шкале ~[-1, 1]
    # Это важно для нейронных сетей — они плохо работают с "сырыми" большими числами
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Переводим в тензор PyTorch
    node_features = torch.tensor(X_scaled, dtype=torch.float)
    
    # Метки: 0 = нормальная транзакция, 1 = фрод
    labels = torch.tensor(df['isFraud'].values, dtype=torch.long)
    
    # --- 2. Строим рёбра графа ---
    
    # Идея: если две транзакции имеют одинаковый card1 — они связаны
    # card1 — это закодированный номер карты в датасете
    
    print("Строим рёбра по card1 (это может занять минуту)...")
    
    # Группируем индексы транзакций по значению card1
    # {card_value: [idx1, idx2, idx3, ...]}
    card_groups = df.groupby('card1').apply(lambda g: g.index.tolist())
    
    src_nodes = []  # откуда идёт ребро
    dst_nodes = []  # куда идёт ребро
    
    for card_val, indices in card_groups.items():
        # Соединяем все транзакции одной карты между собой
        # Но не больше 10 транзакций на карту — иначе граф станет слишком плотным
        indices = indices[:10]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                src_nodes.append(indices[i])
                dst_nodes.append(indices[j])
                # Граф неориентированный: добавляем ребро в обе стороны
                src_nodes.append(indices[j])
                dst_nodes.append(indices[i])
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    print(f"Узлов (транзакций): {node_features.shape[0]}")
    print(f"Рёбер: {edge_index.shape[1]}")
    
    # --- 3. Маски для обучения/валидации ---
    # В графовых задачах мы делаем сплит не по данным а по маскам
    n = len(df)
    indices = torch.randperm(n)  # перемешиваем случайно
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:int(0.7 * n)]] = True   # 70% — обучение
    val_mask[indices[int(0.7 * n):int(0.85 * n)]] = True  # 15% — валидация
    test_mask[indices[int(0.85 * n):]] = True    # 15% — тест
    
    # Собираем всё в объект Data — стандартный контейнер PyG
    from torch_geometric.data import Data
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data

if __name__ == "__main__":
    df = load_data()
    data = build_graph(df)
    torch.save(data, "data/graph.pt")
    print("Граф сохранён в data/graph.pt")