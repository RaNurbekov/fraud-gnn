🔍 Fraud Detection with Graph Neural Networks

Детекция мошенничества на транзакционных данных с помощью Graph Convolutional Network (GCN) и FastAPI


📊 Результаты
МетрикаЗначениеTest AUC-ROC0.7807Fraud Recall0.62Fraud Precision0.10Узлов в графе590,540Рёбер в графе493,718Параметров модели5,122

Почему Recall важнее Precision в задаче фрод-детекции?
Лучше ложно заблокировать 10 нормальных транзакций, чем пропустить 1 мошенническую. Модель ловит 62% всех фродов.


🧠 Идея проекта
Обычные модели (LightGBM, XGBoost) смотрят на каждую транзакцию изолированно.
GNN смотрит на транзакцию в контексте её связей — если карта уже участвовала в подозрительных операциях, модель это видит.
card_001 ──── txn_A (normal)
         ──── txn_B (FRAUD)  ← GCN "заражает" соседей подозрением
         ──── txn_C (normal?) ← повышенный риск

🏗️ Архитектура
IEEE-CIS Dataset
      │
      ▼
graph_builder.py   ← строим граф: узлы = транзакции, рёбра = общая карта
      │
      ▼
  GCN Model        ← 2 слоя GCNConv + Dropout + Linear classifier
      │
      ▼
  train.py         ← weighted cross-entropy (компенсация дисбаланса 3.5% фрод)
      │
      ▼
FastAPI /scan      ← real-time inference
Граф транзакций

Узел = одна транзакция (590K узлов)
Ребро = две транзакции используют одну карту (card1)
Node features = сумма, адрес, временные дельты, счётчики (10 признаков)

GCN модель
pythonGCNConv(10 → 64)  →  ReLU  →  Dropout(0.3)
GCNConv(64 → 64)  →  ReLU
Linear(64 → 2)    →  Softmax

🚀 Быстрый старт
1. Установка зависимостей
bashpython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric
python -m pip install pandas numpy scikit-learn networkx fastapi uvicorn
2. Данные
Скачай с Kaggle IEEE-CIS Fraud Detection:

train_transaction.csv
train_identity.csv

Положи оба файла в папку data/
3. Построить граф
bashpython graph_builder.py
4. Обучить модель
bashpython train.py
5. Запустить API
bashcd api
python -m uvicorn main:app --reload
API доступен на http://127.0.0.1:8000/docs

📡 API
POST /scan
json// Request
{
  "card_id": "card_001",
  "amount": 150.00
}

// Response
{
  "card_id": "card_001",
  "amount": 150.00,
  "fraud_probability": 0.0821,
  "decision": "ALLOW",
  "model_used": "GCN-v1"
}

📁 Структура проекта
fraud_gnn/
├── data/
│   ├── train_transaction.csv   # не в репозитории (gitignore)
│   ├── train_identity.csv      # не в репозитории (gitignore)
│   └── graph.pt                # не в репозитории (gitignore)
├── api/
│   └── main.py                 # FastAPI сервер
├── graph_builder.py            # построение графа из CSV
├── model.py                    # архитектура GCN
├── train.py                    # обучение и оценка
├── requirements.txt
└── README.md

🔮 Что можно улучшить

 GAT (Graph Attention Network) — механизм внимания даёт +3-5% AUC
 Гетерогенный граф — добавить узлы email и device (сейчас только card)
 GraphSAGE — лучше масштабируется на большие графы
 Temporal features — учитывать время между транзакциями
 SMOTE для графов — oversampling миноритарного класса

PyTorch + PyTorch Geometric — GCN модель
NetworkX — анализ графа
FastAPI — REST API
scikit-learn — метрики и preprocessing
Датасет: IEEE-CIS Fraud Detection (Kaggle)