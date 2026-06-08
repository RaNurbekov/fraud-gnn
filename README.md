# 🔍 Fraud Detection with Graph Neural Networks (GCN)

> **Детекция мошенничества через анализ транзакционного графа**
> GCN видит не только транзакцию, но и её связи — как Visa и Mastercard в production

---

## 📊 Результаты

| Метрика | Значение |
|---|---|
| **Test AUC-ROC** | **0.7807** |
| **Fraud Recall** | **0.62** (ловит 62% всех фродов) |
| Fraud Precision | 0.10 |
| Узлов в графе | 590,540 |
| Рёбер в графе | 493,718 |
| Параметров модели | 5,122 |

> 💡 **Почему Recall важнее Precision в антифроде?** Лучше ложно заблокировать 10 нормальных транзакций, чем пропустить 1 мошенническую. Финансовые потери от фрода несравнимо выше, чем неудобство честного клиента.

---

## 🧠 Ключевая идея: почему GNN лучше классических моделей

Обычные модели (LightGBM, XGBoost) смотрят на каждую транзакцию **изолированно**. GNN смотрит на транзакцию **в контексте её связей** в графе:

```
card_001 ──── txn_A (normal)
         ──── txn_B (FRAUD)   ← GCN "заражает" соседей подозрением
         ──── txn_C (normal?) ← повышенный риск из-за связи с txn_B
```

Если карта уже участвовала в подозрительных операциях — модель это видит через рёбра графа. Именно так работают антифрод системы Visa, Mastercard и PayPal.

---

## 🛠 Стек технологий

| Компонент | Технология |
|---|---|
| **Deep Learning** | PyTorch |
| **Graph ML** | PyTorch Geometric (`GCNConv`) |
| **Graph Construction** | NetworkX |
| **Backend API** | FastAPI |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |
| **Dataset** | IEEE-CIS Fraud Detection (Kaggle) |

---

## ⚙️ Архитектура системы

```
IEEE-CIS Dataset (590K транзакций)
        │
        ▼
graph_builder.py
  Узел = транзакция (590,540 узлов)
  Ребро = общая карта (card1) между транзакциями
  Node features = сумма, адрес, временные дельты (10 признаков)
        │
        ▼
GCN Model (model.py)
  GCNConv(10 → 64) → ReLU → Dropout(0.3)   ← соседи 1-го порядка
  GCNConv(64 → 64) → ReLU                   ← соседи 2-го порядка
  Linear(64 → 2)   → Softmax                ← классификация
  
  Всего параметров: 5,122
        │
        ▼
train.py
  Weighted Cross-Entropy Loss
  (компенсация дисбаланса классов: 3.5% фрод vs 96.5% норма)
        │
        ▼
api/main.py (FastAPI)
  POST /scan → real-time inference
```

---

## 🔑 Ключевые особенности

### 1. Граф транзакций как Feature Store
Вместо ручного feature engineering (velocity check, count features) граф автоматически кодирует историю поведения карты через топологию рёбер. Каждый узел "видит" своих соседей через механизм агрегации GCNConv.

### 2. Двухслойная агрегация соседей
```python
# Слой 1: видит транзакции на той же карте (1-й порядок)
self.conv1 = GCNConv(in_channels, hidden_channels)

# Слой 2: видит транзакции на картах связанных карт (2-й порядок)
# — более широкий контекст мошеннической сети
self.conv2 = GCNConv(hidden_channels, hidden_channels)
```

### 3. Weighted Loss для дисбаланса классов
Фрод составляет лишь 3.5% датасета. Weighted Cross-Entropy штрафует модель за пропущенные фроды в 28x сильнее, чем за ложные срабатывания.

### 4. FastAPI /scan для real-time инференса
```json
// POST /scan
{
  "card_id": "card_001",
  "amount": 150.00
}

// Response
{
  "card_id": "card_001",
  "fraud_probability": 0.0821,
  "decision": "ALLOW",
  "model_used": "GCN-v1"
}
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pandas numpy scikit-learn networkx fastapi uvicorn
```

### 2. Данные
Скачайте с Kaggle [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection):
- `train_transaction.csv`
- `train_identity.csv`

Положите оба файла в папку `data/`.

### 3. Построить граф
```bash
python graph_builder.py
# → data/graph.pt (~590K узлов, ~493K рёбер)
```

### 4. Обучить модель
```bash
python train.py
# → Test AUC-ROC: 0.7807
```

### 5. Запустить API
```bash
cd api
uvicorn main:app --reload
# → http://127.0.0.1:8000/docs
```

---

## 📁 Структура проекта

```
fraud-gnn/
├── api/
│   └── main.py           # FastAPI: POST /scan
├── data/                 # CSV файлы (gitignore)
├── graph_builder.py      # Построение графа из транзакций
├── model.py              # Архитектура FraudGCN
├── train.py              # Обучение + оценка метрик
├── requirements.txt
└── README.md
```
## 🚀 Live Demo

🔗 **[fraud-gnn-1.onrender.com](https://fraud-gnn-1.onrender.com)**

| Endpoint | Description |
|---|---|
| [GET /](https://fraud-gnn-1.onrender.com/) | Model info & metrics |
| [POST /scan](https://fraud-gnn-1.onrender.com/docs) | Fraud scoring |
| [GET /health](https://fraud-gnn-1.onrender.com/health) | Health check |
| [GET /docs](https://fraud-gnn-1.onrender.com/docs) | Swagger UI |
---

## 🔮 Направления улучшений

| Улучшение | Ожидаемый эффект |
|---|---|
| **GAT** (Graph Attention Network) | +3-5% AUC за счёт механизма внимания |
| **GraphSAGE** | Лучшее масштабирование на большие графы |
| **Гетерогенный граф** | Добавить узлы `email` и `device` — расширить связи |
| **Temporal features** | Учитывать время между транзакциями одной карты |

---

## 🔗 Связанные проекты

- [**fraud-detection-api**](https://github.com/RaNurbekov/fraud-detection-api) — гибридный антифрод: Redis Velocity Check + LightGBM + A/B Testing
- [**kafka-fraud-streaming**](https://github.com/RaNurbekov/kafka-fraud-streaming) — потоковая архитектура Kafka для real-time детекции

> 💡 **GNN vs классический ML:** `fraud-detection-api` использует LightGBM (AUC ~0.89, быстрый инференс), этот проект использует GCN (AUC 0.78, но видит коллаборативный фрод через граф). Оба подхода дополняют друг друга в production.
