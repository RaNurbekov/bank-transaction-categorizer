# 💳 Bank Transaction NLP Categorizer

Система автоматической категоризации "грязных" банковских транзакций (PFM - Personal Finance Management) с использованием Deep Learning. 

## 🛠 Стек технологий
* **Deep Learning:** PyTorch, Hugging Face `transformers`, `datasets`
* **Model:** `distilbert-base-multilingual-cased` (Fine-Tuning)
* **Frontend:** Streamlit
* **Data:** Синтетический датасет банковских чеков (генерация с шумом)

## ⚙️ Особенности проекта
1. **Fine-Tuning Трансформеров:** Базовая многоязычная модель DistilBERT дообучена (Sequence Classification) на распознавание 5 банковских категорий (Транспорт, Супермаркеты, Подписки и т.д.).
2. **Robustness (Устойчивость к шуму):** Модель успешно справляется с опечатками, латиницей, сокращениями и случайными цифрами (ID терминалов), выхватывая семантический контекст транзакции.
3. **Inference UI:** Удобный веб-интерфейс на Streamlit для демонстрации предсказаний и вывода уверенности модели (Confidence Score).

## 🚀 Как запустить

### 1. Подготовка
Установите зависимости (потребуется PyTorch):
```bash
pip install -r requirements.txt