import pandas as pd
import random
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

print("1. Генерируем синтетический датасет банковских транзакций...")
# Шаблоны грязных транзакций
templates = {
    "Супермаркеты":["MAGNUM CASH&CARRY ALMATY", "SMALL SUPERMARKET KAZ", "GALMART ASTANA", "IP_IVANOV_PRODUKTY", "KASPI_MAGAZIN_EDA"],
    "Транспорт":["YANDEX*TAXI ALMATY KAZ", "UBER *TRIP", "CITYBUS_PAYMENT", "ONAY_KAZ", "TAXI_REGION_02"],
    "Снятие наличных":["KASPI_ATM_WD", "HALYK BANK ATM ASTANA", "BCC_ATM_ALMATY", "ATM_WITHDRAWAL_KAZ", "JUSAN_ATM_001"],
    "Подписки":["NETFLIX.COM", "YANDEX*PLUS", "SPOTIFY_KAZ", "APPLE.COM/BILL", "KINOPOISK_SUB"],
    "Рестораны":["MCDONALDS_ALMATY", "KFC_ASTANA", "STARBUCKS_KAZ", "IP_KAFE_MANGAL", "NAVAT_RESTAURANT"]
}

data =[]
# Генерируем 1000 транзакций с небольшим шумом (добавляем случайные цифры)
for _ in range(200):
    for category, texts in templates.items():
        text = random.choice(texts) + " " + str(random.randint(1000, 9999))
        data.append({"text": text, "label": category})

random.shuffle(data)
df = pd.DataFrame(data)

# Создаем словари для перевода Текста в Цифры (Нейросети понимают только цифры)
labels = df['label'].unique()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

df['label_id'] = df['label'].map(label2id)

print("2. Загружаем Токенизатор (Переводчик слов в векторы)...")
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Превращаем DataFrame в формат Hugging Face Dataset
dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    # padding и truncation обрезают или удлиняют чеки до одинаковой длины (16 токенов)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=16)

print("3. Токенизируем данные...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Оставляем только нужные колонки
tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")
tokenized_datasets = tokenized_datasets.select_columns(["input_ids", "attention_mask", "labels"])

# Разбиваем на Train и Test (80/20)
split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("4. Загружаем базовую нейросеть DistilBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

print("5. Настраиваем параметры обучения (Fine-Tuning)...")
training_args = TrainingArguments(
    output_dir="./models/results",
    eval_strategy="epoch",       # <--- ИЗМЕНИЛИ СЮДА!
    learning_rate=2e-5,          
    per_device_train_batch_size=16,
    num_train_epochs=3,          
    weight_decay=0.01,
    logging_dir='./models/logs',
    logging_steps=10,
    save_strategy="no"           
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("🚀 СТАРТ ОБУЧЕНИЯ НЕЙРОСЕТИ! (Это может занять 2-5 минут, в зависимости от процессора)...")
trainer.train()

print("\n6. Сохраняем нашего банковского NLP-эксперта...")
os.makedirs('models/transaction_nlp', exist_ok=True)
# Сохраняем и саму модель, и токенизатор
model.save_pretrained('models/transaction_nlp')
tokenizer.save_pretrained('models/transaction_nlp')

print("✅ Нейросеть успешно обучена и сохранена!")