import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1. Настройка страницы
st.set_page_config(page_title="NLP Категоризатор", page_icon="💳", layout="centered")
st.title("💳 ИИ-Категоризатор Транзакций")
st.write("Введите «грязный» текст банковского чека или выписки, и нейросеть (DistilBERT) определит его категорию.")

# 2. Загрузка тяжелой нейросети в кэш (чтобы не грузить её при каждом клике)
@st.cache_resource
def load_ai_model():
    model_path = "./models/transaction_nlp"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Переводим модель в режим предсказания (отключаем обучение)
    model.eval()
    return tokenizer, model

with st.spinner("Загрузка весов нейросети (DistilBERT)... 🧠"):
    tokenizer, model = load_ai_model()

# 3. Интерфейс (Поле ввода)
st.markdown("---")
sample_text = "YANDEX*TAXI ALMATY KAZ 4512"
user_input = st.text_input("Текст транзакции:", value=sample_text)

# 4. Логика предсказания
if st.button("🔍 Определить категорию", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст транзакции!")
    else:
        with st.spinner("Анализирую контекст слов..."):
            # А. Токенизация (переводим буквы в цифры для нейросети)
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=16)
            
            # Б. Инференс (прогон через нейросеть без обновления весов)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # В. Достаем сырые ответы (логиты) и превращаем их в вероятности (Softmax)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Г. Находим самый вероятный класс
            predicted_class_id = probabilities.argmax().item()
            predicted_label = model.config.id2label[predicted_class_id]
            confidence = probabilities[predicted_class_id].item()
            
            # 5. Красивый вывод результата
            st.markdown("### Результат:")
            
            # Выбираем цвет в зависимости от уверенности нейросети
            if confidence > 0.8:
                st.success(f"**Категория:** {predicted_label}")
            elif confidence > 0.5:
                st.warning(f"**Категория:** {predicted_label}")
            else:
                st.error(f"**Категория:** {predicted_label} (Неуверенное предсказание)")
                
            st.info(f"**Уверенность ИИ (Confidence):** {confidence:.2%}")