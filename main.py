import streamlit as st
import pandas as pd
import time
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from rake_nltk import Rake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import torch
import nltk

# Загрузка необходимых данных NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Инициализация модели SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Определение категорий для классификации сообщений
labels = [
    "Политика", "Спорт", "Повседневная жизнь", "Технологии", "Развлечения", "Бизнес", "Здоровье",
    "Образование", "Наука", "Культура", "Экономика", "Экология", "Путешествия", "История",
    "Психология", "Медицина", "Литература", "Музыка", "Кино", "Игры",
    "Еда и кулинария", "Автомобили", "Мода", "Строительство", "Архитектура",
    "Религия", "Философия", "Социальные проблемы", "Право", "Урбанистика",
    "Семья и отношения", "Животные", "Астрономия", "Космос", "Финансы", "Кибербезопасность",
    "Социальные сети", "Сельское хозяйство", "Энергетика", "Работа и карьера",
    "Продажи", "Объявления", "Образование", "война", "специальная военная операция"
]

# Кэширование кодировок категорий для ускорения
label_embeddings = model.encode(labels)

# Инициализация RAKE для извлечения ключевых слов
rake = Rake(language='russian')

# Инициализация суммаризатора LSA
summarizer = LsaSummarizer()


def classify_message(message: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    # Кодировка сообщения
    message_embedding = model.encode(message)

    # Вычисление косинусного сходства
    similarities = util.pytorch_cos_sim(message_embedding, label_embeddings)[0]

    # Получение лучших совпадений
    best_label_index = similarities.argmax().item()
    best_label = labels[best_label_index]
    best_score = similarities[best_label_index].item()

    # Получение топ-5 лучших совпадений
    top_5_indices = similarities.topk(5).indices.tolist()
    top_5_scores = similarities.topk(5).values.tolist()
    # Убираем вызов item() для float значений
    top_5_labels = [(labels[idx], score) for idx, score in zip(top_5_indices, top_5_scores)]

    return best_label, best_score, top_5_labels



def process_message(message: str) -> Tuple[str, str, float, List[Tuple[str, float]]]:
    # Классификация сообщения
    best_label, best_score, top_5_labels = classify_message(message)
    return message, best_label, best_score, top_5_labels


# Интерфейс Streamlit
st.title("Извлечение Тем из Сообщений")

# Ввод текста
text = st.text_area("Введите текст с новостями для анализа", "")

if st.button("Анализировать текст"):
    if text:
        start_time = time.time()
        results = []
        lines = text.split("\n")

        for line in lines:
            if line.strip():
                original_message, best_label, best_score, top_5_labels = process_message(line)

                # Если уверенность ниже 10%, показываем топ-5 тем
                if best_score < 0.10:
                    top_labels_str = ", ".join([f"{label} ({score:.2f})" for label, score in top_5_labels])
                    results.append((original_message, f"Неуверенно, топ-5 тем: {top_labels_str}"))
                else:
                    top_labels_str = ""  # Инициализация пустой строки, если не используется
                    results.append((original_message, f"{best_label} (уверенность: {best_score:.2f})"))

        end_time = time.time()

        # Отображение результатов
        df = pd.DataFrame(results, columns=['Сообщение', 'Предсказанная тема'])
        st.write(df)
        st.write(top_labels_str)
        st.write(f"Анализ завершён за {end_time - start_time:.2f} секунд.")

        # Возможность скачать результаты
        csv = df.to_csv(index=False)
        st.download_button(label="Скачать результаты", data=csv, file_name='classified_topics.csv')
    else:
        st.error("Введите текст для анализа.")

