

# retrieval.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ячейка 2: Модифицированный класс с сохранением эмбеддингов
import pickle
import os

class WineRetriever:
    def __init__(self, csv_path, embeddings_path="wine_embeddings.pkl", force_recompute=False):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # Пробуем загрузить существующие эмбеддинги
        if os.path.exists(embeddings_path) and not force_recompute:
            print("🔄 Загружаем существующие эмбеддинги...")
            with open(embeddings_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.wine_embeddings = saved_data['embeddings']
                self.wine_descriptions = saved_data['descriptions']
            print(f"✅ Загружено {len(self.wine_embeddings)} эмбеддингов")
        else:
            # Создаем новые эмбеддинги
            print("🔄 Создаем эмбеддинги для вин...")
            self.wine_descriptions = self._create_wine_descriptions()
            self.wine_embeddings = self.model.encode(self.wine_descriptions, show_progress_bar=True)

            # Сохраняем для будущего использования
            saved_data = {
                'embeddings': self.wine_embeddings,
                'descriptions': self.wine_descriptions
            }
            with open(embeddings_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print(f"✅ Создано и сохранено {len(self.wine_embeddings)} эмбеддингов в {embeddings_path}")

    def _create_wine_descriptions(self):
        """Создаем текстовые описания из ваших колонок"""
        descriptions = []
        for _, row in self.df.iterrows():
            desc_parts = []

            # Используем доступные колонки
            if pd.notna(row.get('title')) and row.get('title'):
                desc_parts.append(f"Вино: {row['title']}")

            if pd.notna(row.get('variety')) and row.get('variety'):
                desc_parts.append(f"Сорт: {row['variety']}")

            if pd.notna(row.get('winery')) and row.get('winery'):
                desc_parts.append(f"Винодельня: {row['winery']}")

            if pd.notna(row.get('country')) and row.get('country'):
                desc_parts.append(f"Страна: {row['country']}")

            if pd.notna(row.get('region_1')) and row.get('region_1'):
                desc_parts.append(f"Регион: {row['region_1']}")

            if pd.notna(row.get('region_2')) and row.get('region_2'):
                desc_parts.append(f"Подрегион: {row['region_2']}")

            if pd.notna(row.get('province')) and row.get('province'):
                desc_parts.append(f"Провинция: {row['province']}")

            if pd.notna(row.get('price')) and row.get('price'):
                desc_parts.append(f"Цена: ${row['price']}")

            if pd.notna(row.get('description')) and row.get('description'):

                desc_parts.append(f"Описание: {row['description']}")

            if pd.notna(row.get('designation')) and row.get('designation'):
                desc_parts.append(f"Дизайнация: {row['designation']}")

            description = ". ".join(desc_parts)
            descriptions.append(description)

        return descriptions

    def retrieve(self, user_query, top_k=20):
        """Векторный поиск по запросу пользователя"""
        query_embedding = self.model.encode([user_query])[0]

        # Косинусная схожесть
        similarities = cosine_similarity([query_embedding], self.wine_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_wines = []
        for idx in top_indices:
            wine_data = self.df.iloc[idx].to_dict()
            wine_data['similarity_score'] = float(similarities[idx])
            retrieved_wines.append(wine_data)

        # Сортируем по схожести
        retrieved_wines.sort(key=lambda x: x['similarity_score'], reverse=True)

        return retrieved_wines

    def search_by_filters(self, query=None, variety=None, country=None, max_price=None, top_k=10):
        """
        Гибридный поиск: семантический + фильтры
        
        Note: This method has been converted to JavaScript for the web interface.
        The JavaScript version (searchByFiltersAsync) supports multiple varieties/countries
        via checkboxes, but maintains the same core logic:
        - Substring matching for variety/country (case-insensitive)
        - Price filtering (extended to support min_price in JS)
        - Semantic search when query is provided, otherwise neutral scores
        """
        if query:
            # Семантический поиск
            all_results = self.retrieve(query, top_k=50)
        else:
            # Если нет запроса, берем все вина
            all_results = [self.df.iloc[i].to_dict() for i in range(len(self.df))]
            for result in all_results:
                result['similarity_score'] = 0.5  # нейтральный скор

        # Применяем фильтры
        filtered_results = []
        for wine in all_results:
            # Variety filter: substring match (case-insensitive)
            # JS equivalent: varieties.some(v => wineVariety.includes(v.toLowerCase()))
            if variety and pd.notna(wine.get('variety')):
                if variety.lower() not in str(wine['variety']).lower():
                    continue

            # Country filter: substring match (case-insensitive)
            # JS equivalent: countries.some(c => wineCountry.includes(c.toLowerCase()))
            if country and pd.notna(wine.get('country')):
                if country.lower() not in str(wine['country']).lower():
                    continue

            # Price filter: max_price only (JS extends to min_price for UI)
            # JS equivalent: if (price < minPrice || price > maxPrice) return false
            if max_price and pd.notna(wine.get('price')):
                if wine['price'] > max_price:
                    continue

            filtered_results.append(wine)

        return filtered_results[:top_k]

# Ячейка 3: Первый запуск (сохранит эмбеддинги)
print("🚀 Первый запуск - создаем и сохраняем эмбеддинги...")
retriever = WineRetriever("df.csv", embeddings_path="my_wine_embeddings.pkl")

def visualize_wine_embeddings(retriever, query=None):
    """Визуализация эмбеддингов вин в 2D"""

    # Преобразуем эмбеддинги в numpy array если это список
    if isinstance(retriever.wine_embeddings, list):
        embeddings = np.array(retriever.wine_embeddings)
    else:
        embeddings = retriever.wine_embeddings

    print(f"📊 Визуализируем {len(embeddings)} эмбеддингов...")

    # Уменьшаем размерность для визуализации
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    # Цвета по странам
    countries = retriever.df['variety'].fillna('Unknown')
    unique_countries = countries.unique()

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_countries)))

    for i, country in enumerate(unique_countries):
        mask = countries == country
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   label=country, alpha=0.7, c=[colors[i]], s=30)

    if query:
        # Показываем запрос пользователя
        query_embedding = retriever.model.encode([query])[0]

        # Для query тоже используем numpy array
        query_2d = tsne.fit_transform(np.array([query_embedding]))

        plt.scatter(query_2d[0, 0], query_2d[0, 1],
                   marker='*', s=300, c='red', label=f'Запрос: "{query}"',
                   edgecolors='black', linewidth=2)

    plt.title("Визуализация вин в семантическом пространстве (t-SNE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return embeddings_2d

query = "rich red wine with notes of cherry and chocolate"
results = retriever.retrieve(query, top_k=5)

print(f"🔍 Results for: '{query}'\n")
for i, wine in enumerate(results, 1):
    print(f"{i}. {wine.get('title', 'Без названия')}")
    print(f"   Variety: {wine.get('variety', 'N/A')}")
    print(f"   Country {wine.get('country', 'N/A')}")
    print(f"   Price: ${wine.get('price', 'N/A')}")
    print(f"   Similarity: {wine['similarity_score']:.3f}")
    print()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "microsoft/Phi-3.5-mini-instruct"

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка модели
# device_map="cuda" автоматически переместит модель на доступный GPU


# Загрузка модели с trust_remote_code=False
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False # <-- ИЗМЕНИТЕ ЗДЕСЬ
)

# Сначала создадим пайплайн для генерации текста
phi_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate_explanation(query, wine, similarity_score):
    """Генерирует объяснение почему вино подходит под запрос"""

    prompt = f"""
You're an expert sommelier. Tell the user why this wine is perfect for his request.

User request: "{query}"

Wie info:
- Title: {wine.get('title', 'Не указано')}
- Variety: {wine.get('variety', 'Не указан')}
- Coutry: {wine.get('country', 'Не указана')}
- Region: {wine.get('region_1', 'Не указан')}
- Winary: {wine.get('winery', 'Не указана')}
- Price: ${wine.get('price', 'Не указана')}
- Description: {wine.get('description', 'Описание отсутствует')}
- Similarity: {similarity_score:.3f}

explanation (2-3 sentences, in english):
"""

    try:
        response = phi_pipeline(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        explanation = response[0]['generated_text'].replace(prompt, "").strip()
        return explanation

    except Exception as e:
        return f"Не удалось сгенерировать объяснение: {e}"

def format_long_text(text, width=70):
    """Форматирует длинный текст с переносами строк"""
    import textwrap
    return '\n   '.join(textwrap.wrap(text, width=width))

# Обновленная функция вывода результатов
def print_recommendations_with_explanations(query, results, retriever):
    """Печатает рекомендации с LLM-объяснениями"""

    print(f"🔍 Results for: '{query}'\n")
    print("=" * 80)

    for i, wine in enumerate(results, 1):
        print(f"\n{i}. 🍷 {wine.get('title', 'Без названия')}")
        print(f"   📍 Variety: {wine.get('variety', 'N/A')}")
        print(f"   🌍 Country: {wine.get('country', 'N/A')}")
        print(f"   💰 Price: ${wine.get('price', 'N/A')}")
        print(f"   ⭐ Similarity: {wine['similarity_score']:.3f}")

        # Генерируем и показываем объяснение
        print(f"\n   💡 Explanation:")
        explanation = generate_explanation(query, wine, wine['similarity_score'])

        # Форматируем объяснение с переносами
        formatted_explanation = format_long_text(explanation)
        print(f"   {formatted_explanation}")

        print("\n" + "-" * 80)

query = "rich red wine with notes of cherry and chocolate"
results = retriever.retrieve(query, top_k=5)
print_recommendations_with_explanations(query, results, retriever)

filtered_results = retriever.search_by_filters(
    query="rich red wine with notes of cherry and chocolate",
    variety="Red Blend",
    country="US",
    max_price=100,
    top_k=3
)

print("🎯 Results with filters:\n")
for i, wine in enumerate(filtered_results, 1):
    print(f"{i}. {wine.get('title', 'Без названия')}")
    print(f"   📍 {wine.get('country', 'N/A')} - {wine.get('region_1', 'N/A')}")
    print(f"   💰 ${wine.get('price', 'N/A')}")
    print(f"   ⭐ Similarity: {wine['similarity_score']:.3f}")
    print()

import pandas as pd
df=pd.read_csv('df.csv')

df['variety'].unique()

import numpy
print("Текущая версия numpy в Colab:", numpy.__version__)

# save_embeddings.py
import pickle
import json
import numpy as np

# Загрузите ваш pickle файл
with open("my_wine_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data['embeddings']
descriptions = data['descriptions']

# Конвертируем numpy в списки
if isinstance(embeddings, np.ndarray):
    embeddings_list = embeddings.tolist()
else:
    embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb
                      for emb in embeddings]

# Сохраняем оптимизированную версию
json_data = {
    "embeddings": embeddings_list,
    "count": len(embeddings_list),
    "dimension": len(embeddings_list[0]) if embeddings_list else 0
}

with open("wine_embeddings.json", "w") as f:
    json.dump(json_data, f)

print(f"✅ Сохранено {len(embeddings_list)} эмбеддингов в wine_embeddings.json")