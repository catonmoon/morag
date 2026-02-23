# Современное состояние больших языковых моделей (LLM)

Большие языковые модели (Large Language Models, LLM) — это нейросетевые архитектуры на основе трансформеров, обученные на триллионах токенов текста. С 2022 года они стали центральным инструментом в задачах генерации текста, кода, анализа данных и построения диалоговых систем.

## Архитектурные основы

### Трансформер и механизм внимания

Все современные LLM основаны на архитектуре трансформера (Vaswani et al., 2017). Ключевой компонент — механизм **scaled dot-product attention**:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Где:
- `Q` — матрица запросов (queries)
- `K` — матрица ключей (keys)
- `V` — матрица значений (values)
- `d_k` — размерность ключей (используется для нормализации)

На практике применяется **multi-head attention** — несколько параллельных механизмов внимания, результаты которых конкатенируются:

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / self.d_k ** 0.5
        attn = scores.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_o(out)
```

### Позиционное кодирование

Трансформер не имеет встроенного понятия порядка токенов. Современные модели используют несколько подходов:

| Метод | Модели | Особенность |
|---|---|---|
| Sinusoidal PE | GPT-1, BERT | Фиксированное, не обучается |
| Learned PE | GPT-2, GPT-3 | Обучаемые эмбеддинги позиций |
| RoPE | LLaMA, Mistral, Qwen | Относительное, хорошо обобщается на длинный контекст |
| ALiBi | MPT, BLOOM | Линейное смещение, без дополнительных параметров |
| NoPE | некоторые исследования | Без позиционного кодирования вообще |

RoPE (Rotary Position Embedding) стал де-факто стандартом для open-source моделей благодаря возможности экстраполяции на контексты длиннее обучающих.

---

## Ключевые модели 2024–2025

### Закрытые модели (Proprietary)

На сегодняшний день наиболее мощными закрытыми моделями являются:

| Модель | Компания | Контекст | Особенности |
|---|---|---|---|
| GPT-4o | OpenAI | 128k токенов | Мультимодальная, голос, изображения |
| Claude 3.5 Sonnet | Anthropic | 200k токенов | Сильный coding, long context |
| Claude 3 Opus | Anthropic | 200k токенов | Лучшее рассуждение в семействе |
| Gemini 1.5 Pro | Google | 1M токенов | Рекордный контекст, мультимодальная |
| Gemini 2.0 Flash | Google | 1M токенов | Быстрая, инструментальная |
| o3 | OpenAI | — | Chain-of-thought, олимпиадные задачи |

### Открытые модели (Open Weights)

Экосистема open-source моделей резко выросла с выходом LLaMA 2 и LLaMA 3:

| Модель | Параметры | Лицензия | Сильные стороны |
|---|---|---|---|
| LLaMA 3.1 | 8B / 70B / 405B | Llama 3.1 Community | Общее назначение, instruction |
| Mistral 7B | 7B | Apache 2.0 | Компактная, быстрая |
| Mixtral 8x7B | ~47B (MoE) | Apache 2.0 | MoE-архитектура, многоязычность |
| Qwen2.5 | 0.5B–72B | Apache 2.0 | Сильный coding и math |
| DeepSeek-R1 | 7B–671B | MIT | Chain-of-thought рассуждение |
| Gemma 2 | 2B / 9B / 27B | Gemma ToU | Компактные, эффективные |
| Phi-4 | 14B | MIT | Microsoft, академические задачи |

### Русскоязычные модели

| Модель | Параметры | Организация | Примечание |
|---|---|---|---|
| FRIDA | — | SberAI | Эмбеддинги, не генеративная |
| Vikhr | 7B | Vikhr Team | Дообученный Mistral |
| ruGPT-3.5 | 13B | SberAI | Устаревшая, заменена GigaChat |
| GigaChat | — | Сбер | Закрытая, коммерческая |
| YandexGPT 5 | — | Яндекс | Закрытая, Алиса |

---

## Методы обучения

### Pre-training

Предобучение происходит на задаче предсказания следующего токена (Causal Language Modeling):

```
L = -sum( log P(x_t | x_1, ..., x_{t-1}) )
```

Объём данных для современных моделей:

| Модель | Токены предобучения |
|---|---|
| GPT-3 | 300B |
| LLaMA 1 | 1.4T |
| LLaMA 3 | 15T |
| Mistral 7B | ~8T (предположительно) |
| DeepSeek-V3 | 14.8T |

### Instruction Tuning и RLHF

После предобучения модель обучается следовать инструкциям:

1. **SFT (Supervised Fine-Tuning)** — дообучение на парах (инструкция → правильный ответ)
2. **RLHF (Reinforcement Learning from Human Feedback)** — обучение модели награды, затем PPO/GRPO
3. **DPO (Direct Preference Optimization)** — упрощённая альтернатива RLHF без отдельной reward model

```python
# Упрощённая иллюстрация DPO loss
def dpo_loss(
    policy_logprob_chosen: float,
    policy_logprob_rejected: float,
    ref_logprob_chosen: float,
    ref_logprob_rejected: float,
    beta: float = 0.1,
) -> float:
    delta_chosen = policy_logprob_chosen - ref_logprob_chosen
    delta_rejected = policy_logprob_rejected - ref_logprob_rejected
    return -torch.log(torch.sigmoid(beta * (delta_chosen - delta_rejected)))
```

### Quantization

Для запуска крупных моделей на потребительском железе применяется квантизация весов:

| Метод | Точность | Потеря качества | Библиотека |
|---|---|---|---|
| FP16 | 2 байта/параметр | Нет | transformers |
| GPTQ | 4 бит | Минимальная | auto-gptq |
| AWQ | 4 бит | Минимальная | autoawq |
| GGUF (Q4_K_M) | ~4.5 бит | Незначительная | llama.cpp |
| GGUF (Q8_0) | 8 бит | Практически нет | llama.cpp |
| bitsandbytes (NF4) | 4 бит | Незначительная | bitsandbytes |

Квантизация позволяет запустить модель LLaMA 3 70B в формате GGUF Q4_K_M примерно на 40 ГБ VRAM вместо 140 ГБ в FP16.

---

## RAG и работа с внешними знаниями

### Зачем нужен RAG

LLM обладают рядом фундаментальных ограничений:
- **Knowledge cutoff** — знания ограничены датой окончания обучения
- **Hallucinations** — модель может уверенно генерировать неверные факты
- **Context window** — невозможно загрузить весь корпус документов в контекст

RAG (Retrieval-Augmented Generation) решает эти проблемы, подтягивая релевантные фрагменты из внешней базы знаний в момент генерации ответа.

### Схема работы RAG

```
Запрос пользователя
       │
       ▼
 [Retriever]  ←──── Vector DB (Qdrant, Weaviate, Pinecone)
       │
       ▼ (топ-K чанков)
  [Reranker]  ──── опционально, LLM или cross-encoder
       │
       ▼ (отфильтрованный контекст)
  [Generator] ──── LLM с системным промптом
       │
       ▼
   Ответ
```

### Стратегии чанкинга

Качество RAG критически зависит от того, как документы разбиты на чанки:

| Стратегия | Описание | Плюсы | Минусы |
|---|---|---|---|
| Fixed-size | Разбивка по числу символов/токенов | Простота | Разрывает смысловые блоки |
| Sentence | По границам предложений | Семантическая целостность | Неравномерный размер |
| Recursive | Иерархическое разбиение по разделителям | Гибкость | Сложнее настраивать |
| Markdown-aware | По заголовкам и структуре MD | Сохраняет структуру документа | Только для структурированных текстов |
| Semantic | По семантической схожести предложений | Лучшее качество | Медленнее, требует эмбеддингов |

### Гибридный поиск

Только dense-векторов часто недостаточно. Современный подход — гибридный поиск:

```python
# Псевдокод гибридного поиска с RRF
def hybrid_search(query: str, limit: int = 10) -> list[Chunk]:
    dense_results = qdrant.search(
        collection="chunks",
        query_vector=embed_dense(query),
        limit=limit * 2,
    )
    sparse_results = qdrant.search(
        collection="chunks",
        query_vector=embed_sparse(query),
        limit=limit * 2,
    )
    # Reciprocal Rank Fusion
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
```

---

## Серверы для локального запуска

Для самостоятельного хостинга LLM существует несколько решений:

| Инструмент | Бэкенд | Особенности |
|---|---|---|
| Ollama | llama.cpp | Простой CLI, автоматическое управление моделями |
| llama.cpp server | llama.cpp | Минималистичный HTTP-сервер, GGUF |
| vLLM | CUDA | PagedAttention, высокая пропускная способность |
| TGI (HuggingFace) | PyTorch | Production-ready, continuous batching |
| LM Studio | llama.cpp | GUI, удобно для разработки |
| Jan | llama.cpp / cortex | Desktop-приложение |

### Пример запуска через Ollama

```bash
# Установка модели
ollama pull llama3.1:8b

# Запуск сервера (совместим с OpenAI API)
ollama serve

# Запрос через curl
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Объясни RAG простыми словами"}],
    "stream": false
  }'
```

### Пример через vLLM

```bash
# Запуск сервера
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --dtype bfloat16 \
    --max-model-len 8192

# Тест через Python-клиент
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Привет!"}],
)
print(response.choices[0].message.content)
```

---

## Метрики оценки

### Бенчмарки

| Бенчмарк | Что измеряет | Тип |
|---|---|---|
| MMLU | Академические знания, 57 дисциплин | Multiple choice |
| HumanEval | Генерация кода (Python) | Functional correctness |
| GSM8K | Математические задачи уровня школы | Chain-of-thought |
| MATH | Олимпиадная математика | Open-ended |
| GPQA | Аспирантский уровень науки | Multiple choice |
| MT-Bench | Диалог, следование инструкциям | LLM-as-judge |
| RULER | Длинный контекст (needle-in-haystack) | Retrieval |

### Метрики RAG

Для оценки RAG-пайплайнов применяются специализированные метрики:

- **Faithfulness** — насколько ответ соответствует предоставленному контексту
- **Answer Relevancy** — насколько ответ релевантен вопросу
- **Context Precision** — доля релевантных чанков среди retrieved
- **Context Recall** — доля необходимой информации, покрытой retrieved чанками

Инструменты автоматической оценки: **RAGAS**, **TruLens**, **DeepEval**.

---

## Тенденции 2025 года

### Reasoning Models

Появление моделей с встроенным chain-of-thought рассуждением (o1, o3, DeepSeek-R1, QwQ) изменило представление о возможностях LLM в задачах, требующих многошагового рассуждения. Эти модели «думают» перед ответом — генерируют внутренние рассуждения, которые могут занимать тысячи токенов.

### Multimodality

Граница между языковыми и мультимодальными моделями стирается. GPT-4o, Gemini и Claude 3 работают с текстом, изображениями, аудио и видео в едином интерфейсе.

### Small Language Models (SLM)

Растёт интерес к компактным моделям (1–7B параметров), способным работать на edge-устройствах и ноутбуках без GPU. Phi-4 (14B) показывает результаты, сопоставимые с более крупными моделями на ряде бенчмарков.

### Контекст как альтернатива RAG

Модели с контекстом 1M+ токенов (Gemini 1.5 Pro) поставили под вопрос необходимость RAG для ряда задач — весь корпус документов можно загрузить напрямую. Однако стоимость инференса и латентность пока делают RAG предпочтительным для production-систем.
