# Как устроены LLM изнутри

Этот документ разбирает внутреннее устройство больших языковых моделей: архитектуру трансформерного блока, токенизацию, обучающие данные и процесс обучения — от сырого текста до instruction-following модели.

---

## Токенизация

Прежде чем текст попадёт в модель, он преобразуется в последовательность целочисленных токенов. Токен — не всегда слово: это подслово (subword), определяемое алгоритмом.

### BPE (Byte Pair Encoding)

Самый распространённый алгоритм. Используется в GPT-2/3/4, LLaMA, Qwen и большинстве современных моделей.

**Принцип построения словаря:**
1. Начать с отдельных байт (256 символов)
2. Найти наиболее часто встречающуюся пару соседних токенов
3. Слить эту пару в новый токен
4. Повторять до достижения нужного размера словаря

```python
# Пример: как BPE токенизирует русский текст
# Словарь tiktoken (cl100k_base, 100k токенов):

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

text = "Большие языковые модели работают на трансформерах"
tokens = enc.encode(text)
print(tokens)
# [6395, 69811, 13187, 21068, 8891, 45021, 357, 2484, 90611, 268, 15473]

# Декодирование каждого токена
for t in tokens:
    print(repr(enc.decode([t])))
# 'Б' 'ольш' 'ие' ' язык' 'овые' ' модели' ' работ' 'ают' ' на' ' трансформ' 'ерах'
```

Русский текст токенизируется менее эффективно, чем английский: в среднем 1 слово ≈ 2–3 токена (vs ~1.3 для английского). Это влияет на стоимость инференса и размер контекста.

### Размеры словарей

| Модель | Словарь | Алгоритм |
|---|---|---|
| GPT-2 | 50 257 | BPE |
| GPT-3 / GPT-4 | 100 256 | BPE (tiktoken cl100k) |
| LLaMA 1/2 | 32 000 | BPE (sentencepiece) |
| LLaMA 3 | 128 256 | BPE (tiktoken) |
| Qwen2.5 | 151 646 | BPE |
| Mistral | 32 000 | BPE (sentencepiece) |
| DeepSeek-V3 | 129 280 | BPE |

Больший словарь → меньше токенов на тот же текст, но больше параметров в embedding-слое.

---

## Архитектура трансформерного блока

Современный decoder-only трансформер (GPT-подобный) состоит из стека идентичных блоков:

```
Input Tokens
     │
     ▼
[Token Embedding] + [Position Embedding]
     │
     ▼  ×N блоков
┌────────────────────────────────────┐
│  LayerNorm (Pre-Norm)              │
│       │                           │
│  Multi-Head Attention              │
│       │                           │
│  + Residual connection             │
│       │                           │
│  LayerNorm (Pre-Norm)              │
│       │                           │
│  Feed-Forward Network (MLP)        │
│       │                           │
│  + Residual connection             │
└────────────────────────────────────┘
     │
     ▼
[LayerNorm]
     │
     ▼
[LM Head: Linear → Logits → Softmax]
     │
     ▼
Next Token Probabilities
```

### Pre-Norm vs Post-Norm

Классический трансформер (Vaswani 2017) использовал **Post-Norm**: LayerNorm после residual. Современные модели используют **Pre-Norm**: LayerNorm перед подслоем.

```python
# Post-Norm (оригинальный трансформер)
x = LayerNorm(x + Attention(x))

# Pre-Norm (GPT-2 и далее)
x = x + Attention(LayerNorm(x))
```

Pre-Norm обеспечивает более стабильное обучение на больших глубинах — без него градиенты взрываются при N > 24 слоёв.

### RMSNorm вместо LayerNorm

LLaMA, Mistral, Qwen и большинство современных моделей заменили LayerNorm на **RMSNorm** — упрощённую версию без вычисления среднего:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
```

RMSNorm быстрее и при этом не уступает LayerNorm по качеству.

---

## Feed-Forward Network (MLP)

FFN занимает ~⅔ параметров модели. В стандартном трансформере:

```python
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

Типичное соотношение `d_ff = 4 × d_model`.

### SwiGLU — стандарт для современных моделей

LLaMA, Mistral, Qwen и большинство open-source моделей используют **SwiGLU** (Shazeer, 2020):

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down  = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

SwiGLU добавляет третью матрицу (`gate`), но при меньшем `d_ff` (≈ 8/3 × d_model вместо 4×) достигает лучшего качества.

### Mixture of Experts (MoE)

В MoE-архитектуре FFN заменяется набором «экспертов» — отдельных MLP. Для каждого токена активируются только K из N экспертов (sparse activation):

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts: int, top_k: int, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(num_experts)])
        self.router  = nn.Linear(d_model, num_experts)
        self.top_k   = top_k

    def forward(self, x):
        # x: (batch, seq, d_model)
        logits = self.router(x)                    # (batch, seq, num_experts)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[..., i]           # какой эксперт
            expert_w   = weights[..., i:i+1]       # его вес
            # применяем нужного эксперта к каждому токену
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    out[mask] += expert_w[mask] * self.experts[e](x[mask])
        return out
```

| Модель | Всего экспертов | Активных (top-K) | Итого параметров |
|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | ~47B (активных ~13B) |
| DeepSeek-V3 | 256 | 8 | 671B (активных ~37B) |
| Qwen1.5-MoE | 64 | 4 | ~14.3B |
| Grok-1 | 8 | 2 | 314B |

---

## Grouped Query Attention (GQA)

В стандартном MHA каждая голова имеет свои Q, K, V. При большом числе голов K и V занимают много памяти в KV-cache.

**GQA** (используется в LLaMA 3, Mistral, Qwen2.5): несколько Q-голов разделяют одну пару K/V:

```
MHA:  Q1K1V1  Q2K2V2  Q3K3V3  Q4K4V4   ← 4 независимых головы
MQA:  Q1      Q2      Q3      Q4        ← 4 Q-головы
       └───────K1V1───────┘              ← 1 общая K/V пара
GQA:  Q1Q2    Q3Q4                      ← 2 группы
       K1V1   K2V2                       ← 2 K/V пары
```

GQA снижает размер KV-cache в `num_heads / num_kv_heads` раз без значимой потери качества.

| Модель | Heads (Q) | KV Heads | Сжатие KV-cache |
|---|---|---|---|
| LLaMA 2 7B | 32 | 32 (MHA) | 1× |
| LLaMA 3 8B | 32 | 8 (GQA) | 4× |
| Mistral 7B | 32 | 8 (GQA) | 4× |
| Qwen2.5 72B | 64 | 8 (GQA) | 8× |

---

## Данные для предобучения

### Основные источники

| Источник | Описание | Доля в типичном датасете |
|---|---|---|
| CommonCrawl | Веб-краулинг всего интернета | 60–80% |
| GitHub | Исходный код (все языки) | 5–15% |
| Wikipedia | Энциклопедические статьи | 1–5% |
| Books / Project Gutenberg | Художественная и научная литература | 3–10% |
| ArXiv | Научные статьи (PDF → текст) | 1–3% |
| StackExchange | Вопросы/ответы по техническим темам | 1–2% |
| News | Новостные агрегаторы (Reuters, CNN и т.д.) | 1–5% |

### Обработка и фильтрация

Сырой веб-текст требует масштабной обработки перед обучением:

```
CommonCrawl (сотни ТБ)
       │
       ▼
[Language Detection]    ← fasttext, удаляем нецелевые языки
       │
       ▼
[Quality Filtering]     ← эвристики: длина, доля букв, punct ratio
       │
       ▼
[Deduplication]         ← MinHash LSH: убираем почти-дубликаты
       │
       ▼
[Safety Filtering]      ← удаление порнографии, hate speech, PII
       │
       ▼
[Domain Upsampling]     ← Wikipedia, Books умножаются в 3–10×
       │
       ▼
Финальный корпус (~1–15T токенов)
```

**Дедупликация критически важна**: без неё модель переобучается на частых шаблонах и генерирует их дословно. LLaMA использовала MinHash с 9000 хэш-функциями, оставив только уникальные 13-граммы.

### Данные для кода

Качество на coding-задачах резко растёт при включении кода в предобучение. StarCoder Dataset (The Stack) включает 86 языков программирования, отфильтрованных по лицензии.

```
GitHub (2TB raw)
       │
       ▼
[License Filter]         ← только MIT, Apache 2.0, BSD и т.д.
       │
       ▼
[Near-dedup by file]     ← убираем fork-копии
       │
       ▼
[Quality heuristics]     ← avg line length, alphanum ratio
       │
       ▼
The Stack (~300GB)
```

---

## Процесс обучения

### Распределённое обучение

Модели с миллиардами параметров не помещаются на одну GPU. Применяется комбинация трёх видов параллелизма:

| Тип | Что разбивается | Пример |
|---|---|---|
| Data Parallelism (DP) | Батч данных | 8 GPU, каждая обрабатывает 1/8 батча |
| Tensor Parallelism (TP) | Веса матриц (по строкам/столбцам) | Attention-матрица разбита по головам |
| Pipeline Parallelism (PP) | Слои модели | GPU 1: слои 1–8, GPU 2: слои 9–16 |

DeepSeek-V3 (671B) обучался на 2048 GPU H800 с комбинацией DP + TP + PP + Expert Parallelism.

### Оптимизатор и гиперпараметры

Стандарт для LLM — **AdamW** с decoupled weight decay:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # пиковый LR, затем косинусный decay
    betas=(0.9, 0.95), # β2 = 0.95 вместо стандартного 0.999
    eps=1e-8,
    weight_decay=0.1,
)
```

**Learning rate schedule:**
```
LR
 │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │   ╱ warmup                 ╲ cosine decay
 │  ╱                          ╲___________
 │_╱
 └────────────────────────────────── шаги
```

- Warmup: 1000–4000 шагов линейно до пикового LR
- Decay: косинусный до 10% пикового LR

**Gradient clipping**: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` — обязателен для стабильности.

### Закон масштабирования Chinchilla

В 2022 году DeepMind (Hoffmann et al.) установили оптимальное соотношение между числом параметров (N) и числом токенов обучения (D):

```
D_optimal = 20 × N
```

То есть модель на 7B параметров нужно обучать минимум на 140B токенов. LLaMA 3 намеренно нарушила это правило, обучив 8B-модель на 15T токенов — это «over-trained» модель, оптимальная для инференса, а не для flops обучения.

| Модель | Параметры | Токены обучения | Chinchilla-оптимум |
|---|---|---|---|
| GPT-3 | 175B | 300B | 3.5T |
| LLaMA 1 (7B) | 7B | 1T | 140B |
| LLaMA 3 (8B) | 8B | 15T | 160B |
| DeepSeek-V3 | 671B | 14.8T | 13.4T |

---

## От предобученной к instruction-following модели

### Этап 1: SFT (Supervised Fine-Tuning)

После предобучения модель умеет «продолжать текст», но не «отвечать на вопросы». SFT обучает её на демонстрациях правильного поведения:

```
Пример SFT-пары:

System: Ты полезный AI-ассистент.
User:   Объясни, что такое градиентный спуск.
Assistant: Градиентный спуск — это итеративный алгоритм оптимизации...
```

SFT-датасеты: OpenAssistant, Alpaca, ShareGPT, Dolly и тысячи специализированных.

### Этап 2: RLHF

**Reward Model (RM)** обучается на предпочтениях людей:

```
Для одного промпта генерируются 2 ответа:
  - Ответ A: "Для сортировки используй bubble sort"
  - Ответ B: "Для больших массивов лучше quicksort O(n log n)"

Люди-аннотаторы выбирают: B лучше
RM учится: score(B) > score(A)
```

Затем **PPO** (Proximal Policy Optimization) обновляет модель, максимизируя награду от RM с KL-штрафом за отклонение от базовой модели:

```
L = E[RM(response)] - β × KL(policy || ref_policy)
```

### Этап 3: DPO (современная альтернатива)

DPO убирает отдельную reward model — модель обучается напрямую на предпочтениях:

```python
def dpo_loss(
    π_logprob_chosen: torch.Tensor,    # log P_policy(y_w | x)
    π_logprob_rejected: torch.Tensor,  # log P_policy(y_l | x)
    ref_logprob_chosen: torch.Tensor,  # log P_ref(y_w | x)
    ref_logprob_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    log_ratio_chosen   = π_logprob_chosen   - ref_logprob_chosen
    log_ratio_rejected = π_logprob_rejected - ref_logprob_rejected
    return -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
```

DPO используется в LLaMA 3, Qwen2.5, Mistral Instruct и большинстве современных open-source моделей.

### Этап 4: Reasoning (Chain-of-Thought)

Модели вроде DeepSeek-R1 и o3 добавляют этап обучения на синтетических цепочках рассуждений. Модель генерирует `<think>...</think>` блок перед финальным ответом:

```
User: Сколько 'r' в слове "strawberry"?

<think>
Разобью слово на буквы: s-t-r-a-w-b-e-r-r-y
Считаю 'r': позиция 3, позиция 8, позиция 9
Итого: 3
</think>

В слове "strawberry" три буквы 'r'.
```

---

## KV-Cache и инференс

При генерации модель авторегрессивно предсказывает по одному токену. Без оптимизации каждый шаг требует O(N²) вычислений — пересчёт attention по всей истории.

**KV-Cache** сохраняет уже вычисленные Key и Value для каждого слоя:

```python
# Без KV-cache: O(T²) на T токенов
for t in range(T):
    K = W_k @ x[:t+1]   # пересчитываем всю историю
    V = W_v @ x[:t+1]
    attn = softmax(Q @ K.T / sqrt(d_k)) @ V

# С KV-cache: O(T) после prefill
cache_k, cache_v = [], []
for t in range(T):
    k_t = W_k @ x[t]
    v_t = W_v @ x[t]
    cache_k.append(k_t)
    cache_v.append(v_t)
    K = torch.stack(cache_k)   # только append
    V = torch.stack(cache_v)
    attn = softmax(Q_t @ K.T / sqrt(d_k)) @ V
```

Размер KV-cache для одного запроса:
```
2 × num_layers × num_kv_heads × head_dim × seq_len × sizeof(dtype)

Пример: LLaMA 3 8B, 8192 токенов, bf16:
2 × 32 × 8 × 128 × 8192 × 2 = 1 073 741 824 байт ≈ 1 ГБ
```

### Стратегии сэмплинга

Из логитов модели получают распределение вероятностей. Следующий токен выбирается по одной из стратегий:

| Стратегия | Параметры | Когда использовать |
|---|---|---|
| Greedy | — | Детерминированность, но повторяемость |
| Temperature | `T ∈ (0, 2]` | T<1: консервативнее, T>1: креативнее |
| Top-K | `K = 40–100` | Отрезает токены с низкой вероятностью |
| Top-P (Nucleus) | `P = 0.9–0.95` | Адаптивный отбор по кумулятивной P |
| Min-P | `min_p = 0.05` | Относительный порог (от макс. вероятности) |

```python
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> int:
    logits = logits / temperature
    # Top-K
    if top_k > 0:
        top_k_values = torch.topk(logits, top_k).values
        logits[logits < top_k_values[-1]] = -float('inf')
    # Top-P (Nucleus)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()
    # Сэмплирование
    return sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item()
```

---

## Масштаб: сравнение архитектурных параметров

| Модель | Слои | d_model | Heads | d_ff | Параметры |
|---|---|---|---|---|---|
| GPT-2 (small) | 12 | 768 | 12 | 3072 | 117M |
| GPT-2 (XL) | 48 | 1600 | 25 | 6400 | 1.5B |
| LLaMA 3 8B | 32 | 4096 | 32 | 14336 | 8B |
| LLaMA 3 70B | 80 | 8192 | 64 | 28672 | 70B |
| Mistral 7B | 32 | 4096 | 32 | 14336 | 7B |
| Qwen2.5 72B | 80 | 8192 | 64 | 29568 | 72B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |

Число параметров FFN ≈ `2 × num_layers × d_model × d_ff` (для SwiGLU × 3).
Число параметров Attention ≈ `num_layers × 4 × d_model²`.

---

## Эмбеддинги и lm_head

Первый и последний слой модели делят веса (weight tying):

```python
# Embedding: token_id → вектор
embed = nn.Embedding(vocab_size, d_model)          # (V, D)

# LM Head: вектор → логиты по словарю
lm_head = nn.Linear(d_model, vocab_size, bias=False)  # (D, V)

# Weight tying: они используют одну матрицу транспонированно
lm_head.weight = embed.weight
```

Это экономит ~`vocab_size × d_model × 2` параметров. Для LLaMA 3 (vocab=128k, d=4096): экономия ~1B параметров.