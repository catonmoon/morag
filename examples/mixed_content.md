# Развёртывание ML-модели в production

Руководство описывает полный цикл вывода модели машинного обучения из Jupyter-ноутбука в production-окружение с мониторингом, масштабированием и откатом.

## Подготовка модели

Прежде чем разворачивать модель, необходимо сериализовать её в формат, пригодный для inference-сервера. Для моделей на PyTorch рекомендуется использовать TorchScript или ONNX, для TensorFlow — SavedModel. ONNX (Open Neural Network Exchange) обеспечивает переносимость между фреймворками и оптимизацию через ONNX Runtime.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


class TextClassifier(nn.Module):
    """Классификатор текста на основе LSTM с attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        lstm_out, _ = self.lstm(embedded)     # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)
        return self.classifier(context)  # (batch, num_classes)


def export_to_onnx(model: TextClassifier, save_path: Path, max_seq_len: int = 512) -> None:
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, max_seq_len))
    dummy_mask = torch.ones(1, max_seq_len)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(save_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch'},
        },
        opset_version=17,
    )
    print(f'Model exported to {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)')
```

В результате экспорта получается файл `.onnx`, который можно загрузить в ONNX Runtime без зависимости от PyTorch.

## Инфраструктура развёртывания

Типичная архитектура inference-сервиса включает несколько компонентов. Ниже приведена полная спецификация Kubernetes для развёртывания.

| Компонент | Описание | Ресурсы | Реплики | Health Check |
|---|---|---|---|---|
| inference-server | ONNX Runtime или Triton Inference Server | 4 CPU, 8 GB RAM, 1 GPU (опционально) | 2-8 (HPA) | /v2/health/ready |
| api-gateway | FastAPI + uvicorn, маршрутизация и валидация | 2 CPU, 2 GB RAM | 3 | /health |
| model-registry | MLflow Model Registry, версионирование артефактов | 2 CPU, 4 GB RAM | 1 | /api/2.0/mlflow/health |
| feature-store | Redis + Feast, кэш признаков для real-time inference | 4 CPU, 16 GB RAM | 3 (cluster) | PING |
| monitoring | Prometheus + Grafana, метрики латентности и качества | 2 CPU, 4 GB RAM | 1 | /-/healthy |
| queue | RabbitMQ или Kafka, для batch и async inference | 2 CPU, 4 GB RAM | 3 (cluster) | rabbitmqctl status |
| cache | Redis, кэш предсказаний для дедупликации запросов | 2 CPU, 8 GB RAM | 3 (sentinel) | PING |
| log-collector | Fluentd / Vector, сбор логов для аудита предсказаний | 1 CPU, 1 GB RAM | DaemonSet | /api/v1/health |

## Canary deployment

При обновлении модели критически важно не сломать production. Стратегия canary deployment позволяет постепенно переключать трафик на новую версию, контролируя метрики качества на каждом этапе. Процесс начинается с развёртывания новой версии модели рядом со старой — обе работают одновременно. Ingress-контроллер (или service mesh вроде Istio) маршрутизирует определённый процент трафика на новую версию: сначала 5%, затем 10%, 25%, 50% и наконец 100%. На каждом этапе автоматическая система мониторинга сравнивает ключевые метрики: p50/p95/p99 латентность inference, процент ошибок (4xx, 5xx), метрики качества модели (accuracy, F1 если доступна ground truth), потребление ресурсов (CPU, GPU, память). Если любая метрика деградирует сверх установленного порога, происходит автоматический откат. Для реализации используется Argo Rollouts — Kubernetes-контроллер, расширяющий стандартный Deployment прогрессивной доставкой. В отличие от стандартного RollingUpdate, Argo Rollouts поддерживает точное управление процентом трафика, паузы между шагами для анализа метрик, интеграцию с Prometheus для автоматического анализа, и автоматический откат при обнаружении аномалий. Альтернативный подход — shadow deployment, когда новая модель получает копию всего production-трафика, но её ответы не отправляются пользователям, а только логируются для сравнения со старой моделью. Это безопасно, но требует двойных вычислительных ресурсов и не выявляет проблемы с реальными побочными эффектами предсказаний.

```yaml
# argo-rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: ml-inference
  namespace: ml-production
spec:
  replicas: 6
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: ml-inference
  strategy:
    canary:
      canaryService: ml-inference-canary
      stableService: ml-inference-stable
      trafficRouting:
        istio:
          virtualServices:
            - name: ml-inference-vsvc
              routes:
                - primary
      steps:
        - setWeight: 5
        - pause: {duration: 10m}
        - analysis:
            templates:
              - templateName: latency-check
              - templateName: error-rate-check
        - setWeight: 25
        - pause: {duration: 15m}
        - analysis:
            templates:
              - templateName: latency-check
              - templateName: error-rate-check
              - templateName: quality-check
        - setWeight: 50
        - pause: {duration: 30m}
        - analysis:
            templates:
              - templateName: full-analysis
        - setWeight: 100
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
        - name: inference
          image: registry.example.com/ml-inference:v2.3.0
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 30
          env:
            - name: MODEL_VERSION
              value: "2.3.0"
            - name: ONNX_RUNTIME_THREADS
              value: "4"
            - name: MAX_BATCH_SIZE
              value: "32"
            - name: TIMEOUT_MS
              value: "5000"
```

## Мониторинг качества модели

После развёртывания необходимо постоянно отслеживать качество предсказаний. В отличие от классического мониторинга ПО, ML-мониторинг включает специфические метрики: drift входных данных (изменение распределения признаков), drift предсказаний (изменение распределения выходов модели), и деградация качества (если доступна отложенная обратная связь).

| Метрика | Описание | Порог алерта | Grafana panel | Prometheus query |
|---|---|---|---|---|
| p50_latency_ms | Медианная латентность inference | > 100ms | Inference Latency | `histogram_quantile(0.5, rate(inference_duration_seconds_bucket[5m]))` |
| p99_latency_ms | 99-й перцентиль латентности | > 500ms | Inference Latency | `histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))` |
| error_rate | Процент ошибок (5xx + timeout) | > 1% | Error Rate | `rate(inference_errors_total[5m]) / rate(inference_requests_total[5m])` |
| prediction_drift | KL-divergence распределения предсказаний | > 0.1 | Prediction Distribution | custom (Python exporter) |
| feature_drift | PSI (Population Stability Index) входных признаков | > 0.2 | Feature Drift | custom (Python exporter) |
| gpu_utilization | Загрузка GPU (inference) | < 20% или > 90% | GPU Metrics | `nvidia_gpu_duty_cycle` |
| batch_queue_depth | Глубина очереди batch-inference | > 1000 | Queue Metrics | `rabbitmq_queue_messages{queue="inference"}` |
| cache_hit_rate | Процент попаданий в кэш предсказаний | < 30% | Cache Performance | `rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])` |
| model_memory_mb | Потребление памяти моделью | > 7000MB | Resource Usage | `process_resident_memory_bytes{job="inference"}` |
| throughput_rps | Пропускная способность (requests/sec) | < 100 rps | Throughput | `rate(inference_requests_total[1m])` |

## A/B тестирование моделей

Для статистически значимого сравнения моделей используется A/B тестирование с контрольной группой. Трафик разделяется по user_id (consistent hashing), чтобы один пользователь всегда видел одну версию модели.

```python
import hashlib
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Конфигурация A/B-эксперимента."""
    name: str
    control_model: str          # ID модели-контроля
    treatment_model: str        # ID модели-кандидата
    traffic_split: float        # доля трафика на treatment (0.0–1.0)
    min_samples: int = 1000     # минимальное число наблюдений для теста
    significance_level: float = 0.05  # уровень значимости


def assign_variant(user_id: str, experiment: str, split: float) -> str:
    """Детерминистичное назначение варианта по user_id."""
    key = f'{experiment}:{user_id}'.encode()
    hash_value = int(hashlib.sha256(key).hexdigest(), 16) % 10000
    return 'treatment' if hash_value < split * 10000 else 'control'


def analyze_experiment(
    control_metrics: list[float],
    treatment_metrics: list[float],
    significance_level: float = 0.05,
) -> dict:
    """Статистический анализ A/B-эксперимента."""
    control = np.array(control_metrics)
    treatment = np.array(treatment_metrics)

    # Welch's t-test (не предполагает равенства дисперсий)
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)

    # Размер эффекта (Cohen's d)
    pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
    cohens_d = (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0

    # Доверительный интервал для разницы средних
    diff = treatment.mean() - control.mean()
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    return {
        'control_mean': float(control.mean()),
        'treatment_mean': float(treatment.mean()),
        'absolute_diff': float(diff),
        'relative_diff_pct': float(diff / control.mean() * 100) if control.mean() != 0 else 0,
        'p_value': float(p_value),
        'significant': p_value < significance_level,
        'cohens_d': float(cohens_d),
        'confidence_interval': (float(ci_low), float(ci_high)),
        'control_n': len(control),
        'treatment_n': len(treatment),
    }
```

## Rollback-процедура

Если мониторинг обнаруживает деградацию, необходим быстрый откат. Процедура зависит от стратегии развёртывания. При canary deployment Argo Rollouts автоматически откатывает при провале analysis step. При blue-green достаточно переключить service selector обратно на старую версию. При прямом deployment — стандартный `kubectl rollout undo`. Во всех случаях критически важно сохранить логи предсказаний проблемной версии для post-mortem анализа. Также необходимо обновить model registry: пометить версию как «failed» с указанием причины, чтобы она не была случайно развёрнута повторно. Автоматическое оповещение через PagerDuty или Opsgenie должно включать: какая модель откатилась, какие метрики деградировали, сколько пользователей было затронуто, и ссылку на dashboard с детализацией.

```bash
# Ручной откат через Argo Rollouts
kubectl argo rollouts undo ml-inference -n ml-production

# Проверка статуса
kubectl argo rollouts status ml-inference -n ml-production

# Пометка версии как failed в MLflow
mlflow models update-model-version \
  --name text-classifier \
  --version 23 \
  --description "Rolled back: p99 latency increased 3x, error rate 5.2%"
```

## Чеклист перед релизом

Прежде чем отправлять новую версию модели в production, убедитесь что все пункты выполнены:

1. Модель прошла все offline-тесты (accuracy, F1, AUC на тестовой выборке)
2. Модель протестирована на edge cases и adversarial inputs
3. ONNX-экспорт валиден (`onnxruntime.InferenceSession` загружает без ошибок)
4. Latency benchmark пройден (p99 < SLA на target hardware)
5. Memory footprint в пределах лимита (< requested memory в k8s)
6. Model card заполнена (описание, метрики, ограничения, этические соображения)
7. A/B-эксперимент спланирован (гипотеза, метрики, размер выборки, длительность)
8. Rollback-процедура задокументирована и протестирована
9. Мониторинг настроен (Prometheus exporter, Grafana dashboard, алерты)
10. Data pipeline проверен (feature store актуален, нет schema drift)
