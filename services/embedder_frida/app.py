#!/usr/bin/env python3
"""FRIDA embedder server (OpenAI-compatible).

Универсальный сервер: работает и в Docker (CPU, offline), и нативно (MPS/CUDA).

Docker (через uvicorn app:app):
    Использует MODEL_PATH, TRANSFORMERS_OFFLINE, device=cpu.

Native:
    python services/embedder_frida/app.py                      # auto device
    python services/embedder_frida/app.py --device mps          # force MPS
    python services/embedder_frida/app.py --port 8092
    python services/embedder_frida/app.py --model ai-forever/FRIDA
"""
from __future__ import annotations

import argparse
import os
from base64 import b64encode
from typing import Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, list[str]]
    encoding_format: Optional[str] = 'float'  # 'float' | 'base64'
    user: Optional[str] = None


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def create_app(
    model_name: str | None = None,
    device: str | None = None,
) -> FastAPI:
    model_name = model_name or os.environ.get('MODEL_PATH', 'ai-forever/FRIDA')
    device = device or os.environ.get('DEVICE', _detect_device())
    local_files_only = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'

    application = FastAPI(title='FRIDA Embeddings (OpenAI-compatible)', version='1.0')

    print(f'Loading FRIDA model: {model_name} (device={device}, offline={local_files_only})')
    _model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    dim = _model.get_sentence_embedding_dimension()
    print(f'FRIDA model loaded: dim={dim}, device={device}')

    @application.get('/health')
    def health():
        return {'status': 'ok', 'model': model_name, 'device': device, 'dim': dim}

    @application.post('/v1/embeddings')
    def create_embeddings(req: EmbeddingsRequest):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts:
            raise HTTPException(400, 'input is empty')

        vecs = _model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)

        data = []
        for i, v in enumerate(vecs):
            if req.encoding_format == 'base64':
                payload = b64encode(v.astype(np.float32).tobytes()).decode('ascii')
            else:
                payload = v.tolist()
            data.append({'object': 'embedding', 'index': i, 'embedding': payload})

        return {
            'object': 'list',
            'data': data,
            'model': model_name,
            'usage': {'prompt_tokens': 0, 'total_tokens': 0},
        }

    return application


# Docker: uvicorn app:app
app = create_app()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRIDA embedder server')
    parser.add_argument('--model', default=None, help='Model name or path (default: ai-forever/FRIDA)')
    parser.add_argument('--device', default=None, help='Device: mps, cuda, cpu (auto if omitted)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    # CLI args перезаписывают app, созданный на уровне модуля
    app = create_app(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)